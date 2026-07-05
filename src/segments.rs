//! 标点切句状态机：监听流式 ASR partial，按优先级做 checkpoint / reset 切句。
//!
//! 四级优先级（从高到低）：
//! 1. VAD 静音收尾 → 提交剩余尾巴 + 重置 stream
//! 2. 稳定句末标点 → checkpoint 切句（不重建 stream）
//! 3. 无标点 10s 兜底 → checkpoint 切句
//! 4. 25s 极端强切 → 重置 stream
//!
//! 本次只实现 B 推荐参数组（加强版 Zipformer-en）的预设，不泛化面板。

use std::time::Instant;

// ─── 配置 ────────────────────────────────────────────────────────────────

/// 切句策略参数（字段用 f32/u64，不用 Duration，方便 `const` 初始化）。
#[derive(Clone, Debug)]
pub(crate) struct SegConfig {
    pub(crate) name: &'static str,

    // VAD（收尾信号）
    pub(crate) vad_min_silence_sec: f32,
    pub(crate) vad_threshold: f32,
    pub(crate) vad_max_speech_sec: f32,

    // 句末标点切句
    pub(crate) punct_stability_sec: f32,
    pub(crate) punct_cooldown_sec: f32,
    pub(crate) punct_min_new_words: usize,

    // 无标点兜底
    pub(crate) no_punct_timeout_sec: f32,
    pub(crate) no_punct_stability_sec: f32,
    pub(crate) no_punct_min_words: usize,
    pub(crate) no_punct_keep_last_words: usize,

    // 极端兜底
    pub(crate) extreme_timeout_sec: f32,
}

/// 连接词/功能词列表：切点结尾命中则不切。
const CONNECTION_WORDS: &[&str] = &[
    "and", "a", "an", "the", "this", "these", "those", "my", "your", "his", "her", "their",
    "our", "its", "or", "but", "for", "to", "with", "because", "that", "of", "in", "on", "at",
    "from", "into", "until", "if", "when", "while", "as", "so",
];

/// 句末标点（切句信号）
const SENTENCE_ENDINGS: &[char] = &['.', '?', '!'];

// ─── B 推荐预设 ──────────────────────────────────────────────────────────

pub(crate) const SEG_CONFIG_ENHANCED_ZIPFORMER_EN: SegConfig = SegConfig {
    name: "加强版Zipformer-en",
    vad_min_silence_sec: 0.3,
    vad_threshold: 0.5,
    vad_max_speech_sec: 18.0,
    punct_stability_sec: 0.7,
    punct_cooldown_sec: 1.5,
    punct_min_new_words: 3,
    no_punct_timeout_sec: 10.0,
    no_punct_stability_sec: 1.0,
    no_punct_min_words: 12,
    no_punct_keep_last_words: 3,
    extreme_timeout_sec: 25.0,
};

// ─── 切句动作 ────────────────────────────────────────────────────────────

#[derive(Debug)]
pub(crate) enum SegmentAction {
    /// Checkpoint 切句：提交 `text`，不重建 stream。`draft` 是剩余草稿。
    Checkpoint { text: String, draft: String },
    /// VAD 检测到静音：提交尾巴 `text`，需要重置 stream。
    VadReset { text: String },
    /// 25s 极端强切：提交全部 `text`，需要重置 stream。
    ExtremeReset { text: String },
}

// ─── 统计 ────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Default)]
pub(crate) struct SegStats {
    /// partial 中出现句末标点的次数
    pub(crate) punct_candidate_count: usize,
    /// 稳定后成功按标点切句的次数
    pub(crate) punct_success_count: usize,
    /// 标点候选出现后、稳定前消失或位置变化
    pub(crate) punct_regret_count: usize,
    /// 10s 无标点兜底触发次数
    pub(crate) no_punct_fallback_count: usize,
    /// 25s 极端强切次数
    pub(crate) extreme_cut_count: usize,
    /// 句末标点首次出现到提交的累计延迟（ms）
    pub(crate) total_cut_latency_ms: u64,
    /// 所有提交文本的累计词数
    pub(crate) total_words_sum: usize,
    /// 累计提交段数
    pub(crate) total_segment_count: usize,
    /// 已提交文本又出现在草稿里的次数（重复风险）
    pub(crate) repeat_risk_count: usize,
    /// checkpoint 拼接文本和原 partial 明显缺词的次数
    pub(crate) missing_word_risk_count: usize,
}

impl SegStats {
    pub(crate) fn avg_cut_latency_ms(&self) -> f64 {
        if self.total_segment_count == 0 {
            return 0.0;
        }
        self.total_cut_latency_ms as f64 / self.total_segment_count as f64
    }

    pub(crate) fn avg_words_per_segment(&self) -> f64 {
        if self.total_segment_count == 0 {
            return 0.0;
        }
        self.total_words_sum as f64 / self.total_segment_count as f64
    }

    pub(crate) fn summary_line(&self) -> String {
        format!(
            "标点切 {}/{} | 兜底 {} | 强切 {} | 反悔 {} | 平均延迟 {:.1}s | 平均词数 {:.0} | 重复风险 {} | 漏词风险 {}",
            self.punct_success_count,
            self.punct_candidate_count,
            self.no_punct_fallback_count,
            self.extreme_cut_count,
            self.punct_regret_count,
            self.avg_cut_latency_ms() / 1000.0,
            self.avg_words_per_segment(),
            self.repeat_risk_count,
            self.missing_word_risk_count,
        )
    }
}

// ─── 内部候选 ────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct PunctCandidate {
    /// 标点在 partial 中的字节偏移
    byte_pos: usize,
    /// 首次出现在该位置的时间
    first_seen: Instant,
}

// ─── Segmenter ───────────────────────────────────────────────────────────

pub(crate) struct Segmenter {
    pub(crate) config: SegConfig,

    // ── 时间锚点 ──
    stream_started: Instant,
    last_cut: Option<Instant>,
    last_nonempty_partial: Option<Instant>,

    // ── partial 跟踪 ──
    last_partial: String,
    punct_candidate: Option<PunctCandidate>,

    // ── 无标点兜底跟踪 ──
    /// 无标点场景下，稳定 partial 的首次出现时间与文本
    no_punct_stable_since: Option<(Instant, String)>,

    // ── 已提交前缀 ──
    /// 最近一次切句提交的文本（用于计算 draft，避免重复）
    committed_text: String,

    // ── 统计 ──
    pub(crate) stats: SegStats,
}

impl Segmenter {
    pub(crate) fn new(config: SegConfig, now: Instant) -> Self {
        Self {
            config,
            stream_started: now,
            last_cut: None,
            last_nonempty_partial: None,
            last_partial: String::new(),
            punct_candidate: None,
            no_punct_stable_since: None,
            committed_text: String::new(),
            stats: SegStats::default(),
        }
    }

    // ── 主入口 ──────────────────────────────────────────────────────────

    /// 喂入新的 partial 文本。返回 `Some(SegmentAction)` 表示触发了切句。
    pub(crate) fn feed_partial(
        &mut self,
        partial: &str,
        now: Instant,
    ) -> Option<SegmentAction> {
        let partial = partial.trim();

        // 空 partial：重置候选但不触发任何动作
        if partial.is_empty() {
            self.punct_candidate = None;
            self.no_punct_stable_since = None;
            return None;
        }

        // 首次非空 partial 时间锚点
        if self.last_nonempty_partial.is_none() {
            self.last_nonempty_partial = Some(now);
        }

        let partial_changed = self.last_partial != partial;
        self.last_partial = partial.to_string();

        // ── 优先级 4：极端兜底 25s ──
        if let Some(action) = self.check_extreme(partial, now) {
            return Some(action);
        }

        // ── 优先级 2：标点切句 ──
        if let Some(action) = self.check_punctuation(partial, partial_changed, now) {
            return Some(action);
        }

        // ── 优先级 3：无标点兜底 10s ──
        if let Some(action) = self.check_no_punct(partial, partial_changed, now) {
            return Some(action);
        }

        None
    }

    /// VAD 静音收尾（优先级 1）：外部 VAD 检测到"人真的停了"。
    /// 返回 `Some(VadReset)` 提交剩余尾巴并重建 stream。
    pub(crate) fn on_vad_silence(&mut self, _now: Instant) -> Option<SegmentAction> {
        let new_text = self.compute_new_text(&self.last_partial).to_string();
        if new_text.trim().is_empty() {
            return None;
        }
        self.record_segment(&new_text, false);
        Some(SegmentAction::VadReset { text: new_text })
    }

    /// 流被外部重置后（VadReset / ExtremeReset），清空内部状态。
    pub(crate) fn on_stream_reset(&mut self, now: Instant) {
        self.stream_started = now;
        self.last_cut = None;
        self.last_nonempty_partial = None;
        self.last_partial.clear();
        self.punct_candidate = None;
        self.no_punct_stable_since = None;
        self.committed_text.clear();
    }

    /// 当前草稿文本 = 完整 partial - 已提交前缀。
    /// 用 `floor_char_boundary` 防止 committed_text 过期时切到多字节字符中间。
    pub(crate) fn draft(&self) -> &str {
        let partial = &self.last_partial;
        let cut = self.committed_text.len().min(partial.len());
        // 对齐到 UTF-8 字符边界，防止 panic
        let safe_cut = if partial.is_char_boundary(cut) {
            cut
        } else {
            // 向前找最近的合法边界
            (0..cut).rev().find(|&i| partial.is_char_boundary(i)).unwrap_or(0)
        };
        &partial[safe_cut..]
    }

    // ── 内部：按优先级逐级检查 ──────────────────────────────────────────

    /// 优先级 4：25s 极端强切。
    fn check_extreme(&mut self, partial: &str, now: Instant) -> Option<SegmentAction> {
        let elapsed = self
            .last_nonempty_partial
            .map(|t| (now - t).as_secs_f32())
            .unwrap_or(0.0);
        if elapsed < self.config.extreme_timeout_sec {
            return None;
        }
        let new_text = self.compute_new_text(partial);
        if new_text.trim().is_empty() {
            return None;
        }
        self.stats.extreme_cut_count += 1;
        self.record_segment(new_text, false);
        Some(SegmentAction::ExtremeReset {
            text: new_text.to_string(),
        })
    }

    /// 优先级 2：句末标点稳定切句。
    fn check_punctuation(
        &mut self,
        partial: &str,
        partial_changed: bool,
        now: Instant,
    ) -> Option<SegmentAction> {
        // 找到最后一个句末标点
        let last_punct = partial
            .char_indices()
            .rev()
            .find(|(_, c)| SENTENCE_ENDINGS.contains(c));

        let (punct_pos, punct_char) = match last_punct {
            Some(p) => p,
            None => {
                // 没有标点：清除候选，更新反悔计数
                if self.punct_candidate.is_some() {
                    self.stats.punct_regret_count += 1;
                    self.punct_candidate = None;
                }
                return None;
            }
        };

        // 标点候选出现次数统计
        if self.punct_candidate.is_none() {
            self.stats.punct_candidate_count += 1;
        }

        // 检查冷却
        if let Some(last_cut) = self.last_cut {
            let cooldown_elapsed = (now - last_cut).as_secs_f32();
            if cooldown_elapsed < self.config.punct_cooldown_sec {
                return None;
            }
        }

        // 检查最小新增长度（3 个英文词）
        let cut_pos = punct_pos + punct_char.len_utf8(); // 切到标点之后（含标点）
        let after_committed = if partial.starts_with(&self.committed_text) {
            &partial[self.committed_text.len()..]
        } else {
            // ASR 修订了已提交前缀：反悔
            self.stats.punct_regret_count += 1;
            self.punct_candidate = None;
            self.committed_text.clear();
            partial
        };
        let new_part = &after_committed[..cut_pos.saturating_sub(self.committed_text.len()).min(after_committed.len())];
        let new_word_count = new_part.split_whitespace().count();
        if new_word_count < self.config.punct_min_new_words {
            return None;
        }

        // 连接词保护：切点最后词是连接词则不切
        if Self::ends_with_connection_word(&partial[..cut_pos]) {
            return None;
        }

        // 更新/创建候选
        let is_same_candidate = self
            .punct_candidate
            .as_ref()
            .is_some_and(|c| c.byte_pos == punct_pos);

        if partial_changed || !is_same_candidate {
            if is_same_candidate {
                // same position but partial changed — wait for stability
                return None;
            }
            // 新候选（位置变了或首次出现）
            if self.punct_candidate.is_some() {
                self.stats.punct_regret_count += 1;
            }
            self.punct_candidate = Some(PunctCandidate {
                byte_pos: punct_pos,
                first_seen: now,
            });
            return None;
        }

        // 候选位置稳定：检查是否超过 stability 阈值
        let candidate = self.punct_candidate.as_ref().unwrap();
        let stable_elapsed = (now - candidate.first_seen).as_secs_f32();
        if stable_elapsed < self.config.punct_stability_sec {
            return None;
        }

        // 稳定！执行 checkpoint 切句
        let text = partial[..cut_pos].to_string();
        let draft = partial[cut_pos..].trim().to_string();
        let latency_ms = (now - candidate.first_seen).as_millis() as u64;

        self.stats.punct_success_count += 1;
        self.stats.total_cut_latency_ms += latency_ms;
        self.record_segment(&text, true);
        self.last_cut = Some(now);
        self.last_nonempty_partial = Some(now); // #fix: 重置计时基线，避免极端兜底误触发
        self.punct_candidate = None;
        self.committed_text = text.clone();

        Some(SegmentAction::Checkpoint { text, draft })
    }

    /// 优先级 3：无标点 10s 兜底。
    fn check_no_punct(
        &mut self,
        partial: &str,
        partial_changed: bool,
        now: Instant,
    ) -> Option<SegmentAction> {
        let elapsed = self
            .last_nonempty_partial
            .map(|t| (now - t).as_secs_f32())
            .unwrap_or(0.0);
        if elapsed < self.config.no_punct_timeout_sec {
            return None;
        }

        let word_count = partial.split_whitespace().count();
        if word_count < self.config.no_punct_min_words {
            return None;
        }

        // 稳定性检查：partial 在 no_punct_stability_sec 内没变
        if partial_changed {
            self.no_punct_stable_since = Some((now, partial.to_string()));
            return None;
        }
        if let Some((stable_since, ref stable_text)) = self.no_punct_stable_since {
            if *stable_text != partial {
                // 不应该到这里（partial_changed 已经处理），防御
                self.no_punct_stable_since = Some((now, partial.to_string()));
                return None;
            }
            let stable_elapsed = (now - stable_since).as_secs_f32();
            if stable_elapsed < self.config.no_punct_stability_sec {
                return None;
            }
        } else {
            self.no_punct_stable_since = Some((now, partial.to_string()));
            return None;
        }

        // 保留最后 N 个词不提交
        let words: Vec<&str> = partial.split_whitespace().collect();
        let keep = self.config.no_punct_keep_last_words.min(words.len());
        let keep_start = words.len() - keep;
        // 把切点之前的 words 连成提交文本（保留空格连接）
        let keep_words = &words[keep_start..];
        let draft = keep_words.join(" ");

        // 找到提交文本（保持原始空白）
        let committed_words = &words[..keep_start];
        if committed_words.is_empty() {
            return None;
        }
        // 通过最后一个要提交的 word 的字节位置确定切点
        let cut_pos = Self::find_nth_word_end(partial, keep_start.saturating_sub(1))
            .unwrap_or(partial.len());

        // 连接词保护
        if Self::ends_with_connection_word(&partial[..cut_pos]) {
            return None;
        }

        let text = partial[..cut_pos].trim().to_string();
        if text.is_empty() {
            return None;
        }

        self.stats.no_punct_fallback_count += 1;
        self.record_segment(&text, true);
        self.last_cut = Some(now);
        self.last_nonempty_partial = Some(now); // #fix: 重置计时基线
        self.committed_text = text.clone();
        self.no_punct_stable_since = None;

        Some(SegmentAction::Checkpoint {
            text,
            draft: draft.to_string(),
        })
    }

    // ── 辅助 ────────────────────────────────────────────────────────────

    /// 计算自上次提交以来的新文本。
    fn compute_new_text<'a>(&self, partial: &'a str) -> &'a str {
        if self.committed_text.is_empty() {
            return partial;
        }
        if partial.starts_with(&self.committed_text) {
            &partial[self.committed_text.len()..]
        } else {
            // ASR 修订了已提交前缀
            partial
        }
    }

    /// 检查文本是否以连接词结尾。
    fn ends_with_connection_word(text: &str) -> bool {
        let last_word = text
            .split_whitespace()
            .last()
            .unwrap_or("")
            .trim_end_matches(|c: char| !c.is_alphabetic())
            .to_ascii_lowercase();
        if last_word.is_empty() {
            return false;
        }
        CONNECTION_WORDS.contains(&last_word.as_str())
    }

    /// 定位 `split_whitespace()` 顺序下第 `n` 个词的结束字节偏移。
    /// 按序扫描原串，每次通过 `find` 前进到下一个词，累积偏移。
    fn find_nth_word_end(haystack: &str, n: usize) -> Option<usize> {
        let mut offset = 0;
        for (i, word) in haystack.split_whitespace().enumerate() {
            let pos = haystack[offset..].find(word)?;
            offset += pos + word.len();
            if i == n {
                return Some(offset);
            }
        }
        None
    }

    /// 记录一次切句：累积统计。
    /// `is_checkpoint`：true 表示 text 是完整前缀文本（Checkpoint 路径），
    /// false 表示 text 是增量文本（VadReset / ExtremeReset 路径）。
    fn record_segment(&mut self, text: &str, is_checkpoint: bool) {
        self.stats.total_segment_count += 1;
        self.stats.total_words_sum += text.split_whitespace().count();

        // 重复风险：committed_text 出现在 text 的非前缀位置（ASR 修订后旧文本混入新文本）
        if !self.committed_text.is_empty() && !text.is_empty() {
            if let Some(pos) = text.find(&self.committed_text) {
                if pos > 0 {
                    self.stats.repeat_risk_count += 1;
                }
            }
        }

        // 漏词风险：仅对 checkpoint（完整前缀文本）做比较——上一条 committed 的词数
        // 和本次提交文本词数相比，如果本次明显更短，说明 ASR 可能漏词。
        if is_checkpoint && !self.committed_text.is_empty() {
            let prev_words = self.committed_text.split_whitespace().count();
            let new_words = text.split_whitespace().count();
            if prev_words > 0 && new_words > 0 && new_words < prev_words / 3 {
                self.stats.missing_word_risk_count += 1;
            }
        }
    }
}

// ─── 测试 ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn now() -> Instant {
        Instant::now()
    }

    fn make_seg() -> Segmenter {
        Segmenter::new(SEG_CONFIG_ENHANCED_ZIPFORMER_EN.clone(), now())
    }

    fn advance(base: Instant, secs: f32) -> Instant {
        base + Duration::from_secs_f32(secs)
    }

    // ── 空 partial / 去重 ──

    #[test]
    fn empty_partial_returns_none() {
        let mut seg = make_seg();
        let t = now();
        assert!(seg.feed_partial("", t).is_none());
        assert!(seg.feed_partial("   ", t).is_none());
    }

    // ── 标点检测：出现 → 稳定 → checkpoint ──

    #[test]
    fn punct_stable_cut() {
        let mut seg = make_seg();
        let t0 = now();

        // 首次出现标点（≥3 词才触发）— 候选创建
        assert!(seg.feed_partial("Hello beautiful world.", t0).is_none());
        assert_eq!(seg.stats.punct_candidate_count, 1);

        // 同一 partial 在 stability 时间内 — 不切
        let t1 = advance(t0, 0.5);
        assert!(seg.feed_partial("Hello beautiful world.", t1).is_none());

        // 超过 700ms — 切
        let t2 = advance(t0, 0.75);
        let action = seg.feed_partial("Hello beautiful world.", t2);
        assert!(action.is_some());
        match action.unwrap() {
            SegmentAction::Checkpoint { text, draft } => {
                assert_eq!(text, "Hello beautiful world.");
                assert!(draft.is_empty());
            }
            _ => panic!("expected Checkpoint"),
        }
        assert_eq!(seg.stats.punct_success_count, 1);
    }

    // ── 标点反悔：标点出现后消失 ──

    #[test]
    fn punct_regret_on_disappear() {
        let mut seg = make_seg();
        let t0 = now();

        // ≥3 词才有候选
        assert!(seg.feed_partial("Hello beautiful world.", t0).is_none());
        assert_eq!(seg.stats.punct_candidate_count, 1);
        assert_eq!(seg.stats.punct_regret_count, 0);

        // 标点消失 → 反悔
        let t1 = advance(t0, 0.3);
        assert!(seg.feed_partial("Hello beautiful world", t1).is_none());
        assert_eq!(seg.stats.punct_regret_count, 1);
    }

    // ── 连接词保护 ──

    #[test]
    fn connection_word_blocks_cut() {
        let mut seg = make_seg();
        let t0 = now();

        // 以 "and" 结尾即使有足够时间和新词也不切
        let _ = seg.feed_partial("I like this and.", t0);
        // 追加更多词使稳定，但标点前的词是 "and"（在连接词列表中）
        // 注意：连接词检查的是切点之前整个前缀的最后一个词
        // "I like this and." 末尾词是 "and" → 被保护
        let t1 = advance(t0, 2.0);
        assert!(seg.feed_partial("I like this and.", t1).is_none());
        // 应该一直不切（因为连接词保护）
    }

    // ── 极端兜底 25s ──

    #[test]
    fn extreme_cut_after_25s() {
        let mut seg = make_seg();
        let t0 = now();

        // 无标点的长文本持续 25s
        assert!(seg.feed_partial("no punctuation at all just words going on and on and on and on", t0).is_none());
        let t1 = advance(t0, 26.0);
        let action = seg.feed_partial("no punctuation at all just words going on and on and on and on even more", t1);
        assert!(action.is_some());
        match action.unwrap() {
            SegmentAction::ExtremeReset { .. } => {}
            _ => panic!("expected ExtremeReset"),
        }
        assert_eq!(seg.stats.extreme_cut_count, 1);
    }

    // ── 无标点兜底 10s ──

    #[test]
    fn no_punct_fallback() {
        let mut seg = make_seg();
        let t0 = now();

        // 12+ 词、无标点、超过 10s、稳定 1s
        // 注意：第 10 词不能是连接词（at/in/for/and 等），否则连接词保护会拦截切句
        let text = "this is a long sentence without any punctuation marks hello world really long";
        assert!(text.split_whitespace().count() >= 12);

        // 先在 10s 内喂一次，让 no_punct_stable_since 有机会被设置
        assert!(seg.feed_partial(text, t0).is_none());

        // 超过 10s：设置 stable_since 锚点
        let t1 = advance(t0, 11.0);
        assert!(seg.feed_partial(text, t1).is_none());

        // 再等 1.5s（超过 no_punct_stability_sec=1.0）：触发
        let t2 = advance(t1, 1.5);
        let action = seg.feed_partial(text, t2);
        assert!(action.is_some());
        match action.unwrap() {
            SegmentAction::Checkpoint { draft, .. } => {
                assert!(!draft.is_empty());
                let draft_words: Vec<&str> = draft.split_whitespace().collect();
                assert_eq!(draft_words.len(), 3, "draft should keep last 3 words, got: {draft}");
            }
            _ => panic!("expected Checkpoint"),
        }
        assert_eq!(seg.stats.no_punct_fallback_count, 1);
    }

    // ── 最小词数不满足时不切 ──

    #[test]
    fn no_punct_insufficient_words() {
        let mut seg = make_seg();
        let t0 = now();

        // 少于 12 词，即使超过 10s 也不切
        let text = "short text here";
        assert!(seg.feed_partial(text, t0).is_none());
        let t1 = advance(t0, 11.0);
        assert!(seg.feed_partial(text, t1).is_none());
        assert_eq!(seg.stats.no_punct_fallback_count, 0);
    }

    // ── 统计指标正确性 ──

    #[test]
    fn stats_accumulate_correctly() {
        let mut seg = make_seg();
        let t0 = now();

        // ≥3 词
        let _ = seg.feed_partial("Hello beautiful world.", t0);
        let t1 = advance(t0, 1.0);
        seg.feed_partial("Hello beautiful world.", t1);

        assert_eq!(seg.stats.total_segment_count, 1);
        assert_eq!(seg.stats.punct_success_count, 1);
        assert!(seg.stats.total_cut_latency_ms >= 700);
        assert_eq!(seg.stats.total_words_sum, 3);
    }

    // ── VAD silence ──

    #[test]
    fn vad_silence_commits_tail() {
        let mut seg = make_seg();
        let t0 = now();

        // 先有一个 partial
        seg.feed_partial("Hello world. This is a test.", t0);
        // VAD 检测到静音
        let t1 = advance(t0, 2.0);
        let action = seg.on_vad_silence(t1);
        assert!(action.is_some());
        match action.unwrap() {
            SegmentAction::VadReset { text } => {
                assert!(!text.is_empty());
            }
            _ => panic!("expected VadReset"),
        }
    }

    // ── stream reset 清空状态 ──

    #[test]
    fn stream_reset_clears_state() {
        let mut seg = make_seg();
        let t0 = now();

        seg.feed_partial("Hello world.", t0);
        assert!(!seg.last_partial.is_empty());

        seg.on_stream_reset(advance(t0, 1.0));
        assert!(seg.last_partial.is_empty());
        assert!(seg.committed_text.is_empty());
        assert!(seg.punct_candidate.is_none());
    }

    // ── find_nth_word_end ──

    #[test]
    fn find_nth_word_end_basic() {
        let s = "hello world this is a test";
        // word 0 "hello" → ends at 5
        assert_eq!(Segmenter::find_nth_word_end(s, 0), Some(5));
        // word 1 "world" → ends at 11
        assert_eq!(Segmenter::find_nth_word_end(s, 1), Some(11));
        // word 5 "test" → ends at len
        assert_eq!(Segmenter::find_nth_word_end(s, 5), Some(s.len()));
        // out of range
        assert_eq!(Segmenter::find_nth_word_end(s, 10), None);
    }

    // ── checkpoint 后计时基线重置 ──

    #[test]
    fn timer_reset_after_checkpoint() {
        let mut seg = make_seg();
        let t0 = now();

        // 第一次标点切句（≥3 词）
        let _ = seg.feed_partial("This is first sentence.", t0);
        let t1 = advance(t0, 1.0);
        seg.feed_partial("This is first sentence.", t1);
        assert_eq!(seg.stats.punct_success_count, 1);

        // 紧接着第二句——如果计时基线没重置，极端兜底会误触发
        let t2 = advance(t1, 0.5);
        let _ = seg.feed_partial("This is first sentence. Another whole new sentence here.", t2);
        let t3 = advance(t2, 1.0);
        let action = seg.feed_partial("This is first sentence. Another whole new sentence here.", t3);
        // 应该正常标点切句，不是极端强切
        match action {
            Some(SegmentAction::Checkpoint { .. }) => {} // OK
            Some(SegmentAction::ExtremeReset { .. }) => panic!("should not extreme-reset after normal checkpoint"),
            _ => {} // might need more time or words
        }
        assert_eq!(seg.stats.extreme_cut_count, 0);
    }
}
