//! 流式槽（OnlineRecognizer：Zipformer / Nemotron）：真流式 transducer，增量吃帧。
//!
//! 支持两条路径：
//! - 旧路径（默认）：Sherpa 内置 endpoint
//! - 新路径（`use_custom_endpoint = true`）：自定义 segmenter + 收尾 VAD

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use sherpa_onnx::{OnlineRecognizer, OnlineRecognizerConfig, OnlineStream, VoiceActivityDetector};

use arcvoice_core::asr::streaming::SherpaModelType;

use crate::config::{
    endpoint_rules_from_env, engine_models_dir, hotwords_from_env, silero_vad_model_path,
    FINALS_RETAIN, PARTIAL_HISTORY_RETAIN, VAD_SAMPLE_RATE,
};
use crate::models::{ModelDesc, SlotKind};
use crate::segments::{SegConfig, SegmentAction, Segmenter};
use crate::util::with_stdout_suppressed;
use crate::vad::create_silero_vad;

pub(crate) struct OnlineSlot {
    pub(crate) name: &'static str,
    pub(crate) recognizer: Arc<OnlineRecognizer>,
    pub(crate) stream: OnlineStream,
    pub(crate) partial: String,
    pub(crate) all_partials: VecDeque<String>,
    pub(crate) finals: Vec<String>,
    pub(crate) all_finals: Vec<String>,
    pub(crate) enabled: bool,
    pub(crate) finals_scroll: u16,

    // ── 自定义 endpoint（加强版）──
    /// 是否使用自定义 segmenter 替代 Sherpa 内置 endpoint
    pub(crate) use_custom_endpoint: bool,
    /// 标点切句状态机（仅 `use_custom_endpoint` 时存在）
    pub(crate) segmenter: Option<Segmenter>,
    /// 收尾 VAD：检测"人真的停了"（仅 `use_custom_endpoint` 时存在；
    /// 若 Silero 模型文件缺失则为 None，此时优先级 1 不可用）。
    pub(crate) tail_vad: Option<VoiceActivityDetector>,
}

pub(crate) fn build_online_slot(
    desc: &ModelDesc,
    seg_config: Option<&SegConfig>,
) -> Result<OnlineSlot> {
    let model_dir = engine_models_dir().join(desc.subdir);
    let model_type = match desc.kind {
        SlotKind::Online(mt) => mt,
        SlotKind::Offline(_) => bail!("build_online_slot 收到离线模型"),
    };
    let (rule1, rule2, rule3) = endpoint_rules_from_env();
    let files = online_model_files(desc.name, &model_dir, model_type)?;
    let hotwords = if matches!(model_type, SherpaModelType::Zipformer) {
        hotwords_from_env()
    } else {
        Vec::new()
    };
    let mut cfg = OnlineRecognizerConfig::default();
    cfg.feat_config.feature_dim = online_feature_dim(model_type);
    cfg.model_config.transducer.encoder = Some(files.encoder.to_string_lossy().into_owned());
    cfg.model_config.transducer.decoder = Some(files.decoder.to_string_lossy().into_owned());
    cfg.model_config.transducer.joiner = Some(files.joiner.to_string_lossy().into_owned());
    cfg.model_config.tokens = Some(files.tokens.to_string_lossy().into_owned());
    cfg.model_config.num_threads = 1;
    cfg.model_config.provider = Some("cpu".into());
    cfg.model_config.debug = false;
    cfg.decoding_method = Some(
        match (model_type, hotwords.is_empty()) {
            (SherpaModelType::Zipformer, false) => "modified_beam_search",
            _ => "greedy_search",
        }
        .into(),
    );
    cfg.max_active_paths = 4;

    // ── 自定义 endpoint 路径 ──
    let use_custom_endpoint = seg_config.is_some();
    if use_custom_endpoint {
        // 禁用 Sherpa 内置 endpoint，交给自定义 segmenter 全权管理
        cfg.enable_endpoint = false;
    } else {
        cfg.enable_endpoint = true;
        cfg.rule1_min_trailing_silence = rule1;
        cfg.rule2_min_trailing_silence = rule2;
        cfg.rule3_min_utterance_length = rule3;
    }

    if !hotwords.is_empty() {
        cfg.hotwords_score = 2.0;
        cfg.hotwords_buf = Some(hotwords.join("\n").into_bytes());
    }

    let recognizer = Arc::new(
        with_stdout_suppressed(|| OnlineRecognizer::create(&cfg)).with_context(|| {
            format!("加载模型 {} 失败，目录 {:?} 是否存在", desc.name, model_dir)
        })?,
    );
    let stream = if hotwords.is_empty() {
        recognizer.create_stream()
    } else {
        recognizer.create_stream_with_hotwords(&hotwords.join("\n"))
    };

    // ── 创建自定义 segmenter + 收尾 VAD ──
    let (segmenter, tail_vad) = if let Some(config) = seg_config {
        let now = Instant::now();
        let seg = Segmenter::new(config.clone(), now);
        let vad_model_path = silero_vad_model_path();
        let vad = if vad_model_path.exists() {
            Some(create_silero_vad(
                &vad_model_path,
                config.vad_threshold,
                config.vad_min_silence_sec,
                0.0,                          // min_speech_sec = 0（一检测到就开始）
                config.vad_max_speech_sec,
                5.0,                          // buffer_sec
            )?)
        } else {
            eprintln!(
                "[加强版] Silero VAD 模型缺失，VAD 收尾不可用。路径: {}",
                vad_model_path.display()
            );
            None
        };
        (Some(seg), vad)
    } else {
        (None, None)
    };

    Ok(OnlineSlot {
        name: desc.name,
        recognizer,
        stream,
        partial: String::new(),
        all_partials: VecDeque::new(),
        finals: Vec::new(),
        all_finals: Vec::new(),
        enabled: true,
        finals_scroll: 0,
        use_custom_endpoint,
        segmenter,
        tail_vad,
    })
}

pub(crate) fn feed_online_frame(slot: &mut OnlineSlot, pcm: &[f32]) -> Option<String> {
    if !slot.enabled {
        return None;
    }
    slot.stream.accept_waveform(VAD_SAMPLE_RATE as i32, pcm);
    while slot.recognizer.is_ready(&slot.stream) {
        slot.recognizer.decode(&slot.stream);
    }

    // ── 旧路径：Sherpa 内置 endpoint ──
    if !slot.use_custom_endpoint {
        let next_partial = slot
            .recognizer
            .get_result(&slot.stream)
            .map(|result| result.text)
            .unwrap_or_default();
        let trimmed_partial = next_partial.trim();
        if !trimmed_partial.is_empty()
            && slot
                .all_partials
                .back()
                .is_none_or(|last| last.trim() != trimmed_partial)
        {
            slot.all_partials.push_back(trimmed_partial.to_string());
            if slot.all_partials.len() > PARTIAL_HISTORY_RETAIN {
                slot.all_partials.pop_front();
            }
        }
        slot.partial = next_partial;
        if slot.recognizer.is_endpoint(&slot.stream) {
            let final_text = slot
                .recognizer
                .get_result(&slot.stream)
                .map(|result| result.text)
                .unwrap_or_default();
            slot.recognizer.reset(&slot.stream);
            slot.partial.clear();
            if !final_text.trim().is_empty() {
                slot.finals.push(final_text.clone());
                if slot.finals.len() > FINALS_RETAIN {
                    slot.finals.remove(0);
                }
                slot.all_finals.push(final_text.clone());
                return Some(final_text);
            }
        }
        return None;
    }

    // ── 新路径：自定义 segmenter + 收尾 VAD ──
    let now = Instant::now();

    // 1) 喂收尾 VAD，检测语音段结束（静音 ≥ 300ms）
    if let Some(vad) = &mut slot.tail_vad {
        vad.accept_waveform(pcm);
        // 检查是否有已完成的语音段（Silero 返回整个 speech segment）
        while let Some(front) = vad.front() {
            let seg_samples = front.samples();
            let seg_len_sec = seg_samples.len() as f32 / VAD_SAMPLE_RATE as f32;
            drop(front);
            vad.pop();
            // 只有 ≥ 0.3s 的语音段才算"真正说话"，避免噪声误触
            if seg_len_sec >= 0.3 {
                if let Some(ref mut seg) = slot.segmenter {
                    if let Some(action) = seg.on_vad_silence(now) {
                        return apply_segment_action(slot, action);
                    }
                }
            }
        }
    }

    // 2) 取 partial
    let partial = slot
        .recognizer
        .get_result(&slot.stream)
        .map(|result| result.text)
        .unwrap_or_default();

    // 记录 partial 历史
    let trimmed_partial = partial.trim();
    if !trimmed_partial.is_empty()
        && slot
            .all_partials
            .back()
            .is_none_or(|last| last.trim() != trimmed_partial)
    {
        slot.all_partials.push_back(trimmed_partial.to_string());
        if slot.all_partials.len() > PARTIAL_HISTORY_RETAIN {
            slot.all_partials.pop_front();
        }
    }

    // 3) 喂 segmenter
    if let Some(ref mut seg) = slot.segmenter {
        if let Some(action) = seg.feed_partial(&partial, now) {
            return apply_segment_action(slot, action);
        }
        // 无切句：partial = 草稿（full partial - committed prefix）
        slot.partial = seg.draft().to_string();
    }

    None
}

/// 把 segment action 落地到 slot 的 finals/partial/stream 上。
fn apply_segment_action(slot: &mut OnlineSlot, action: SegmentAction) -> Option<String> {
    match action {
        SegmentAction::Checkpoint { text, draft } => {
            // Checkpoint 不重建流（保留声学上下文），text 已是增量文本
            slot.partial = draft;
            if text.trim().is_empty() {
                return None;
            }
            slot.finals.push(text.clone());
            if slot.finals.len() > FINALS_RETAIN {
                slot.finals.remove(0);
            }
            slot.all_finals.push(text.clone());
            Some(text)
        }
        SegmentAction::VadReset { text } | SegmentAction::ExtremeReset { text } => {
            slot.partial.clear();
            if text.trim().is_empty() {
                // 即使无文本也需重置 segmenter 状态（流已 reset）
                if let Some(ref mut seg) = slot.segmenter {
                    seg.on_stream_reset(Instant::now());
                }
                slot.recognizer.reset(&slot.stream);
                return None;
            }
            slot.finals.push(text.clone());
            if slot.finals.len() > FINALS_RETAIN {
                slot.finals.remove(0);
            }
            slot.all_finals.push(text.clone());
            // 重置
            slot.recognizer.reset(&slot.stream);
            if let Some(ref mut seg) = slot.segmenter {
                seg.on_stream_reset(Instant::now());
            }
            Some(text)
        }
    }
}

struct OnlineModelFiles {
    encoder: PathBuf,
    decoder: PathBuf,
    joiner: PathBuf,
    tokens: PathBuf,
}

fn online_model_files(
    name: &str,
    model_dir: &Path,
    model_type: SherpaModelType,
) -> Result<OnlineModelFiles> {
    Ok(OnlineModelFiles {
        encoder: first_existing_model_file(
            name,
            model_dir,
            &["encoder.int8.onnx", "encoder.onnx"],
        )?,
        decoder: first_existing_model_file(
            name,
            model_dir,
            match model_type {
                SherpaModelType::NemotronStreaming => &["decoder.int8.onnx", "decoder.onnx"],
                SherpaModelType::Zipformer => &["decoder.onnx", "decoder.int8.onnx"],
            },
        )?,
        joiner: first_existing_model_file(name, model_dir, &["joiner.int8.onnx", "joiner.onnx"])?,
        tokens: first_existing_model_file(name, model_dir, &["tokens.txt"])?,
    })
}

fn first_existing_model_file(name: &str, model_dir: &Path, candidates: &[&str]) -> Result<PathBuf> {
    candidates
        .iter()
        .map(|filename| model_dir.join(filename))
        .find(|path| path.exists())
        .with_context(|| {
            format!(
                "{} 模型文件缺失: 需要 {} 之一，目录 {}",
                name,
                candidates.join(" / "),
                model_dir.display()
            )
        })
}

fn online_feature_dim(model_type: SherpaModelType) -> i32 {
    match model_type {
        SherpaModelType::NemotronStreaming => 128,
        SherpaModelType::Zipformer => 80,
    }
}
