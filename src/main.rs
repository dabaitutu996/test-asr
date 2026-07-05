//! ASR 模型对比 TUI 工具：采集系统音频，实时对比多个 Sherpa 流式模型的识别效果。
//!
//! 启动时进入选择界面，用 1-9 勾选要加载的引擎，Enter 确认，q 退出。
//! 运行中：q / Esc / Ctrl-C 退出；c 清空历史；1-9 切换引擎启用/禁用。
//!
//! 模型路径指向 game-video/engine/models/streaming/。
//! 运行：cargo run

#![cfg(not(target_os = "windows"))]

use std::fs;
use std::io;
use std::os::unix::io::RawFd;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Terminal;
use sherpa_onnx::{
    OfflineRecognizer, OfflineRecognizerConfig, OnlineRecognizer, OnlineRecognizerConfig,
    OnlineStream, SileroVadModelConfig, VadModelConfig, VoiceActivityDetector,
};
use tokio::sync::mpsc;

use arcvoice_core::asr::streaming::SherpaModelType;
use arcvoice_core::asr::Language;
use arcvoice_core::audio::mic::MicInput;
use asr_compare_tui::system_audio::{self, OutputDevice};

/// 顶部 partial 之外，每个 slot 最多保留多少条 final 历史。
const FINALS_RETAIN: usize = 6;
/// 每个在线槽最多记录多少条 partial 变化，用于诊断 endpoint/reset 是否丢词。
const PARTIAL_HISTORY_RETAIN: usize = 2000;
/// TUI 轮询周期：终端事件与重绘节奏。
const POLL_INTERVAL: Duration = Duration::from_millis(50);
const DEFAULT_MEDIA_MP3: &str = "ARC_Raiders_is_getting_DARKER....mp3";
const DEFAULT_MEDIA_SRT: &str = "ARC_Raiders_is_getting_DARKER....en.srt";

// ─── 离线模型 VAD 参数 ─────────────────────────────────────────────────
const VAD_SAMPLE_RATE: usize = 16_000;
const SILERO_VAD_THRESHOLD: f32 = 0.5;
const SILERO_VAD_MIN_SILENCE_SEC: f32 = 0.5;
const SILERO_VAD_MIN_SPEECH_SEC: f32 = 0.25;
const SILERO_VAD_MAX_SPEECH_SEC: f32 = 8.0;
const SILERO_VAD_WINDOW_SIZE: i32 = 512;
const SILERO_VAD_BUFFER_SEC: f32 = 10.0;

/// RMS 低于此值视为静音；仅在 `VAD_BACKEND=rms` 时使用。
const VAD_RMS_THRESHOLD: f32 = 0.012;
/// RMS 兜底路径里，静音持续多少秒后触发离线解码。
const VAD_SILENCE_SEC: f32 = 0.4;
/// RMS 兜底路径里，缓冲最长多少秒强制触发解码。
const VAD_MAX_BUFFER_SEC: f32 = 8.0;

// ─── 工具函数 ──────────────────────────────────────────────────────────

/// 在闭包执行期间临时将 stdout（fd 1）和 stderr（fd 2）重定向到 /dev/null，
/// 屏蔽 C 库的调试输出（sherpa-onnx / onnxruntime 加载时会通过 stderr dump
/// 大量 encoder/decoder 配置与 CoreML 警告，这些原始字节会与 TUI 画面混在一起）。
fn with_stdout_suppressed<F: FnOnce() -> R, R>(f: F) -> R {
    extern "C" {
        fn dup(fd: RawFd) -> RawFd;
        fn dup2(oldfd: RawFd, newfd: RawFd) -> RawFd;
        fn close(fd: RawFd) -> RawFd;
        fn open(path: *const u8, oflag: RawFd, ...) -> RawFd;
    }
    const O_WRONLY: RawFd = 1;

    unsafe {
        let saved_stdout = dup(1);
        let saved_stderr = dup(2);
        let dev_null = open(b"/dev/null\0".as_ptr(), O_WRONLY);
        if dev_null >= 0 {
            dup2(dev_null, 1);
            dup2(dev_null, 2);
            close(dev_null);
        }

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));

        if saved_stdout >= 0 {
            dup2(saved_stdout, 1);
            close(saved_stdout);
        }
        if saved_stderr >= 0 {
            dup2(saved_stderr, 2);
            close(saved_stderr);
        }

        match result {
            Ok(val) => val,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }
}

/// engine 模型基础目录：从本项目 Cargo.toml 向上两级到 Desktop，
/// 再进入 game-video/engine/models/streaming/。
fn engine_models_dir() -> PathBuf {
    // 优先用环境变量覆盖，方便灵活部署。
    if let Ok(dir) = std::env::var("ENGINE_MODELS_DIR") {
        return PathBuf::from(dir);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../game-video/engine/models/streaming")
}

/// 离线模型（Canary 等）目录：本项目下的 models/streaming/。
/// 与流式模型分开存放——流式模型沿用 sibling game-video 仓库，
/// 离线模型体积大、独立性强，放本仓库更清晰。
fn offline_models_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("OFFLINE_MODELS_DIR") {
        return PathBuf::from(dir);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/streaming")
}

fn silero_vad_model_path() -> PathBuf {
    if let Ok(path) = std::env::var("SILERO_VAD_MODEL") {
        return PathBuf::from(path);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/vad/silero_vad/silero_vad.int8.onnx")
}

// ─── 模型静态描述 ─────────────────────────────────────────────────────

/// 区分流式（OnlineRecognizer，真流式 transducer）和离线（OfflineRecognizer，
/// 需要外部 VAD 触发解码）两类后端。
#[derive(Clone, Copy, PartialEq)]
enum SlotKind {
    /// sherpa-onnx OnlineRecognizer：Zipformer / Nemotron 等真流式 transducer。
    Online(SherpaModelType),
    /// sherpa-onnx OfflineRecognizer：离线模型 + 共享 VAD 触发。
    Offline(OfflineFamily),
}

#[derive(Clone, Copy, PartialEq)]
enum OfflineFamily {
    Canary,
    ParakeetNemoCtc,
}

impl OfflineFamily {
    fn as_str(self) -> &'static str {
        match self {
            Self::Canary => "canary",
            Self::ParakeetNemoCtc => "parakeet_nemo_ctc",
        }
    }

    fn required_files(self) -> &'static [&'static str] {
        match self {
            Self::Canary => &["encoder.int8.onnx", "decoder.int8.onnx", "tokens.txt"],
            Self::ParakeetNemoCtc => &["model.int8.onnx", "tokens.txt"],
        }
    }

    fn download_script(self) -> &'static str {
        match self {
            Self::Canary => "./scripts/download-canary.sh",
            Self::ParakeetNemoCtc => "./scripts/download-parakeet-tdt-ctc-110m.sh",
        }
    }
}

struct ModelDesc {
    name: &'static str,
    subdir: &'static str,
    kind: SlotKind,
    language: Language,
}

impl ModelDesc {
    /// 离线模型要求目录存在且文件齐全；不齐全时返回 false，调用方据此从选择屏过滤。
    /// 流式模型（Online）一律返回 true（缺文件会在 build_slot 阶段报错，那是另一条路径）。
    fn files_present(&self) -> bool {
        match self.kind {
            SlotKind::Online(_) => true,
            SlotKind::Offline(family) => {
                let dir = offline_models_dir().join(self.subdir);
                family
                    .required_files()
                    .iter()
                    .all(|filename| dir.join(filename).exists())
            }
        }
    }

    fn missing_files_hint(&self) -> Option<&'static str> {
        match self.kind {
            SlotKind::Online(_) => None,
            SlotKind::Offline(family) => Some(family.download_script()),
        }
    }
}

const MODEL_DESCS: &[ModelDesc] = &[
    ModelDesc {
        name: "Zipformer-zh",
        subdir: "zipformer-zh",
        kind: SlotKind::Online(SherpaModelType::Zipformer),
        language: Language::Chinese,
    },
    ModelDesc {
        name: "Zipformer-en",
        subdir: "zipformer-en",
        kind: SlotKind::Online(SherpaModelType::Zipformer),
        language: Language::English,
    },
    ModelDesc {
        name: "Nemotron-en",
        subdir: "nemotron-en",
        kind: SlotKind::Online(SherpaModelType::NemotronStreaming),
        language: Language::English,
    },
    ModelDesc {
        name: "Canary-180m-flash",
        subdir: "canary-180m-flash",
        kind: SlotKind::Offline(OfflineFamily::Canary),
        language: Language::English,
    },
    ModelDesc {
        name: "Parakeet-TDT-CTC-110M",
        subdir: "parakeet-tdt-ctc-110m",
        kind: SlotKind::Offline(OfflineFamily::ParakeetNemoCtc),
        language: Language::English,
    },
];

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.trim().parse::<f32>().ok())
        .unwrap_or(default)
}

fn endpoint_rules_from_env() -> (f32, f32, f32) {
    (
        env_f32("ASR_ENDPOINT_RULE1", 2.4),
        env_f32("ASR_ENDPOINT_RULE2", 1.2),
        env_f32("ASR_ENDPOINT_RULE3", 20.0),
    )
}

fn hotwords_from_env() -> Vec<String> {
    std::env::var("ASR_HOTWORDS")
        .ok()
        .map(|raw| {
            raw.split([',', '\n'])
                .map(str::trim)
                .filter(|word| !word.is_empty())
                .map(ToOwned::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

// ─── 运行态 ───────────────────────────────────────────────────────────

// ─── 流式槽（OnlineRecognizer：Zipformer / Nemotron）──────────────────

struct OnlineSlot {
    name: &'static str,
    #[allow(dead_code)]
    recognizer: Arc<OnlineRecognizer>,
    stream: OnlineStream,
    partial: String,
    all_partials: Vec<String>,
    finals: Vec<String>,
    all_finals: Vec<String>,
    enabled: bool,
    finals_scroll: u16,
}

fn build_online_slot(desc: &ModelDesc) -> Result<OnlineSlot> {
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
    cfg.enable_endpoint = true;
    cfg.rule1_min_trailing_silence = rule1;
    cfg.rule2_min_trailing_silence = rule2;
    cfg.rule3_min_utterance_length = rule3;
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
    Ok(OnlineSlot {
        name: desc.name,
        recognizer,
        stream,
        partial: String::new(),
        all_partials: Vec::new(),
        finals: Vec::new(),
        all_finals: Vec::new(),
        enabled: true,
        finals_scroll: 0,
    })
}

fn feed_online_frame(slot: &mut OnlineSlot, pcm: &[f32]) -> Option<String> {
    if !slot.enabled {
        return None;
    }
    slot.stream.accept_waveform(VAD_SAMPLE_RATE as i32, pcm);
    while slot.recognizer.is_ready(&slot.stream) {
        slot.recognizer.decode(&slot.stream);
    }
    let next_partial = slot
        .recognizer
        .get_result(&slot.stream)
        .map(|result| result.text)
        .unwrap_or_default();
    let trimmed_partial = next_partial.trim();
    if !trimmed_partial.is_empty()
        && slot
            .all_partials
            .last()
            .is_none_or(|last| last.trim() != trimmed_partial)
    {
        slot.all_partials.push(trimmed_partial.to_string());
        if slot.all_partials.len() > PARTIAL_HISTORY_RETAIN {
            slot.all_partials.remove(0);
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
            slot.all_finals.push(final_text.clone());
            if slot.finals.len() > FINALS_RETAIN {
                slot.finals.remove(0);
            }
            return Some(final_text);
        }
    }
    None
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

// ─── 离线槽（OfflineRecognizer：离线模型 + 共享 VAD）──────────────────
//
// 离线模型没有 partial / endpoint 概念。主循环把同一个 VAD segment 广播给
// 所有启用的离线槽，保证 Canary / Parakeet 在完全相同的音频片段上对比。

struct OfflineSlot {
    name: &'static str,
    family: OfflineFamily,
    #[allow(dead_code)]
    recognizer: OfflineRecognizer,
    partial: String, // 离线模型恒为空，UI 显示"等待 VAD 触发"
    finals: Vec<String>,
    all_finals: Vec<String>,
    enabled: bool,
    finals_scroll: u16,
    segments_decoded: usize,
    last_segment_samples: usize,
}

fn build_offline_slot(desc: &ModelDesc) -> Result<OfflineSlot> {
    let family = match desc.kind {
        SlotKind::Offline(family) => family,
        SlotKind::Online(_) => bail!("build_offline_slot 收到流式模型"),
    };
    let mut cfg = OfflineRecognizerConfig::default();

    match family {
        OfflineFamily::Canary => {
            let encoder = require_offline_file(desc, family, "encoder", "encoder.int8.onnx")?;
            let decoder = require_offline_file(desc, family, "decoder", "decoder.int8.onnx")?;
            let tokens = require_offline_file(desc, family, "tokens", "tokens.txt")?;

            // Canary 用 128 维 mel filterbank（sherpa-onnx 默认 80，必须显式设置）。
            cfg.feat_config.feature_dim = 128;
            cfg.model_config.canary.encoder = Some(encoder.to_string_lossy().into_owned());
            cfg.model_config.canary.decoder = Some(decoder.to_string_lossy().into_owned());
            cfg.model_config.canary.src_lang = Some("en".into());
            cfg.model_config.canary.tgt_lang = Some("en".into());
            cfg.model_config.canary.use_pnc = true; // 标点 + 大小写
            cfg.model_config.tokens = Some(tokens.to_string_lossy().into_owned());
        }
        OfflineFamily::ParakeetNemoCtc => {
            let model = require_offline_file(desc, family, "model", "model.int8.onnx")?;
            let tokens = require_offline_file(desc, family, "tokens", "tokens.txt")?;

            cfg.model_config.nemo_ctc.model = Some(model.to_string_lossy().into_owned());
            cfg.model_config.tokens = Some(tokens.to_string_lossy().into_owned());
        }
    }

    cfg.model_config.num_threads = 1;
    cfg.model_config.provider = Some("cpu".into());
    cfg.model_config.debug = false;

    let recognizer = with_stdout_suppressed(|| {
        OfflineRecognizer::create(&cfg)
            .with_context(|| format!("创建 {} OfflineRecognizer 失败", desc.name))
    })?;

    Ok(OfflineSlot {
        name: desc.name,
        family,
        recognizer,
        partial: String::new(),
        finals: Vec::new(),
        all_finals: Vec::new(),
        enabled: true,
        finals_scroll: 0,
        segments_decoded: 0,
        last_segment_samples: 0,
    })
}

fn require_offline_file(
    desc: &ModelDesc,
    family: OfflineFamily,
    label: &str,
    filename: &str,
) -> Result<PathBuf> {
    let path = offline_models_dir().join(desc.subdir).join(filename);
    if path.exists() {
        return Ok(path);
    }
    bail!(
        "{} 模型文件缺失: {} ({})\n请先运行 {} 下载。",
        desc.name,
        label,
        path.display(),
        family.download_script()
    );
}

/// 把一段已经 VAD 切好的音频喂给离线模型解码，返回识别文本。
fn decode_offline_segment(slot: &mut OfflineSlot, segment: &VadSegment) -> Option<String> {
    if !slot.enabled || segment.samples.is_empty() {
        slot.last_segment_samples = 0;
        return None;
    }
    slot.segments_decoded += 1;
    slot.last_segment_samples = segment.samples.len();

    let stream = slot.recognizer.create_stream();
    stream.accept_waveform(VAD_SAMPLE_RATE as i32, &segment.samples);
    slot.recognizer.decode(&stream);
    let text = stream
        .get_result()
        .map(|r| r.text)
        .unwrap_or_default()
        .trim()
        .to_string();
    if text.is_empty() {
        return None;
    }
    slot.finals.push(text.clone());
    slot.all_finals.push(text.clone());
    if slot.finals.len() > FINALS_RETAIN {
        slot.finals.remove(0);
    }
    Some(text)
}

// ─── VAD 状态机（离线槽共享）──────────────────────────────────────────

struct VadSegment {
    start_sample: usize,
    samples: Vec<f32>,
    reason: &'static str,
}

#[derive(Clone, Copy, PartialEq)]
enum VadBackend {
    Disabled,
    Silero,
    Rms,
}

impl VadBackend {
    fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Silero => "silero",
            Self::Rms => "rms",
        }
    }
}

struct VadState {
    inner: VadImpl,
    total_samples_seen: usize,
    segment_count: usize,
    last_segment_start_sample: Option<usize>,
    last_segment_samples: usize,
}

enum VadImpl {
    Disabled,
    Silero(SileroVadState),
    Rms(RmsVadState),
}

struct SileroVadState {
    detector: VoiceActivityDetector,
    settings: SileroVadSettings,
}

struct SileroVadSettings {
    model_path: PathBuf,
    threshold: f32,
    min_silence_sec: f32,
    min_speech_sec: f32,
    max_speech_sec: f32,
}

struct RmsVadState {
    buffer: Vec<f32>,
    buffer_start_sample: usize,
    speaking: bool,
    silence_samples: usize,
    rms_threshold: f32,
}

impl VadState {
    fn new(has_offline_slots: bool) -> Result<Self> {
        let inner = if !has_offline_slots {
            VadImpl::Disabled
        } else {
            match std::env::var("VAD_BACKEND")
                .unwrap_or_else(|_| "silero".to_string())
                .trim()
                .to_ascii_lowercase()
                .as_str()
            {
                "silero" => VadImpl::Silero(SileroVadState::new()?),
                "rms" => VadImpl::Rms(RmsVadState::new()),
                other => bail!("未知 VAD_BACKEND={other:?}，可用值: silero, rms"),
            }
        };
        Ok(Self {
            inner,
            total_samples_seen: 0,
            segment_count: 0,
            last_segment_start_sample: None,
            last_segment_samples: 0,
        })
    }

    /// 喂入一帧。返回所有刚完成的 VAD segment，应广播给所有离线模型。
    fn push(&mut self, pcm: &[f32], log: &mut Vec<String>) -> Vec<VadSegment> {
        let frame_start_sample = self.total_samples_seen;
        self.total_samples_seen += pcm.len();

        let segments = match &mut self.inner {
            VadImpl::Disabled => Vec::new(),
            VadImpl::Silero(state) => state.push(pcm),
            VadImpl::Rms(state) => state.push(pcm, frame_start_sample),
        };

        for segment in &segments {
            self.segment_count += 1;
            self.last_segment_start_sample = Some(segment.start_sample);
            self.last_segment_samples = segment.samples.len();
            push_log(
                log,
                format!(
                    "[VAD/{}] segment #{} ({}) start={:.2}s len={:.2}s",
                    self.backend().as_str(),
                    self.segment_count,
                    segment.reason,
                    segment.start_sample as f32 / VAD_SAMPLE_RATE as f32,
                    segment.samples.len() as f32 / VAD_SAMPLE_RATE as f32
                ),
            );
        }

        segments
    }

    fn backend(&self) -> VadBackend {
        match &self.inner {
            VadImpl::Disabled => VadBackend::Disabled,
            VadImpl::Silero(_) => VadBackend::Silero,
            VadImpl::Rms(_) => VadBackend::Rms,
        }
    }

    fn status_label(&self) -> String {
        let last = if self.last_segment_samples > 0 {
            format!(
                " · 最近 {:.1}s",
                self.last_segment_samples as f32 / VAD_SAMPLE_RATE as f32
            )
        } else {
            String::new()
        };
        match &self.inner {
            VadImpl::Disabled => "（离线 VAD 未启用）".to_string(),
            VadImpl::Silero(state) => {
                let status = if state.detector.detected() {
                    "检测中"
                } else {
                    "等待语音"
                };
                format!("（Silero·{status}{last}）")
            }
            VadImpl::Rms(state) => {
                if state.speaking {
                    format!(
                        "（RMS·检测中 {:.1}s{last}）",
                        state.buffer.len() as f32 / VAD_SAMPLE_RATE as f32
                    )
                } else {
                    format!("（RMS·等待语音{last}）")
                }
            }
        }
    }

    fn append_report(&self, out: &mut String) {
        out.push_str("\n## VAD\n\n");
        out.push_str(&format!("- backend: {}\n", self.backend().as_str()));
        out.push_str(&format!("- sample_rate: {}\n", VAD_SAMPLE_RATE));
        out.push_str(&format!("- segments: {}\n", self.segment_count));
        if let Some(start) = self.last_segment_start_sample {
            out.push_str(&format!(
                "- last_segment: start={:.3}s len={:.3}s\n",
                start as f32 / VAD_SAMPLE_RATE as f32,
                self.last_segment_samples as f32 / VAD_SAMPLE_RATE as f32
            ));
        }
        match &self.inner {
            VadImpl::Disabled => {}
            VadImpl::Silero(state) => {
                out.push_str(&format!(
                    "- silero_model: {}\n",
                    state.settings.model_path.display()
                ));
                out.push_str(&format!(
                    "- silero_threshold: {}\n",
                    state.settings.threshold
                ));
                out.push_str(&format!(
                    "- silero_min_silence_sec: {}\n",
                    state.settings.min_silence_sec
                ));
                out.push_str(&format!(
                    "- silero_min_speech_sec: {}\n",
                    state.settings.min_speech_sec
                ));
                out.push_str(&format!(
                    "- silero_max_speech_sec: {}\n",
                    state.settings.max_speech_sec
                ));
                out.push_str(&format!(
                    "- silero_window_size: {}\n",
                    SILERO_VAD_WINDOW_SIZE
                ));
                out.push_str(&format!("- silero_buffer_sec: {}\n", SILERO_VAD_BUFFER_SEC));
            }
            VadImpl::Rms(state) => {
                out.push_str(&format!("- rms_threshold: {}\n", state.rms_threshold));
                out.push_str(&format!("- rms_silence_sec: {}\n", VAD_SILENCE_SEC));
                out.push_str(&format!("- rms_max_buffer_sec: {}\n", VAD_MAX_BUFFER_SEC));
            }
        }
    }

    fn reset(&mut self) {
        match &mut self.inner {
            VadImpl::Disabled => {}
            VadImpl::Silero(state) => {
                state.detector.clear();
                state.detector.reset();
            }
            VadImpl::Rms(state) => state.reset(),
        }
        self.total_samples_seen = 0;
        self.segment_count = 0;
        self.last_segment_start_sample = None;
        self.last_segment_samples = 0;
    }
}

impl SileroVadState {
    fn new() -> Result<Self> {
        let settings = SileroVadSettings::from_env();
        if !settings.model_path.exists() {
            bail!(
                "Silero VAD 模型文件缺失: {}\n请先运行 ./scripts/download-silero-vad.sh 下载，或用 SILERO_VAD_MODEL 指定路径。",
                settings.model_path.display()
            );
        }
        let cfg = VadModelConfig {
            silero_vad: SileroVadModelConfig {
                model: Some(settings.model_path.to_string_lossy().into_owned()),
                threshold: settings.threshold,
                min_silence_duration: settings.min_silence_sec,
                min_speech_duration: settings.min_speech_sec,
                window_size: SILERO_VAD_WINDOW_SIZE,
                max_speech_duration: settings.max_speech_sec,
            },
            ten_vad: Default::default(),
            sample_rate: VAD_SAMPLE_RATE as i32,
            num_threads: 1,
            provider: Some("cpu".into()),
            debug: false,
        };
        let detector = with_stdout_suppressed(|| {
            VoiceActivityDetector::create(&cfg, SILERO_VAD_BUFFER_SEC)
                .context("创建 Silero VoiceActivityDetector 失败")
        })?;
        Ok(Self { detector, settings })
    }

    fn push(&mut self, pcm: &[f32]) -> Vec<VadSegment> {
        self.detector.accept_waveform(pcm);
        let mut segments = Vec::new();
        while let Some(front) = self.detector.front() {
            let start_sample = front.start().max(0) as usize;
            let samples = front.samples().to_vec();
            drop(front);
            self.detector.pop();
            if !samples.is_empty() {
                segments.push(VadSegment {
                    start_sample,
                    samples,
                    reason: "speech",
                });
            }
        }
        segments
    }
}

impl SileroVadSettings {
    fn from_env() -> Self {
        Self {
            model_path: silero_vad_model_path(),
            threshold: env_f32("SILERO_VAD_THRESHOLD", SILERO_VAD_THRESHOLD),
            min_silence_sec: env_f32("SILERO_VAD_MIN_SILENCE_SEC", SILERO_VAD_MIN_SILENCE_SEC),
            min_speech_sec: env_f32("SILERO_VAD_MIN_SPEECH_SEC", SILERO_VAD_MIN_SPEECH_SEC),
            max_speech_sec: env_f32("SILERO_VAD_MAX_SPEECH_SEC", SILERO_VAD_MAX_SPEECH_SEC),
        }
    }
}

impl RmsVadState {
    fn new() -> Self {
        Self {
            buffer: Vec::new(),
            buffer_start_sample: 0,
            speaking: false,
            silence_samples: 0,
            rms_threshold: env_f32("VAD_RMS_THRESHOLD", VAD_RMS_THRESHOLD),
        }
    }

    fn push(&mut self, pcm: &[f32], frame_start_sample: usize) -> Vec<VadSegment> {
        let frame_rms = rms(pcm);
        if self.buffer.is_empty() {
            self.buffer_start_sample = frame_start_sample;
        }
        self.buffer.extend_from_slice(pcm);

        let now_speaking = frame_rms >= self.rms_threshold;
        if now_speaking {
            self.speaking = true;
            self.silence_samples = 0;
        } else if self.speaking {
            self.silence_samples += pcm.len();
        }

        let silence_limit = (VAD_SILENCE_SEC * VAD_SAMPLE_RATE as f32) as usize;
        let buffer_limit = (VAD_MAX_BUFFER_SEC * VAD_SAMPLE_RATE as f32) as usize;

        // 触发条件 1：说话后静音超过 silence_limit
        let silence_triggered = self.speaking && self.silence_samples >= silence_limit;
        // 触发条件 2：缓冲达到上限强制切（兜底，防止背景音乐持续高于阈值）
        let force_triggered = self.buffer.len() >= buffer_limit;

        if silence_triggered || force_triggered {
            if self.buffer.is_empty() {
                return Vec::new();
            }
            let reason = if force_triggered {
                "max-buffer"
            } else {
                "silence"
            };
            let start_sample = self.buffer_start_sample;
            let segment = std::mem::take(&mut self.buffer);
            self.speaking = false;
            self.silence_samples = 0;
            vec![VadSegment {
                start_sample,
                samples: segment,
                reason,
            }]
        } else {
            Vec::new()
        }
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.buffer_start_sample = 0;
        self.speaking = false;
        self.silence_samples = 0;
    }
}

// ─── 统一槽位枚举 ─────────────────────────────────────────────────────

enum AnySlot {
    Online(OnlineSlot),
    Offline(OfflineSlot),
}

impl AnySlot {
    fn name(&self) -> &str {
        match self {
            AnySlot::Online(s) => s.name,
            AnySlot::Offline(s) => s.name,
        }
    }

    fn is_online(&self) -> bool {
        matches!(self, AnySlot::Online(_))
    }

    fn build(desc: &ModelDesc) -> Result<Self> {
        match desc.kind {
            SlotKind::Online(_) => Ok(Self::Online(build_online_slot(desc)?)),
            SlotKind::Offline(_) => Ok(Self::Offline(build_offline_slot(desc)?)),
        }
    }
}

/// 给 AnySlot 的字段访问提供统一接口（渲染/报告/toggle 共用）。
trait SlotView {
    fn partial(&self) -> &str;
    fn finals(&self) -> &[String];
    fn all_finals(&self) -> &[String];
    fn enabled(&self) -> bool;
    fn finals_scroll(&self) -> u16;
    fn finals_scroll_mut(&mut self) -> &mut u16;
    fn set_enabled(&mut self, enabled: bool);
    fn clear(&mut self);
}

impl SlotView for OnlineSlot {
    fn partial(&self) -> &str {
        &self.partial
    }
    fn finals(&self) -> &[String] {
        &self.finals
    }
    fn all_finals(&self) -> &[String] {
        &self.all_finals
    }
    fn enabled(&self) -> bool {
        self.enabled
    }
    fn finals_scroll(&self) -> u16 {
        self.finals_scroll
    }
    fn finals_scroll_mut(&mut self) -> &mut u16 {
        &mut self.finals_scroll
    }
    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if !enabled {
            self.partial.clear();
        }
    }
    fn clear(&mut self) {
        self.partial.clear();
        self.all_partials.clear();
        self.finals.clear();
        self.all_finals.clear();
        self.finals_scroll = 0;
    }
}

impl SlotView for OfflineSlot {
    fn partial(&self) -> &str {
        &self.partial
    }
    fn finals(&self) -> &[String] {
        &self.finals
    }
    fn all_finals(&self) -> &[String] {
        &self.all_finals
    }
    fn enabled(&self) -> bool {
        self.enabled
    }
    fn finals_scroll(&self) -> u16 {
        self.finals_scroll
    }
    fn finals_scroll_mut(&mut self) -> &mut u16 {
        &mut self.finals_scroll
    }
    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if !enabled {
            self.partial.clear();
        }
    }
    fn clear(&mut self) {
        self.partial.clear();
        self.finals.clear();
        self.all_finals.clear();
        self.finals_scroll = 0;
        self.segments_decoded = 0;
        self.last_segment_samples = 0;
    }
}

impl SlotView for AnySlot {
    fn partial(&self) -> &str {
        match self {
            AnySlot::Online(s) => s.partial(),
            AnySlot::Offline(s) => s.partial(),
        }
    }
    fn finals(&self) -> &[String] {
        match self {
            AnySlot::Online(s) => s.finals(),
            AnySlot::Offline(s) => s.finals(),
        }
    }
    fn all_finals(&self) -> &[String] {
        match self {
            AnySlot::Online(s) => s.all_finals(),
            AnySlot::Offline(s) => s.all_finals(),
        }
    }
    fn enabled(&self) -> bool {
        match self {
            AnySlot::Online(s) => s.enabled(),
            AnySlot::Offline(s) => s.enabled(),
        }
    }
    fn finals_scroll(&self) -> u16 {
        match self {
            AnySlot::Online(s) => s.finals_scroll(),
            AnySlot::Offline(s) => s.finals_scroll(),
        }
    }
    fn finals_scroll_mut(&mut self) -> &mut u16 {
        match self {
            AnySlot::Online(s) => s.finals_scroll_mut(),
            AnySlot::Offline(s) => s.finals_scroll_mut(),
        }
    }
    fn set_enabled(&mut self, enabled: bool) {
        match self {
            AnySlot::Online(s) => s.set_enabled(enabled),
            AnySlot::Offline(s) => s.set_enabled(enabled),
        }
    }
    fn clear(&mut self) {
        match self {
            AnySlot::Online(s) => s.clear(),
            AnySlot::Offline(s) => s.clear(),
        }
    }
}

// ─── 设备选择与采集抽象 ──────────────────────────────────────────────

/// 用户在设备选择屏挑中的采集源。
/// Input 存 cpal 用的设备名，Output 存 CoreAudio 的设备 UID。
enum DevicePick {
    Input(String),
    Output(String),
}

impl DevicePick {
    /// 给 UI 显示用的标签，如「输入端 / MacBook 麦克风」。
    fn label(&self, _inputs: &[String], outputs: &[OutputDevice]) -> String {
        match self {
            DevicePick::Input(name) => {
                if name == "default" {
                    "输入端 / 默认麦克风".to_string()
                } else {
                    format!("输入端 / {name}")
                }
            }
            DevicePick::Output(uid) => {
                let dev_name = outputs
                    .iter()
                    .find(|d| &d.uid == uid)
                    .map(|d| d.name.as_str())
                    .unwrap_or(uid);
                format!("输出端 / {dev_name}")
            }
        }
    }
}

/// 把 system_audio（Process Tap）和 mic（cpal）两条采集路径收拢成统一接口。
/// 预览屏和主屏都只跟 Capture 打交道，不再 if 分支。
struct Capture {
    rx: mpsc::Receiver<Vec<f32>>,
    stop_flag: Arc<AtomicBool>,
    /// 持有底层 stream/tap 保活；drop 即停止采集。
    _guard: CaptureGuard,
}

enum CaptureGuard {
    #[allow(dead_code)]
    System(system_audio::SystemAudioTap),
    #[allow(dead_code)]
    Mic(MicInput),
}

impl Capture {
    fn start(pick: &DevicePick) -> Result<Capture> {
        let stop_flag = Arc::new(AtomicBool::new(false));
        match pick {
            DevicePick::Output(uid) => {
                let rebuild = Arc::new(AtomicBool::new(false));
                let (tap, rx) =
                    system_audio::start(stop_flag.clone(), rebuild, Some(uid.as_str()))?;
                Ok(Capture {
                    rx,
                    stop_flag,
                    _guard: CaptureGuard::System(tap),
                })
            }
            DevicePick::Input(name) => {
                let resolved = if name.is_empty() {
                    "default"
                } else {
                    name.as_str()
                };
                let (mic, rx) = MicInput::start_selected_with_stop(resolved, stop_flag.clone())?;
                Ok(Capture {
                    rx,
                    stop_flag,
                    _guard: CaptureGuard::Mic(mic),
                })
            }
        }
    }

    fn stop(&self) {
        self.stop_flag.store(true, Ordering::Release);
    }
}

/// 枚举输入端（麦克风）。cpal 失败时返回空 Vec，由调用方处理。
fn list_inputs() -> Vec<String> {
    let host = cpal::default_host();
    match host.input_devices() {
        Ok(devs) => devs
            .filter_map(|d| d.name().ok())
            .filter(|n| !n.trim().is_empty())
            .collect(),
        Err(_) => Vec::new(),
    }
}

/// 枚举输出端（扬声器/系统音频）。
/// core 的 `list_output_devices()` 已按 Output scope 通道数过滤掉麦克风，
/// 这里只做同名去重（虚拟声卡多实例场景）。
fn list_outputs() -> Vec<OutputDevice> {
    let mut raw = system_audio::list_output_devices().unwrap_or_default();
    raw.dedup_by(|a, b| a.name == b.name && a.uid == b.uid);
    raw
}

// ─── 应用状态 ─────────────────────────────────────────────────────────

#[derive(Clone)]
struct SubtitleCue {
    index: usize,
    start_ms: u64,
    end_ms: u64,
    text: String,
}

struct MediaState {
    audio_path: PathBuf,
    srt_path: PathBuf,
    cues: Vec<SubtitleCue>,
    duration_ms: u64,
    position_ms: u64,
    anchor_ms: u64,
    anchor_started_at: Instant,
    playing: bool,
    child: Option<Child>,
    last_report: Option<PathBuf>,
}

impl MediaState {
    fn load_default() -> Result<Option<Self>> {
        let audio_path = PathBuf::from(DEFAULT_MEDIA_MP3);
        let srt_path = PathBuf::from(DEFAULT_MEDIA_SRT);
        if !audio_path.exists() || !srt_path.exists() {
            return Ok(None);
        }
        let cues = parse_srt(&srt_path)?;
        let duration_ms = cues.last().map(|cue| cue.end_ms).unwrap_or(0);
        Ok(Some(Self {
            audio_path,
            srt_path,
            cues,
            duration_ms,
            position_ms: 0,
            anchor_ms: 0,
            anchor_started_at: Instant::now(),
            playing: false,
            child: None,
            last_report: None,
        }))
    }

    fn start(&mut self) -> Result<()> {
        self.stop_child();
        self.anchor_ms = self.position_ms;
        self.anchor_started_at = Instant::now();
        self.playing = true;
        self.child = Some(spawn_ffplay(&self.audio_path, self.position_ms)?);
        Ok(())
    }

    fn stop_child(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }

    fn pause(&mut self) {
        self.refresh_position();
        self.playing = false;
        self.stop_child();
    }

    fn toggle_play(&mut self) -> Result<()> {
        if self.playing {
            self.pause();
            Ok(())
        } else {
            self.start()
        }
    }

    fn refresh_position(&mut self) {
        if self.playing {
            self.position_ms = self.anchor_ms + self.anchor_started_at.elapsed().as_millis() as u64;
        }
    }

    fn seek_by(&mut self, delta_ms: i64) -> Result<()> {
        self.refresh_position();
        self.position_ms = if delta_ms.is_negative() {
            self.position_ms.saturating_sub(delta_ms.unsigned_abs())
        } else {
            self.position_ms.saturating_add(delta_ms as u64)
        };
        if self.duration_ms > 0 {
            self.position_ms = self.position_ms.min(self.duration_ms);
        }
        if self.playing {
            self.start()?;
        }
        Ok(())
    }

    fn active_index(&self) -> usize {
        if self.cues.is_empty() {
            return 0;
        }
        match self.cues.binary_search_by(|cue| {
            if self.position_ms < cue.start_ms {
                std::cmp::Ordering::Greater
            } else if self.position_ms > cue.end_ms {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Equal
            }
        }) {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1),
        }
    }
}

impl Drop for MediaState {
    fn drop(&mut self) {
        self.stop_child();
    }
}

fn parse_srt(path: &Path) -> Result<Vec<SubtitleCue>> {
    let raw =
        fs::read_to_string(path).with_context(|| format!("读取字幕失败: {}", path.display()))?;
    let normalized = raw.replace("\r\n", "\n");
    let mut cues = Vec::new();

    for block in normalized.split("\n\n") {
        let mut lines = block.lines().filter(|line| !line.trim().is_empty());
        let Some(index_line) = lines.next() else {
            continue;
        };
        let Ok(index) = index_line.trim().parse::<usize>() else {
            continue;
        };
        let Some(time_line) = lines.next() else {
            continue;
        };
        let Some((start, end)) = time_line.split_once("-->") else {
            continue;
        };
        let text = lines.collect::<Vec<_>>().join(" ").trim().to_string();
        if text.is_empty() {
            continue;
        }
        cues.push(SubtitleCue {
            index,
            start_ms: parse_srt_time(start.trim())?,
            end_ms: parse_srt_time(end.trim())?,
            text,
        });
    }

    Ok(cues)
}

fn parse_srt_time(input: &str) -> Result<u64> {
    let Some((hms, millis)) = input.split_once(',') else {
        anyhow::bail!("无效字幕时间: {input}");
    };
    let parts = hms
        .split(':')
        .map(str::parse::<u64>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if parts.len() != 3 {
        anyhow::bail!("无效字幕时间: {input}");
    }
    let millis = millis.trim().parse::<u64>()?;
    Ok(((parts[0] * 3600 + parts[1] * 60 + parts[2]) * 1000) + millis)
}

fn spawn_ffplay(audio_path: &Path, position_ms: u64) -> Result<Child> {
    let seek_seconds = format!("{:.3}", position_ms as f64 / 1000.0);
    Command::new("ffplay")
        .arg("-nodisp")
        .arg("-autoexit")
        .arg("-loglevel")
        .arg("quiet")
        .arg("-ss")
        .arg(seek_seconds)
        .arg(audio_path)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .with_context(|| "启动 ffplay 失败，请确认 ffplay 在 PATH 中")
}

fn format_media_time(ms: u64) -> String {
    let total = ms / 1000;
    let h = total / 3600;
    let m = (total % 3600) / 60;
    let s = total % 60;
    if h > 0 {
        format!("{h:02}:{m:02}:{s:02}")
    } else {
        format!("{m:02}:{s:02}")
    }
}

fn media_progress_bar(position_ms: u64, duration_ms: u64, width: usize) -> String {
    if duration_ms == 0 || width == 0 {
        return " ".repeat(width);
    }
    let filled = ((position_ms.min(duration_ms) as f64 / duration_ms as f64) * width as f64)
        .round()
        .clamp(0.0, width as f64) as usize;
    format!("{}{}", "█".repeat(filled), "░".repeat(width - filled))
}

fn export_report(app: &App) -> Result<PathBuf> {
    fs::create_dir_all("reports").context("创建 reports 目录失败")?;
    let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let path = PathBuf::from(format!("reports/asr_report_{now}.md"));
    let mut out = String::new();

    out.push_str("# ASR Compare Report\n\n");
    out.push_str(&format!("- source: {}\n", app.source_label));
    out.push_str(&format!(
        "- runtime: {}\n",
        humantime_elapsed(app.started_at)
    ));

    if let Some(media) = &app.media {
        out.push_str(&format!("- audio: {}\n", media.audio_path.display()));
        out.push_str(&format!("- subtitle: {}\n", media.srt_path.display()));
        out.push_str(&format!(
            "- position: {} ({:.3}s)\n",
            format_media_time(media.position_ms),
            media.position_ms as f64 / 1000.0
        ));
        out.push_str("\n## Subtitle Context\n\n");
        if media.cues.is_empty() {
            out.push_str("<no subtitles parsed>\n");
        } else {
            let active = media.active_index();
            let start = active.saturating_sub(3);
            let end = (active + 4).min(media.cues.len());
            for cue in &media.cues[start..end] {
                let marker = if cue.index == media.cues[active].index {
                    ">"
                } else {
                    "-"
                };
                out.push_str(&format!(
                    "{marker} [{} - {}] {}\n",
                    format_media_time(cue.start_ms),
                    format_media_time(cue.end_ms),
                    cue.text
                ));
            }
        }
    }

    app.vad.append_report(&mut out);

    out.push_str("\n## ASR Results\n");
    for slot in &app.slots {
        out.push_str(&format!("\n### {}\n\n", slot.name()));
        if !slot.partial().trim().is_empty() {
            out.push_str(&format!("partial: {}\n\n", slot.partial().trim()));
        }
        if let AnySlot::Online(s) = slot {
            if !s.all_partials.is_empty() {
                out.push_str(&format!(
                    "partial_history_count: {}\n\n",
                    s.all_partials.len()
                ));
                out.push_str("partial_history:\n\n");
                for (idx, partial) in s.all_partials.iter().enumerate() {
                    out.push_str(&format!("{}. {}\n", idx + 1, partial.trim()));
                }
                out.push('\n');
            }
        }
        if let AnySlot::Offline(s) = slot {
            out.push_str(&format!("- family: {}\n", s.family.as_str()));
            out.push_str(&format!("- decoded_segments: {}\n", s.segments_decoded));
            if s.last_segment_samples > 0 {
                out.push_str(&format!(
                    "- last_decoded_segment_sec: {:.3}\n",
                    s.last_segment_samples as f32 / VAD_SAMPLE_RATE as f32
                ));
            }
            out.push('\n');
        }
        if slot.all_finals().is_empty() {
            out.push_str("finals: <none>\n");
        } else {
            out.push_str(&format!("final_count: {}\n\n", slot.all_finals().len()));
            for (idx, final_text) in slot.all_finals().iter().enumerate() {
                out.push_str(&format!("{}. {}\n", idx + 1, final_text.trim()));
            }
        }
    }

    fs::write(&path, out).with_context(|| format!("写入报告失败: {}", path.display()))?;
    Ok(path)
}

struct App {
    slots: Vec<AnySlot>,
    /// 离线槽共享的 VAD 状态机。喂音时 Online 槽增量吃帧，离线槽等 VAD 触发。
    vad: VadState,
    log: Vec<String>,
    last_rms: f32,
    started_at: Instant,
    source_label: String,
    active_slot: usize,
    media: Option<MediaState>,
}

impl App {
    fn clear(&mut self) {
        for slot in &mut self.slots {
            slot.clear();
        }
        self.vad.reset();
        self.log.clear();
    }

    fn toggle(&mut self, index: usize) {
        if let Some(slot) = self.slots.get_mut(index) {
            let new_enabled = !slot.enabled();
            slot.set_enabled(new_enabled);
            let name = slot.name().to_string();
            let state = if new_enabled { "启用" } else { "禁用" };
            self.log.push(format!("[{name}] 已{state}"));
            if self.log.len() > 50 {
                self.log.remove(0);
            }
        }
    }

    fn active_slot_mut(&mut self) -> Option<&mut AnySlot> {
        self.slots.get_mut(self.active_slot)
    }

    fn move_active_slot(&mut self, delta: isize) {
        if self.slots.is_empty() {
            self.active_slot = 0;
            return;
        }
        let last = self.slots.len().saturating_sub(1) as isize;
        let next = (self.active_slot as isize + delta).clamp(0, last);
        self.active_slot = next as usize;
    }

    fn scroll_active_slot(&mut self, delta: i16) {
        if let Some(slot) = self.active_slot_mut() {
            let s = slot.finals_scroll_mut();
            if delta.is_negative() {
                *s = s.saturating_sub(delta.unsigned_abs());
            } else {
                *s = s.saturating_add(delta as u16);
            }
        }
    }

    fn reset_active_scroll(&mut self) {
        if let Some(slot) = self.active_slot_mut() {
            *slot.finals_scroll_mut() = 0;
        }
    }

    fn refresh_media(&mut self) {
        if let Some(media) = &mut self.media {
            media.refresh_position();
        }
    }

    fn handle_media_key(&mut self, key: &KeyEvent) -> Result<()> {
        match key.code {
            KeyCode::Char(' ') => {
                if let Some(media) = &mut self.media {
                    media.toggle_play()?;
                }
            }
            KeyCode::Char('h') => {
                if let Some(media) = &mut self.media {
                    media.seek_by(-10_000)?;
                }
            }
            KeyCode::Char('l') => {
                if let Some(media) = &mut self.media {
                    media.seek_by(10_000)?;
                }
            }
            KeyCode::Char('H') => {
                if let Some(media) = &mut self.media {
                    media.seek_by(-60_000)?;
                }
            }
            KeyCode::Char('L') => {
                if let Some(media) = &mut self.media {
                    media.seek_by(60_000)?;
                }
            }
            KeyCode::Char('e') => {
                let path = export_report(self)?;
                if let Some(media) = &mut self.media {
                    media.last_report = Some(path.clone());
                }
                self.log.push(format!("报告已导出: {}", path.display()));
                if self.log.len() > 50 {
                    self.log.remove(0);
                }
            }
            _ => {}
        }
        Ok(())
    }
}

// ─── 启动选择界面 ─────────────────────────────────────────────────────

fn run_selection_screen(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
) -> Result<Option<Vec<usize>>> {
    // 每个模型是否就绪：流式模型恒 true，离线模型检测文件是否齐全。
    // 不就绪的模型在选择屏打灰、不可勾选（优雅降级）。
    let available: Vec<bool> = MODEL_DESCS.iter().map(|d| d.files_present()).collect();
    let mut selected: Vec<bool> = MODEL_DESCS
        .iter()
        .enumerate()
        .map(|(i, desc)| available[i] && matches!(desc.kind, SlotKind::Offline(_)))
        .collect();
    if !selected.iter().any(|&s| s) {
        selected = available.clone();
    }

    loop {
        terminal.draw(|frame| {
            let area = frame.area();
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(3),
                    Constraint::Min(6),
                    Constraint::Length(3),
                ])
                .split(area);

            let title = Paragraph::new(Line::from(vec![Span::styled(
                " ASR Compare — 选择要加载的引擎 ",
                Style::default().add_modifier(Modifier::BOLD),
            )]))
            .block(Block::default().borders(Borders::ALL));
            frame.render_widget(title, chunks[0]);

            let mut lines: Vec<Line> = Vec::new();
            for (i, desc) in MODEL_DESCS.iter().enumerate() {
                let ok = available[i];
                let check = if !ok {
                    "    " // 缺文件：不可勾选
                } else if selected[i] {
                    "[x] "
                } else {
                    "[ ] "
                };
                let lang = match desc.language {
                    Language::Chinese => "中文",
                    Language::English => "英文",
                };
                let kind_tag = match desc.kind {
                    SlotKind::Online(_) => "流式",
                    SlotKind::Offline(_) => "离线+VAD",
                };
                let style = if !ok {
                    Style::default().fg(Color::DarkGray)
                } else if selected[i] {
                    Style::default().fg(Color::Green)
                } else {
                    Style::default().fg(Color::Gray)
                };
                let key_span = if ok {
                    Span::styled(format!(" {} ", i + 1), Style::default().fg(Color::Yellow))
                } else {
                    Span::raw(format!(" {} ", i + 1))
                };
                let suffix = if !ok {
                    desc.missing_files_hint()
                        .map(|script| format!("  (未下载: {script})"))
                        .unwrap_or_default()
                } else {
                    String::new()
                };
                lines.push(Line::from(vec![
                    key_span,
                    Span::styled(
                        format!("{check}{} ({lang}/{kind_tag}){}", desc.name, suffix),
                        style,
                    ),
                ]));
            }
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                " 默认只勾选离线模型 · 可手动加入流式模型混跑 · Enter 确认 · q 退出 ",
                Style::default().fg(Color::DarkGray),
            )));
            let list = Paragraph::new(lines)
                .block(Block::default().borders(Borders::ALL).title("选择引擎"));
            frame.render_widget(list, chunks[1]);

            let count = selected.iter().filter(|&&s| s).count();
            let hint = if count == 0 {
                " 请至少选择一个引擎".to_string()
            } else {
                format!(" 将加载 {count} 个引擎（模型较大，加载需几秒）")
            };
            let hint_style = if count == 0 {
                Style::default().fg(Color::Red)
            } else {
                Style::default().fg(Color::Green)
            };
            let footer = Paragraph::new(Line::from(Span::styled(hint, hint_style)))
                .block(Block::default().borders(Borders::ALL));
            frame.render_widget(footer, chunks[2]);
        })?;

        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => return Ok(None),
                    KeyCode::Char(c @ '1'..='9') => {
                        let idx = (c as u8 - b'1') as usize;
                        if idx < selected.len() && available[idx] {
                            selected[idx] = !selected[idx];
                        }
                    }
                    KeyCode::Enter | KeyCode::Char('\n') | KeyCode::Char('\r') => {
                        let indices: Vec<usize> = selected
                            .iter()
                            .enumerate()
                            .filter(|(_, &s)| s)
                            .map(|(i, _)| i)
                            .collect();
                        if !indices.is_empty() {
                            return Ok(Some(indices));
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}

// ─── 设备选择屏 ───────────────────────────────────────────────────────

/// 上下两区列出输入/输出设备，方向键跨区扁平导航，Enter 选中。
/// 返回 Some(pick) 表示用户已选定；None 表示按 q/Esc 取消（退回模型选择屏）。
fn run_device_screen(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
) -> Result<Option<(DevicePick, Vec<String>, Vec<OutputDevice>)>> {
    // 进入时一次性枚举。失败用空列表，UI 会显示「无可用设备」。
    let inputs = list_inputs();
    let outputs = list_outputs();

    // 扁平游标：把 (区, 索引) 视为一个一维序列。
    // region: 0=输入区，1=输出区。
    let total = inputs.len() + outputs.len();
    let mut cursor: usize = 0;

    // 初始光标落在第一个非空区的第一项，避免停在空区。
    if inputs.is_empty() && !outputs.is_empty() {
        cursor = inputs.len(); // 跳到输出区第一项
    }

    loop {
        terminal.draw(|frame| {
            let area = frame.area();
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(3),
                    Constraint::Min(8),
                    Constraint::Length(3),
                ])
                .split(area);

            let title = Paragraph::new(Line::from(vec![Span::styled(
                " 选择采集源 ",
                Style::default().add_modifier(Modifier::BOLD),
            )]))
            .block(Block::default().borders(Borders::ALL));
            frame.render_widget(title, chunks[0]);

            // 上下两区按内容高度分配。中间区域竖直拆分。
            let input_h = (inputs.len() + 2).max(3) as u16;
            let output_h = (outputs.len() + 2).max(3) as u16;
            let body = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(input_h as u16),
                    Constraint::Min(output_h as u16),
                ])
                .split(chunks[1]);

            // 输入区
            let mut in_lines: Vec<Line> = Vec::new();
            if inputs.is_empty() {
                in_lines.push(Line::from(Span::styled(
                    " （无可用输入设备）",
                    Style::default().fg(Color::DarkGray),
                )));
            }
            for (i, name) in inputs.iter().enumerate() {
                let flat = i;
                let mark = if flat == cursor { "▸" } else { " " };
                let style = if flat == cursor {
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default()
                };
                in_lines.push(Line::from(vec![
                    Span::raw(format!(" {mark} ")),
                    Span::styled(format!("[I{}] ", i + 1), Style::default().fg(Color::Cyan)),
                    Span::styled(name.clone(), style),
                ]));
            }
            let in_para = Paragraph::new(in_lines).block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" 输入端（麦克风） "),
            );
            frame.render_widget(in_para, body[0]);

            // 输出区
            let mut out_lines: Vec<Line> = Vec::new();
            if outputs.is_empty() {
                out_lines.push(Line::from(Span::styled(
                    " （无可用输出设备）",
                    Style::default().fg(Color::DarkGray),
                )));
            }
            for (i, dev) in outputs.iter().enumerate() {
                let flat = inputs.len() + i;
                let mark = if flat == cursor { "▸" } else { " " };
                let style = if flat == cursor {
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default()
                };
                out_lines.push(Line::from(vec![
                    Span::raw(format!(" {mark} ")),
                    Span::styled(format!("[O{}] ", i + 1), Style::default().fg(Color::Cyan)),
                    Span::styled(dev.name.clone(), style),
                ]));
            }
            let out_para = Paragraph::new(out_lines).block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" 输出端（系统音频） "),
            );
            frame.render_widget(out_para, body[1]);

            let hint = if total == 0 {
                Span::styled(
                    " 无可用设备，请检查麦克风/扬声器 · q 返回",
                    Style::default().fg(Color::Red),
                )
            } else {
                Span::styled(
                    " ↑/↓ 移动 · Enter 选中并预览 · q 返回 ",
                    Style::default().fg(Color::DarkGray),
                )
            };
            let footer =
                Paragraph::new(Line::from(hint)).block(Block::default().borders(Borders::ALL));
            frame.render_widget(footer, chunks[2]);
        })?;

        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => return Ok(None),
                    KeyCode::Up => {
                        if cursor > 0 {
                            cursor -= 1;
                        }
                    }
                    KeyCode::Down => {
                        if cursor + 1 < total {
                            cursor += 1;
                        }
                    }
                    KeyCode::Enter | KeyCode::Char('\n') | KeyCode::Char('\r') => {
                        if cursor < inputs.len() {
                            let name = inputs[cursor].clone();
                            return Ok(Some((
                                DevicePick::Input(name),
                                inputs.clone(),
                                outputs.clone(),
                            )));
                        } else if cursor < total {
                            let dev = &outputs[cursor - inputs.len()];
                            let uid = dev.uid.clone();
                            return Ok(Some((
                                DevicePick::Output(uid),
                                inputs.clone(),
                                outputs.clone(),
                            )));
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}

// ─── 预览屏 ───────────────────────────────────────────────────────────

/// 用选中的设备立即启动采集，只算 RMS 不喂 ASR，让用户确认「这个设备有声音进来」。
/// 返回 Ok(Some(capture))：Enter 进入主屏（采集对象移交，不重启）。
/// 返回 Ok(None)：Esc 返回设备选择屏（停止采集）。
fn run_preview_screen(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    pick: &DevicePick,
    inputs: &[String],
    outputs: &[OutputDevice],
) -> Result<Option<Capture>> {
    let label = pick.label(inputs, outputs);
    let mut capture = match Capture::start(pick) {
        Ok(c) => c,
        Err(e) => {
            // 启动失败：短暂展示错误，等用户按键返回。
            show_error_screen(terminal, &label, &format!("{e:#}"))?;
            return Ok(None);
        }
    };
    let mut rms_disp: f32 = 0.0;
    let preview_started_at = Instant::now();
    let mut received_frames = 0usize;

    loop {
        // 消费已有帧算 RMS（不喂 ASR）。
        while let Ok(frame) = capture.rx.try_recv() {
            received_frames += 1;
            rms_disp = rms(&frame).max(rms_disp * 0.9);
        }

        let show_output_tcc_hint = matches!(pick, DevicePick::Output(_))
            && received_frames == 0
            && preview_started_at.elapsed() >= Duration::from_secs(3);

        terminal.draw(|frame| {
            let area = frame.area();
            let block = Block::default().borders(Borders::ALL).title(" 采集预览 ");
            let inner = block.inner(area);
            frame.render_widget(block, area);

            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(7), Constraint::Length(3), Constraint::Min(1)])
                .split(inner);

            let mut info_lines = vec![
                Line::from(vec![
                    Span::styled(" 选中源：", Style::default().fg(Color::Cyan)),
                    Span::raw(label.clone()),
                ]),
                Line::from(""),
                Line::from(Span::styled(
                    " 对着麦克风说话 / 播放声音，观察下方电平是否跳动",
                    Style::default().fg(Color::DarkGray),
                )),
            ];
            if show_output_tcc_hint {
                info_lines.push(Line::from(""));
                info_lines.push(Line::from(Span::styled(
                    " 3 秒内仍未收到任何系统音频帧：若你是从终端 cargo run，macOS 可能在 TCC 层拒绝了 AudioCapture。",
                    Style::default().fg(Color::Yellow),
                )));
                info_lines.push(Line::from(Span::styled(
                    " 常见场景是终端宿主缺少 NSAudioCaptureUsageDescription；请改用带该键的 .app 宿主或先跑 diag_devices 核对。",
                    Style::default().fg(Color::Yellow),
                )));
            }
            frame.render_widget(Paragraph::new(info_lines), chunks[0]);

            // 电平条：用 ASCII block 字符铺 30 格。
            let bars = 30;
            let filled = ((rms_disp * 3.0 * bars as f32).round() as i64).clamp(0, bars as i64) as usize;
            let bar: String = "█".repeat(filled);
            let rest: String = "░".repeat(bars - filled);
            let level = Paragraph::new(Line::from(vec![
                Span::raw(" "),
                Span::styled(bar, Style::default().fg(Color::Green)),
                Span::styled(rest, Style::default().fg(Color::DarkGray)),
                Span::raw(format!("  RMS {:.4}", rms_disp)),
            ]));
            frame.render_widget(level, chunks[1]);

            let hint = Paragraph::new(Line::from(Span::styled(
                " Enter 开始采集并进入对比 · Esc 返回重选 ",
                Style::default().fg(Color::DarkGray),
            )));
            frame.render_widget(hint, chunks[2]);
        })?;

        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Esc | KeyCode::Char('q') => {
                        capture.stop();
                        return Ok(None);
                    }
                    KeyCode::Enter | KeyCode::Char('\n') | KeyCode::Char('\r') => {
                        return Ok(Some(capture));
                    }
                    _ => {}
                }
            }
        }
    }
}

/// 启动采集失败时展示一行错误，按任意键返回。
fn show_error_screen(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    label: &str,
    err: &str,
) -> Result<()> {
    loop {
        terminal.draw(|frame| {
            let para = Paragraph::new(vec![
                Line::from(Span::styled(
                    format!(" 启动采集失败：{label} "),
                    Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
                )),
                Line::from(""),
                Line::from(err.to_string()),
                Line::from(""),
                Line::from(Span::styled(
                    " 按任意键返回重选 ",
                    Style::default().fg(Color::DarkGray),
                )),
            ])
            .block(Block::default().borders(Borders::ALL).title("错误"));
            frame.render_widget(para, frame.area());
        })?;
        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(_) = event::read()? {
                return Ok(());
            }
        }
    }
}

// ─── 主函数 ────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
        prev_hook(info);
    }));

    // 阶段 1：选择引擎。
    let indices = match run_selection_screen(&mut terminal)? {
        Some(i) => i,
        None => {
            disable_raw_mode()?;
            execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
            terminal.show_cursor()?;
            println!("已取消");
            return Ok(());
        }
    };

    // 阶段 2：加载选中的模型。
    terminal.draw(|frame| {
        let area = frame.area();
        let names: Vec<&str> = indices.iter().map(|&i| MODEL_DESCS[i].name).collect();
        let msg = format!(" 正在加载: {} ... ", names.join(", "));
        let loading = Paragraph::new(Line::from(Span::styled(
            msg,
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        )))
        .block(Block::default().borders(Borders::ALL).title("加载中"));
        frame.render_widget(loading, area);
    })?;

    let mut slots = Vec::with_capacity(indices.len());
    for &i in &indices {
        slots.push(AnySlot::build(&MODEL_DESCS[i])?);
    }
    let has_offline_slots = slots.iter().any(|slot| !slot.is_online());
    let vad = VadState::new(has_offline_slots)?;

    // 阶段 3 + 4：设备选择 → 预览。两屏之间可往返（预览 Esc 回设备选择）。
    let (mut rx, source_label) = loop {
        let (pick, inputs, outputs) = match run_device_screen(&mut terminal)? {
            Some(v) => v,
            None => {
                // 取消：退回模型选择。这里直接退出（简化），与原 q 取消语义一致。
                disable_raw_mode()?;
                execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
                terminal.show_cursor()?;
                println!("已取消");
                return Ok(());
            }
        };
        match run_preview_screen(&mut terminal, &pick, &inputs, &outputs)? {
            Some(capture) => {
                let label = pick.label(&inputs, &outputs);
                break (capture.rx, label);
            }
            None => continue, // Esc：回设备选择屏重选
        }
    };

    let media = MediaState::load_default()?;
    let mut app = App {
        slots,
        vad,
        log: Vec::new(),
        last_rms: 0.0,
        started_at: Instant::now(),
        source_label,
        active_slot: 0,
        media,
    };
    if let Some(media) = &mut app.media {
        match media.start() {
            Ok(()) => app.log.push(format!(
                "已开始播放: {}",
                media
                    .audio_path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
            )),
            Err(e) => app.log.push(format!("音频播放启动失败: {e:#}")),
        }
    }

    // 阶段 5：主 TUI 循环。
    let result = run_loop(&mut terminal, &mut app, &mut rx);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    println!("已退出（运行 {}）", humantime_elapsed(app.started_at));
    result
}

// ─── 主循环 ────────────────────────────────────────────────────────────

fn run_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
    rx: &mut tokio::sync::mpsc::Receiver<Vec<f32>>,
) -> Result<()> {
    loop {
        while let Ok(frame) = rx.try_recv() {
            app.last_rms = rms(&frame);

            // 1) 流式槽：增量喂当前帧（沿用原有 feed 逻辑）。
            for slot in &mut app.slots {
                if let AnySlot::Online(s) = slot {
                    if let Some(final_text) = feed_online_frame(s, &frame) {
                        push_log(&mut app.log, format!("[{}] final: {}", s.name, final_text));
                    }
                }
            }

            // 2) 离线槽：喂 VAD 状态机；触发时把整段广播给所有离线槽。
            for segment in app.vad.push(&frame, &mut app.log) {
                for slot in &mut app.slots {
                    if let AnySlot::Offline(s) = slot {
                        if let Some(final_text) = decode_offline_segment(s, &segment) {
                            push_log(&mut app.log, format!("[{}] final: {}", s.name, final_text));
                        }
                    }
                }
            }
        }

        if event::poll(POLL_INTERVAL)? {
            if let Event::Key(key) = event::read()? {
                if should_quit(&key) {
                    return Ok(());
                }
                if matches!(key.code, KeyCode::Char('c')) {
                    app.clear();
                }
                app.handle_media_key(&key)?;
                match key.code {
                    KeyCode::Left => app.move_active_slot(-1),
                    KeyCode::Right => app.move_active_slot(1),
                    KeyCode::Up => app.scroll_active_slot(-1),
                    KeyCode::Down => app.scroll_active_slot(1),
                    KeyCode::PageUp => app.scroll_active_slot(-6),
                    KeyCode::PageDown => app.scroll_active_slot(6),
                    KeyCode::Home => app.reset_active_scroll(),
                    KeyCode::Char(c @ '1'..='9') => {
                        let idx = (c as u8 - b'1') as usize;
                        app.toggle(idx);
                    }
                    _ => {}
                }
            }
        }

        app.refresh_media();
        terminal.draw(|frame| draw(app, frame))?;
    }
}

fn should_quit(key: &KeyEvent) -> bool {
    matches!(key.code, KeyCode::Char('q'))
        || (key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL))
        || matches!(key.code, KeyCode::Esc)
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum = samples.iter().map(|s| s * s).sum::<f32>();
    (sum / samples.len() as f32).sqrt()
}

fn humantime_elapsed(start: Instant) -> String {
    let s = start.elapsed().as_secs();
    format!("{}分{}秒", s / 60, s % 60)
}

fn push_log(log: &mut Vec<String>, message: impl Into<String>) {
    log.push(message.into());
    if log.len() > 200 {
        log.remove(0);
    }
}

// ─── 渲染 ─────────────────────────────────────────────────────────────

fn draw(app: &mut App, frame: &mut ratatui::Frame) {
    let area = frame.area();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(8),
            Constraint::Length(6),
        ])
        .split(area);

    let title = Line::from(vec![
        Span::styled(
            " ASR Compare ",
            Style::default().add_modifier(Modifier::BOLD),
        ),
        Span::raw("· "),
        Span::styled("←/→", Style::default().fg(Color::Yellow)),
        Span::raw(" 选列 · "),
        Span::styled("↑/↓", Style::default().fg(Color::Yellow)),
        Span::raw(" 滚动 · "),
        Span::styled("PgUp/PgDn", Style::default().fg(Color::Yellow)),
        Span::raw(" 快速滚动 · "),
        Span::styled("q", Style::default().fg(Color::Yellow)),
        Span::raw(" 退出 · "),
        Span::styled("c", Style::default().fg(Color::Yellow)),
        Span::raw(" 清空 · "),
        Span::styled("1-9", Style::default().fg(Color::Yellow)),
        Span::raw(" 切换引擎 · "),
        Span::raw(format!("音频 RMS {:.4}", app.last_rms)),
        Span::raw(" · 源:"),
        Span::styled(app.source_label.clone(), Style::default().fg(Color::Cyan)),
    ]);
    let top = Paragraph::new(title).block(Block::default().borders(Borders::ALL).title("实时对比"));
    frame.render_widget(top, chunks[0]);

    let asr_area = if let Some(media) = &app.media {
        let body = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(34), Constraint::Percentage(66)])
            .split(chunks[1]);
        render_media_panel(media, frame, body[0]);
        body[1]
    } else {
        chunks[1]
    };

    let count = app.slots.len().max(1);
    let col_constraints: Vec<Constraint> = (0..count)
        .map(|_| Constraint::Ratio(1, count as u32))
        .collect();
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints(col_constraints)
        .split(asr_area);

    for (index, (slot, col_area)) in app.slots.iter_mut().zip(columns.iter()).enumerate() {
        let is_active = index == app.active_slot;
        let dim = !slot.enabled();
        let is_offline = !slot.is_online();
        let text_style = if dim {
            Style::default().fg(Color::DarkGray)
        } else {
            Style::default()
        };

        // partial 区：流式槽显示实时 partial；离线槽显示 VAD 等待状态。
        let partial_label: String = if dim {
            "[已关闭]".to_string()
        } else if is_offline {
            app.vad.status_label()
        } else {
            slot.partial().to_string()
        };
        let partial_caption = if is_offline && !dim {
            "vad: "
        } else {
            "partial: "
        };
        let partial_lines = vec![Line::from(vec![
            Span::styled(partial_caption, Style::default().fg(Color::Cyan)),
            Span::styled(partial_label, text_style),
        ])];
        let title_suffix = if dim { " [关]" } else { "" };
        let title_style = if dim {
            Style::default().fg(Color::DarkGray)
        } else if is_active {
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().add_modifier(Modifier::BOLD)
        };
        let col_layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(5), Constraint::Min(5)])
            .split(*col_area);
        let partial_para = Paragraph::new(partial_lines)
            .wrap(Wrap { trim: false })
            .block(Block::default().borders(Borders::ALL).title(Span::styled(
                format!(" {}{} ", slot.name(), title_suffix),
                title_style,
            )));
        frame.render_widget(partial_para, col_layout[0]);

        let mut finals_lines: Vec<Line> = Vec::new();
        for f in slot.finals() {
            finals_lines.push(Line::from(Span::styled(format!("· {f}"), text_style)));
        }
        if finals_lines.is_empty() {
            finals_lines.push(Line::from(Span::styled(
                "（暂无 final）",
                Style::default().fg(Color::DarkGray),
            )));
        }
        let finals_title = if is_active {
            " finals (active) "
        } else {
            " finals "
        };
        let finals_para = Paragraph::new(finals_lines)
            .wrap(Wrap { trim: false })
            .scroll((slot.finals_scroll(), 0))
            .block(Block::default().borders(Borders::ALL).title(finals_title));
        frame.render_widget(finals_para, col_layout[1]);
    }

    let log_lines: Vec<Line> = app
        .log
        .iter()
        .rev()
        .take(5)
        .map(|s| Line::from(s.clone()))
        .collect();
    let log_para = Paragraph::new(log_lines).wrap(Wrap { trim: false }).block(
        Block::default()
            .borders(Borders::ALL)
            .title("日志（最近 endpoint）"),
    );
    frame.render_widget(log_para, chunks[2]);
}

fn render_media_panel(media: &MediaState, frame: &mut ratatui::Frame, area: Rect) {
    let active = media.active_index();
    let title = if media.playing {
        " subtitles playing "
    } else {
        " subtitles paused "
    };
    let mut lines = vec![
        Line::from(vec![
            Span::styled("time: ", Style::default().fg(Color::Cyan)),
            Span::styled(
                format!(
                    "{} / {}",
                    format_media_time(media.position_ms),
                    format_media_time(media.duration_ms)
                ),
                Style::default().fg(Color::Yellow),
            ),
        ]),
        Line::from(Span::styled(
            media_progress_bar(media.position_ms, media.duration_ms, 28),
            Style::default().fg(Color::Green),
        )),
        Line::from(vec![
            Span::styled("seek: ", Style::default().fg(Color::Cyan)),
            Span::raw("h/l 10s · H/L 60s · space pause · e report"),
        ]),
        Line::from(""),
    ];

    if media.cues.is_empty() {
        lines.push(Line::from(Span::styled(
            "未解析到字幕",
            Style::default().fg(Color::Red),
        )));
    } else {
        let start = active.saturating_sub(8);
        let end = (active + 10).min(media.cues.len());
        for cue in &media.cues[start..end] {
            let is_active = media.position_ms >= cue.start_ms && media.position_ms <= cue.end_ms;
            let style = if is_active {
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            let marker = if is_active { ">" } else { " " };
            lines.push(Line::from(Span::styled(
                format!(
                    "{marker} {} {}",
                    format_media_time(cue.start_ms),
                    cue.text.replace('\n', " ")
                ),
                style,
            )));
        }
    }

    if let Some(path) = &media.last_report {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            format!("report: {}", path.display()),
            Style::default().fg(Color::Green),
        )));
    }

    let panel = Paragraph::new(lines)
        .wrap(Wrap { trim: false })
        .block(Block::default().borders(Borders::ALL).title(title));
    frame.render_widget(panel, area);
}
