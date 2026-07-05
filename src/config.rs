//! 常量、环境变量解析与模型路径解析。

use std::path::PathBuf;
use std::time::Duration;

/// 顶部 partial 之外，每个 slot 最多保留多少条 final 历史。
pub(crate) const FINALS_RETAIN: usize = 6;
/// 每个在线槽最多记录多少条 partial 变化，用于诊断 endpoint/reset 是否丢词。
pub(crate) const PARTIAL_HISTORY_RETAIN: usize = 2000;
/// TUI 轮询周期：终端事件与重绘节奏。
pub(crate) const POLL_INTERVAL: Duration = Duration::from_millis(50);
pub(crate) const DEFAULT_MEDIA_MP3: &str = "ARC_Raiders_is_getting_DARKER....mp3";
pub(crate) const DEFAULT_MEDIA_SRT: &str = "ARC_Raiders_is_getting_DARKER....en.srt";

// ─── 离线模型 VAD 参数 ─────────────────────────────────────────────────
pub(crate) const VAD_SAMPLE_RATE: usize = 16_000;
pub(crate) const SILERO_VAD_THRESHOLD: f32 = 0.5;
pub(crate) const SILERO_VAD_MIN_SILENCE_SEC: f32 = 0.5;
pub(crate) const SILERO_VAD_MIN_SPEECH_SEC: f32 = 0.25;
pub(crate) const SILERO_VAD_MAX_SPEECH_SEC: f32 = 8.0;
pub(crate) const SILERO_VAD_WINDOW_SIZE: i32 = 512;
pub(crate) const SILERO_VAD_BUFFER_SEC: f32 = 10.0;

/// RMS 低于此值视为静音；仅在 `VAD_BACKEND=rms` 时使用。
pub(crate) const VAD_RMS_THRESHOLD: f32 = 0.012;
/// RMS 兜底路径里，静音持续多少秒后触发离线解码。
pub(crate) const VAD_SILENCE_SEC: f32 = 0.4;
/// RMS 兜底路径里，缓冲最长多少秒强制触发解码。
pub(crate) const VAD_MAX_BUFFER_SEC: f32 = 8.0;

// ─── 环境变量 ──────────────────────────────────────────────────────────

pub(crate) fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.trim().parse::<f32>().ok())
        .unwrap_or(default)
}

pub(crate) fn endpoint_rules_from_env() -> (f32, f32, f32) {
    (
        env_f32("ASR_ENDPOINT_RULE1", 2.4),
        env_f32("ASR_ENDPOINT_RULE2", 1.2),
        env_f32("ASR_ENDPOINT_RULE3", 20.0),
    )
}

pub(crate) fn hotwords_from_env() -> Vec<String> {
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

// ─── 模型路径 ──────────────────────────────────────────────────────────

/// engine 模型基础目录：从本项目 Cargo.toml 向上两级到 Desktop，
/// 再进入 game-video/engine/models/streaming/。
pub(crate) fn engine_models_dir() -> PathBuf {
    // 优先用环境变量覆盖，方便灵活部署。
    if let Ok(dir) = std::env::var("ENGINE_MODELS_DIR") {
        return PathBuf::from(dir);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../game-video/engine/models/streaming")
}

/// 离线模型（Canary 等）目录：本项目下的 models/streaming/。
/// 与流式模型分开存放——流式模型沿用 sibling game-video 仓库，
/// 离线模型体积大、独立性强，放本仓库更清晰。
pub(crate) fn offline_models_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("OFFLINE_MODELS_DIR") {
        return PathBuf::from(dir);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/streaming")
}

pub(crate) fn silero_vad_model_path() -> PathBuf {
    if let Ok(path) = std::env::var("SILERO_VAD_MODEL") {
        return PathBuf::from(path);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/vad/silero_vad/silero_vad.int8.onnx")
}
