//! VAD 状态机：离线槽共享的语音分段，支持 Silero 与 RMS 兜底两种后端。

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use sherpa_onnx::{SileroVadModelConfig, VadModelConfig, VoiceActivityDetector};

use crate::config::{
    env_f32, silero_vad_model_path, SILERO_VAD_BUFFER_SEC, SILERO_VAD_MAX_SPEECH_SEC,
    SILERO_VAD_MIN_SILENCE_SEC, SILERO_VAD_MIN_SPEECH_SEC, SILERO_VAD_THRESHOLD,
    SILERO_VAD_WINDOW_SIZE, VAD_MAX_BUFFER_SEC, VAD_RMS_THRESHOLD, VAD_SAMPLE_RATE,
    VAD_SILENCE_SEC,
};
use crate::util::{push_log, rms, with_stdout_suppressed};

pub(crate) struct VadSegment {
    pub(crate) start_sample: usize,
    pub(crate) samples: Vec<f32>,
    pub(crate) reason: &'static str,
}

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum VadBackend {
    Disabled,
    Silero,
    Rms,
}

impl VadBackend {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Silero => "silero",
            Self::Rms => "rms",
        }
    }
}

pub(crate) struct VadState {
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
    pub(crate) fn new(has_offline_slots: bool) -> Result<Self> {
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
    pub(crate) fn push(&mut self, pcm: &[f32], log: &mut Vec<String>) -> Vec<VadSegment> {
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

    pub(crate) fn backend(&self) -> VadBackend {
        match &self.inner {
            VadImpl::Disabled => VadBackend::Disabled,
            VadImpl::Silero(_) => VadBackend::Silero,
            VadImpl::Rms(_) => VadBackend::Rms,
        }
    }

    pub(crate) fn status_label(&self) -> String {
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

    pub(crate) fn append_report(&self, out: &mut String) {
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

    pub(crate) fn reset(&mut self) {
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
