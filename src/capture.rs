//! 设备选择与采集抽象：把 system_audio（Process Tap）、mic（cpal）
//! 和本地示例音频文件收拢成统一的 `Capture` 接口。

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait};
use tokio::sync::mpsc;

use arcvoice_core::audio::mic::MicInput;
use asr_compare_tui::system_audio::{self, OutputDevice};

use crate::config::{DEFAULT_MEDIA_MP3, VAD_SAMPLE_RATE};

const FILE_CHUNK_SAMPLES: usize = 3_200;
const FILE_FLUSH_CHUNKS: usize = 15;
/// 防止大文件 OOM：最多允许解码 ~52 分钟的 16kHz 单声道音频。
const MAX_SAMPLE_FILE_SAMPLES: usize = 50_000_000;

/// 用户在设备选择屏挑中的采集源。
/// Input 存 cpal 用的设备名，Output 存 CoreAudio 的设备 UID。
pub(crate) enum DevicePick {
    Input(String),
    Output(String),
    SampleFile(PathBuf),
}

impl DevicePick {
    /// 给 UI 显示用的标签，如「输入端 / MacBook 麦克风」。
    pub(crate) fn label(&self, _inputs: &[String], outputs: &[OutputDevice]) -> String {
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
            DevicePick::SampleFile(path) => {
                let name = path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or(DEFAULT_MEDIA_MP3);
                format!("示例音频文件 / {name}")
            }
        }
    }
}

/// 把 system_audio（Process Tap）、mic（cpal）和文件音频收拢成统一接口。
/// 预览屏和主屏都只跟 Capture 打交道，不再 if 分支。
pub(crate) struct Capture {
    pub(crate) rx: mpsc::Receiver<Vec<f32>>,
    stop_flag: Arc<AtomicBool>,
    /// 持有底层 stream/tap 保活；drop 即停止采集。
    _guard: CaptureGuard,
}

enum CaptureGuard {
    System {
        _tap: system_audio::SystemAudioTap,
    },
    Mic {
        _mic: MicInput,
    },
    File {
        control: Arc<Mutex<FileCaptureState>>,
        total_samples: usize,
        _worker: thread::JoinHandle<()>,
    },
}

struct FileCaptureState {
    position_samples: usize,
    playing: bool,
    flush_remaining: usize,
}

impl Capture {
    pub(crate) fn start(pick: &DevicePick) -> Result<Capture> {
        let stop_flag = Arc::new(AtomicBool::new(false));
        match pick {
            DevicePick::Output(uid) => {
                let rebuild = Arc::new(AtomicBool::new(false));
                let (tap, rx) =
                    system_audio::start(stop_flag.clone(), rebuild, Some(uid.as_str()))?;
                Ok(Capture {
                    rx,
                    stop_flag,
                    _guard: CaptureGuard::System { _tap: tap },
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
                    _guard: CaptureGuard::Mic { _mic: mic },
                })
            }
            DevicePick::SampleFile(path) => {
                let pcm = decode_media_to_pcm16k_mono(path)?;
                let total_samples = pcm.len();
                let (rx, control, worker) = stream_file_pcm(pcm, stop_flag.clone())?;
                Ok(Capture {
                    rx,
                    stop_flag,
                    _guard: CaptureGuard::File {
                        control,
                        total_samples,
                        _worker: worker,
                    },
                })
            }
        }
    }

    pub(crate) fn stop(&self) {
        self.stop_flag.store(true, Ordering::Release);
    }

    pub(crate) fn is_sample_file(&self) -> bool {
        matches!(self._guard, CaptureGuard::File { .. })
    }

    /// 查询文件采集是否正在播放（仅 `File` 模式有效；其他模式恒返回 false）。
    pub(crate) fn is_sample_file_playing(&self) -> bool {
        if let CaptureGuard::File { control, .. } = &self._guard {
            control.lock().map(|s| s.playing).unwrap_or(false)
        } else {
            false
        }
    }

    pub(crate) fn toggle_sample_file_playback(&mut self) -> Option<bool> {
        let CaptureGuard::File {
            control,
            total_samples,
            ..
        } = &self._guard
        else {
            return None;
        };

        let mut state = control.lock().ok()?;
        state.flush_remaining = 0;
        if state.playing {
            state.playing = false;
            // 排空 channel 中已缓冲的 PCM，避免暂停后仍有残余音频送入 ASR。
            while self.rx.try_recv().is_ok() {}
            return Some(false);
        }

        if state.position_samples >= *total_samples {
            state.position_samples = 0;
        }
        state.playing = true;
        Some(true)
    }

    pub(crate) fn seek_sample_file_by(&mut self, delta_ms: i64) -> Option<u64> {
        let CaptureGuard::File {
            control,
            total_samples,
            ..
        } = &self._guard
        else {
            return None;
        };

        let mut state = control.lock().ok()?;
        let current_ms = samples_to_ms(state.position_samples);
        let next_ms = if delta_ms.is_negative() {
            current_ms.saturating_sub(delta_ms.unsigned_abs())
        } else {
            current_ms.saturating_add(delta_ms as u64)
        };
        state.position_samples = ms_to_samples(next_ms, *total_samples);
        state.flush_remaining = 0;
        Some(samples_to_ms(state.position_samples))
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        self.stop();
    }
}

/// 枚举输入端（麦克风）。cpal 失败时返回空 Vec，由调用方处理。
pub(crate) fn list_inputs() -> Vec<String> {
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
pub(crate) fn list_outputs() -> Vec<OutputDevice> {
    let mut raw = system_audio::list_output_devices().unwrap_or_default();
    raw.dedup_by(|a, b| a.name == b.name && a.uid == b.uid);
    raw
}

/// 枚举可直接送入 ASR 的示例音频文件。
pub(crate) fn list_sample_files() -> Vec<PathBuf> {
    let path = PathBuf::from(DEFAULT_MEDIA_MP3);
    if path.exists() {
        vec![path]
    } else {
        Vec::new()
    }
}

fn decode_media_to_pcm16k_mono(path: &Path) -> Result<Vec<f32>> {
    let output = Command::new("ffmpeg")
        .arg("-nostdin")
        .arg("-loglevel")
        .arg("error")
        .arg("-i")
        .arg(path)
        .arg("-f")
        .arg("f32le")
        .arg("-acodec")
        .arg("pcm_f32le")
        .arg("-ac")
        .arg("1")
        .arg("-ar")
        .arg(VAD_SAMPLE_RATE.to_string())
        .arg("-")
        .output()
        .with_context(|| "启动 ffmpeg 失败，请确认 ffmpeg 在 PATH 中")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("ffmpeg 解码失败: {}", stderr.trim());
    }

    let mut pcm = Vec::with_capacity(output.stdout.len() / std::mem::size_of::<f32>());
    for bytes in output.stdout.chunks_exact(4) {
        pcm.push(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]));
    }
    if pcm.is_empty() {
        anyhow::bail!("示例音频解码后没有可用 PCM: {}", path.display());
    }
    if pcm.len() > MAX_SAMPLE_FILE_SAMPLES {
        anyhow::bail!(
            "示例音频过大（{} 采样点 ≈ {} 分钟），超过上限 {} 采样点。请使用更短的音频文件。",
            pcm.len(),
            pcm.len() as u64 * 1000 / VAD_SAMPLE_RATE as u64 / 60_000,
            MAX_SAMPLE_FILE_SAMPLES,
        );
    }
    Ok(pcm)
}

fn stream_file_pcm(
    pcm: Vec<f32>,
    stop_flag: Arc<AtomicBool>,
) -> Result<(
    mpsc::Receiver<Vec<f32>>,
    Arc<Mutex<FileCaptureState>>,
    thread::JoinHandle<()>,
)> {
    let (tx, rx) = mpsc::channel(32);
    let control = Arc::new(Mutex::new(FileCaptureState {
        position_samples: 0,
        playing: true,
        flush_remaining: 0,
    }));
    let worker_control = control.clone();
    let worker = thread::Builder::new()
        .name("sample-file-capture".to_string())
        .spawn(move || {
            let chunk_duration =
                Duration::from_secs_f64(FILE_CHUNK_SAMPLES as f64 / VAD_SAMPLE_RATE as f64);
            let silence = vec![0.0_f32; FILE_CHUNK_SAMPLES];
            loop {
                if stop_flag.load(Ordering::Acquire) {
                    return;
                }

                let next_chunk = {
                    let Ok(mut state) = worker_control.lock() else {
                        return;
                    };

                    if state.flush_remaining > 0 {
                        state.flush_remaining -= 1;
                        if state.flush_remaining == 0 {
                            state.playing = false;
                        }
                        Some(silence.clone())
                    } else if !state.playing {
                        None
                    } else if state.position_samples >= pcm.len() {
                        None
                    } else {
                        let start = state.position_samples;
                        let end = (start + FILE_CHUNK_SAMPLES).min(pcm.len());
                        state.position_samples = end;
                        if end >= pcm.len() {
                            state.flush_remaining = FILE_FLUSH_CHUNKS;
                        }
                        Some(pcm[start..end].to_vec())
                    }
                };

                let Some(chunk) = next_chunk else {
                    thread::sleep(Duration::from_millis(20));
                    continue;
                };

                let started = Instant::now();
                if tx.blocking_send(chunk).is_err() {
                    return;
                }
                let elapsed = started.elapsed();
                if elapsed < chunk_duration {
                    thread::sleep(chunk_duration - elapsed);
                }
            }
        })
        .context("启动示例音频采集线程失败")?;

    Ok((rx, control, worker))
}

fn samples_to_ms(samples: usize) -> u64 {
    ((samples as u128) * 1000 / VAD_SAMPLE_RATE as u128) as u64
}

fn ms_to_samples(ms: u64, total_samples: usize) -> usize {
    (((ms as u128) * VAD_SAMPLE_RATE as u128) / 1000).min(total_samples as u128) as usize
}
