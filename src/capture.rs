//! 设备选择与采集抽象：把 system_audio（Process Tap）和 mic（cpal）
//! 两条采集路径收拢成统一的 `Capture` 接口。

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait};
use tokio::sync::mpsc;

use arcvoice_core::audio::mic::MicInput;
use asr_compare_tui::system_audio::{self, OutputDevice};

/// 用户在设备选择屏挑中的采集源。
/// Input 存 cpal 用的设备名，Output 存 CoreAudio 的设备 UID。
pub(crate) enum DevicePick {
    Input(String),
    Output(String),
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
        }
    }
}

/// 把 system_audio（Process Tap）和 mic（cpal）两条采集路径收拢成统一接口。
/// 预览屏和主屏都只跟 Capture 打交道，不再 if 分支。
pub(crate) struct Capture {
    pub(crate) rx: mpsc::Receiver<Vec<f32>>,
    stop_flag: Arc<AtomicBool>,
    /// 持有底层 stream/tap 保活；drop 即停止采集。
    _guard: CaptureGuard,
}

enum CaptureGuard {
    System { _tap: system_audio::SystemAudioTap },
    Mic { _mic: MicInput },
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
        }
    }

    pub(crate) fn stop(&self) {
        self.stop_flag.store(true, Ordering::Release);
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
