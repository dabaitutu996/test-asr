//! 音频采集（Linux：PipeWire / PulseAudio）。
//!
//! - `list_input_devices()` 用 `pactl -f json list sources` 枚举麦克风（排除 `.monitor`）。
//! - `list_output_devices()` 用 `pactl -f json list sinks` 枚举输出设备，
//!   每个输出设备对应一个 `<sink>.monitor` 采集源。
//! - `start_source()` / `start()` 用 `parec` 子进程抓流，统一转为 16kHz mono f32，
//!   按 200ms 一帧投递到 `tokio::sync::mpsc::Receiver`。
//!
//! 输入与输出共用同一条采集实现；`src/capture.rs` 在此之上做统一抽象。

mod linux;

pub use linux::PulseCapture;

pub const TARGET_SAMPLE_RATE: u32 = 16_000;
/// 每帧样本数：200ms @16kHz，与示例文件回放对齐。
pub(crate) const FRAME_SAMPLES: usize = 3_200;

use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use tokio::sync::mpsc;

/// 音频设备。
/// Linux 下 `uid` 是 PulseAudio 的 source 名（麦克风本身或 `<sink>.monitor`），
/// 可直接作为 `parec --device` 的参数。
#[derive(Debug, Clone)]
pub struct AudioDevice {
    pub id: u32,
    pub name: String,
    pub uid: String,
}

/// 列出可采集的输入端（PulseAudio sources，已排除 monitor）。
pub fn list_input_devices() -> anyhow::Result<Vec<AudioDevice>> {
    linux::list_input_devices()
}

/// 列出当前系统所有音频输出设备（每个都带可采集的 monitor source）。
pub fn list_output_devices() -> anyhow::Result<Vec<AudioDevice>> {
    linux::list_output_devices()
}

/// 从指定 source 采集（麦克风 source 与 monitor source 通用）。
pub fn start_source(
    stop_flag: Arc<AtomicBool>,
    needs_rebuild: Arc<AtomicBool>,
    source_uid: &str,
) -> anyhow::Result<(PulseCapture, mpsc::Receiver<Vec<f32>>)> {
    PulseCapture::start(stop_flag, needs_rebuild, Some(source_uid))
}

/// 采集系统音频，返回采集器句柄与 16kHz mono f32 PCM receiver。
///
/// `device_uid`: 指定要采集的输出设备 UID（从 `list_output_devices` 获取）。
///   传 `None` 则采集系统默认输出设备。
pub fn start(
    stop_flag: Arc<AtomicBool>,
    needs_rebuild: Arc<AtomicBool>,
    device_uid: Option<&str>,
) -> anyhow::Result<(PulseCapture, mpsc::Receiver<Vec<f32>>)> {
    PulseCapture::start(stop_flag, needs_rebuild, device_uid)
}
