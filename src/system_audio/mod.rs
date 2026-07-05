//! 系统音频采集（macOS Core Audio Process Tap, macOS >= 14.4）。
//!
//! 通过 `start()` 启动采集，返回 16kHz mono f32 PCM 的
//! `tokio::sync::mpsc::Receiver`。可选指定采集哪个输出设备。
//!
//! `list_output_devices()` 列出当前所有音频输出设备，供用户选择。

mod macos;
mod macos_coreaudio;
mod watchdog;

pub use macos::SystemAudioTap;

pub const TARGET_SAMPLE_RATE: u32 = 16_000;
pub(crate) const TAP_SAMPLE_CHANNEL_CAPACITY: usize = 256;
pub(crate) const TAP_REBUILD_SILENCE_MS: u64 = 2_000;
pub(crate) const DROPPED_FRAME_WARN_EVERY: u64 = 250;

use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use tokio::sync::mpsc;

/// 音频输出设备信息。
#[derive(Debug, Clone)]
pub struct OutputDevice {
    pub id: u32,
    pub name: String,
    pub uid: String,
}

/// 列出当前系统所有音频输出设备。
pub fn list_output_devices() -> anyhow::Result<Vec<OutputDevice>> {
    macos_coreaudio::enumerate_output_devices()
}

/// 启动系统音频采集，返回采集器句柄与 16kHz mono f32 PCM receiver。
///
/// `device_uid`: 指定要采集的输出设备 UID（从 `list_output_devices` 获取）。
///   传 `None` 则使用系统默认输出设备。
pub fn start(
    stop_flag: Arc<AtomicBool>,
    needs_rebuild: Arc<AtomicBool>,
    device_uid: Option<&str>,
) -> anyhow::Result<(SystemAudioTap, mpsc::Receiver<Vec<f32>>)> {
    SystemAudioTap::start_with_stop_and_rebuild_flag(stop_flag, needs_rebuild, device_uid)
}
