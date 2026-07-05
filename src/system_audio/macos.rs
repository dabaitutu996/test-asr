use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use tokio::sync::mpsc;

/// macOS 系统音频采集器。持有 Core Audio Process Tap 和 aggregate device 的全部资源。
/// Drop 时自动清理。
pub struct SystemAudioTap {
    inner: Option<super::macos_coreaudio::TapObjects>,
    needs_rebuild: Arc<AtomicBool>,
}

impl SystemAudioTap {
    pub fn start_with_stop(
        stop_flag: Arc<AtomicBool>,
    ) -> anyhow::Result<(Self, mpsc::Receiver<Vec<f32>>)> {
        Self::start_with_stop_and_rebuild_flag(stop_flag, Arc::new(AtomicBool::new(false)), None)
    }

    /// 启动系统音频采集。
    /// `device_uid`: 指定输出设备 UID，`None` 用默认输出。
    pub fn start_with_stop_and_rebuild_flag(
        stop_flag: Arc<AtomicBool>,
        needs_rebuild: Arc<AtomicBool>,
        device_uid: Option<&str>,
    ) -> anyhow::Result<(Self, mpsc::Receiver<Vec<f32>>)> {
        if !super::macos_coreaudio::is_process_tap_supported() {
            anyhow::bail!("系统音频 tap 需要 macOS 14.4 或更高版本");
        }
        let (objects, rx) = unsafe {
            super::macos_coreaudio::create_tap_objects(stop_flag, needs_rebuild.clone(), device_uid)
        }?;
        Ok((
            Self {
                inner: Some(objects),
                needs_rebuild,
            },
            rx,
        ))
    }

    pub fn needs_rebuild_flag(&self) -> Arc<AtomicBool> {
        self.needs_rebuild.clone()
    }
}

impl Drop for SystemAudioTap {
    fn drop(&mut self) {
        if let Some(objects) = self.inner.take() {
            unsafe { super::macos_coreaudio::destroy_tap_objects(objects) };
        }
    }
}

unsafe impl Send for SystemAudioTap {}
