/// 检测 tap 数据路径是否"卡死在全 0"。Sequoia/26.x 已知：输出设备状态变化后
/// IOProc 仍按节奏触发但只交付全 0 PCM，无 HAL 信号可辨。连续 `dead_ms`
/// 毫秒只见全 0（且 IOProc 在持续触发）即判定需要重建整条 tap 链。
pub struct SilenceWatchdog {
    dead_ms_threshold: u64,
    accumulated_silent_ms: u64,
    has_seen_signal: bool,
}

impl SilenceWatchdog {
    pub fn new(dead_ms_threshold: u64) -> Self {
        Self {
            dead_ms_threshold,
            accumulated_silent_ms: 0,
            has_seen_signal: false,
        }
    }

    /// 每个 IOProc 回调后调用。`all_zero` = 本帧是否全 0；`frame_ms` = 本帧时长。
    /// 返回 true 表示已达阈值、需要重建。
    pub fn observe(&mut self, all_zero: bool, frame_ms: u64) -> bool {
        if all_zero {
            self.accumulated_silent_ms += frame_ms;
        } else {
            self.has_seen_signal = true;
            self.accumulated_silent_ms = 0;
        }
        self.has_seen_signal && self.accumulated_silent_ms >= self.dead_ms_threshold
    }
}

/// 判断一帧 PCM 是否严格全 0（每个样本恰为 0.0）。
pub fn is_all_zero(samples: &[f32]) -> bool {
    samples.iter().all(|&sample| sample == 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn watchdog_triggers_after_threshold_of_continuous_silence() {
        let mut watchdog = SilenceWatchdog::new(2_000);
        assert!(!watchdog.observe(false, 100));
        for _ in 0..19 {
            assert!(!watchdog.observe(true, 100));
        }
        assert!(watchdog.observe(true, 100));
    }

    #[test]
    fn watchdog_ignores_initial_zero_frames_before_any_signal() {
        let mut watchdog = SilenceWatchdog::new(2_000);
        for _ in 0..30 {
            assert!(!watchdog.observe(true, 100));
        }
    }

    #[test]
    fn watchdog_resets_on_any_signal() {
        let mut watchdog = SilenceWatchdog::new(2_000);
        for _ in 0..15 {
            watchdog.observe(true, 100);
        }
        assert!(!watchdog.observe(false, 100));
        for _ in 0..19 {
            assert!(!watchdog.observe(true, 100));
        }
    }
}
