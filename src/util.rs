//! 跨模块共用的小工具：stdout/stderr 屏蔽、RMS、耗时格式化、日志追加。

use std::os::unix::io::RawFd;
use std::time::Instant;

/// 在闭包执行期间临时将 stdout（fd 1）和 stderr（fd 2）重定向到 /dev/null，
/// 屏蔽 C 库的调试输出（sherpa-onnx / onnxruntime 加载时会通过 stderr dump
/// 大量 encoder/decoder 配置与 CoreML 警告，这些原始字节会与 TUI 画面混在一起）。
pub(crate) fn with_stdout_suppressed<F: FnOnce() -> R, R>(f: F) -> R {
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

/// 持久重定向 stderr 到 /dev/null（不影响 stdout，TUI 渲染需要 stdout）。
/// 返回一个 guard，drop 时自动恢复 stderr。
/// 用于主循环期间屏蔽 Sherpa/ONNX Runtime 后台线程的 C++ 日志。
pub(crate) struct StderrSuppressGuard {
    saved: RawFd,
}

impl StderrSuppressGuard {
    pub(crate) fn new() -> Self {
        // FFI 声明（模块级 extern block 在 impl 块内可见）
        extern "C" {
            fn dup(fd: RawFd) -> RawFd;
            fn dup2(oldfd: RawFd, newfd: RawFd) -> RawFd;
            fn close(fd: RawFd) -> RawFd;
            fn open(path: *const u8, oflag: RawFd, ...) -> RawFd;
        }
        const O_WRONLY: RawFd = 1;

        let saved = unsafe { dup(2) };
        if saved < 0 {
            // fd 耗尽等极端情况：放弃抑制，guard 为 no-op
            return Self { saved: -1 };
        }
        unsafe {
            let dev_null = open(b"/dev/null\0".as_ptr(), O_WRONLY);
            if dev_null >= 0 {
                dup2(dev_null, 2);
                close(dev_null);
            }
        }
        Self { saved }
    }
}

impl Drop for StderrSuppressGuard {
    fn drop(&mut self) {
        if self.saved < 0 {
            return;
        }
        extern "C" {
            fn dup2(oldfd: RawFd, newfd: RawFd) -> RawFd;
            fn close(fd: RawFd) -> RawFd;
        }
        unsafe {
            dup2(self.saved, 2);
            close(self.saved);
        }
    }
}

pub(crate) fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum = samples.iter().map(|s| s * s).sum::<f32>();
    (sum / samples.len() as f32).sqrt()
}

pub(crate) fn humantime_elapsed(start: Instant) -> String {
    let s = start.elapsed().as_secs();
    format!("{}分{}秒", s / 60, s % 60)
}

pub(crate) fn push_log(log: &mut Vec<String>, message: impl Into<String>) {
    log.push(message.into());
    if log.len() > 200 {
        log.remove(0);
    }
}
