//! 设备诊断示例（Linux）：列出麦克风（PulseAudio sources）与系统音频
//! （monitor source），可选对每个 source 做 2 秒实测采集，打印帧数与 RMS。
//!
//! 用法：
//! ```text
//! cargo run --example diag_devices               # 只列设备
//! cargo run --example diag_devices -- --capture  # 列表 + 逐个实测 2 秒
//! ```

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Result;
use asr_compare_tui::system_audio::{self, AudioDevice};

const CAPTURE_SECONDS: u64 = 2;

fn main() -> Result<()> {
    let capture = std::env::args().any(|arg| arg == "--capture");

    println!("========== 输入端（pactl list sources） ==========");
    let inputs = system_audio::list_input_devices().unwrap_or_else(|err| {
        println!("  枚举失败: {err}");
        Vec::new()
    });
    if inputs.is_empty() {
        println!("  （无输入设备：确认 PulseAudio/PipeWire 正在运行，且已安装 pulseaudio-utils）");
    }
    for device in &inputs {
        println!("  [{}] {} -> {}", device.id, device.name, device.uid);
    }

    println!("\n========== 输出端（pactl list sinks → monitor source） ==========");
    let outputs = system_audio::list_output_devices().unwrap_or_else(|err| {
        println!("  枚举失败: {err}");
        Vec::new()
    });
    if outputs.is_empty() {
        println!("  （无输出设备：确认 PulseAudio/PipeWire 正在运行，且已安装 pulseaudio-utils）");
    }
    for device in &outputs {
        println!("  [{}] {} -> {}", device.id, device.name, device.uid);
    }

    if !capture {
        println!("\n提示：加 --capture 可对每个设备实测 2 秒采集。");
        return Ok(());
    }

    println!("\n========== 实测采集（每设备 {CAPTURE_SECONDS}s） ==========");
    for device in inputs.iter().chain(outputs.iter()) {
        measure(device);
    }

    Ok(())
}

fn measure(device: &AudioDevice) {
    let stop = Arc::new(AtomicBool::new(false));
    let rebuild = Arc::new(AtomicBool::new(false));
    let started = Instant::now();

    let (tap, mut rx) =
        match system_audio::start_source(stop.clone(), rebuild.clone(), device.uid.as_str()) {
            Ok(pair) => pair,
            Err(err) => {
                println!("  [{}] 启动失败: {err:#}", device.name);
                return;
            }
        };

    let deadline = started + Duration::from_secs(CAPTURE_SECONDS);
    let mut frames = 0usize;
    let mut samples = 0usize;
    let mut sum_squares = 0f64;
    while Instant::now() < deadline {
        match rx.try_recv() {
            Ok(frame) => {
                frames += 1;
                samples += frame.len();
                sum_squares += frame
                    .iter()
                    .map(|sample| (*sample as f64) * (*sample as f64))
                    .sum::<f64>();
            }
            Err(_) => thread::sleep(Duration::from_millis(20)),
        }
    }
    stop.store(true, Ordering::Release);
    drop(tap);

    let rms = if samples == 0 {
        0.0
    } else {
        (sum_squares / samples as f64).sqrt()
    };
    let verdict = if frames == 0 {
        "无数据（设备空闲？）"
    } else if rms < 1e-4 {
        "有数据但接近静音"
    } else {
        "采集正常"
    };
    println!(
        "  [{}] 帧数={frames} 样本={samples} RMS={rms:.5} -> {verdict}",
        device.name
    );
    if rebuild.load(Ordering::Acquire) {
        println!("     (parec 中途退出，已置位重建标记)");
    }
}
