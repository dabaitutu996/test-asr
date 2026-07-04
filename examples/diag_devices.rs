//! 诊断工具：dump 三个来源的设备列表，并对每个输出设备实采 2 秒看有没有数据。
//!
//! 运行：cargo run --example diag_devices

use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arcvoice_core::audio::system_audio;
use cpal::traits::{DeviceTrait, HostTrait};

fn main() {
    println!("========== cpal INPUT devices ==========");
    let host = cpal::default_host();
    let cpal_inputs: Vec<String> = host
        .input_devices()
        .map(|d| d.filter_map(|x| x.name().ok()).collect())
        .unwrap_or_default();
    for n in &cpal_inputs {
        println!("  INPUT  name={n:?}");
    }

    println!("\n========== cpal OUTPUT devices ==========");
    let cpal_outputs: Vec<String> = host
        .output_devices()
        .map(|d| d.filter_map(|x| x.name().ok()).collect())
        .unwrap_or_default();
    for n in &cpal_outputs {
        println!("  OUTPUT name={n:?}");
    }

    println!("\n========== system_audio::list_output_devices (raw) ==========");
    let raw = match system_audio::list_output_devices() {
        Ok(v) => v,
        Err(e) => {
            println!("  ERROR: {e}");
            Vec::new()
        }
    };
    for d in &raw {
        println!("  name={:?} uid={:?}", d.name, d.uid);
    }

    println!("\n========== 白名单匹配情况 ==========");
    for d in &raw {
        let hit = cpal_outputs.iter().any(|w| w == &d.name);
        println!("  {} name={:?}", if hit { "KEEP" } else { "DROP" }, d.name);
    }

    println!("\n========== 对照采集：先测默认设备（None），再逐个测指定 UID ==========");
    println!("⚠️  Process Tap 只采集「别的 App 播放的声音」，本工具自己不发声。");
    println!("    请现在打开音乐/视频/浏览器并播放（系统默认输出设备即可）！");
    println!("    按回车开始采集...");
    let mut line = String::new();
    let _ = std::io::stdin().read_line(&mut line);

    let mut any_frames = false;

    println!("\n--- [对照] 默认输出设备（传 None，engine 原版用法） ---");
    any_frames |= test_capture_default() > 0;

    for d in &raw {
        println!("\n--- 指定 UID: {} ({}) ---", d.uid, d.name);
        any_frames |= test_capture(&d.name, &d.uid) > 0;
    }

    if !any_frames {
        println!("\n⚠️ 所有对照采集都保持 0 帧。");
        println!("   这通常不是 `Some(uid)` / `None` 或设备枚举差异，而是 macOS TCC 在宿主 App 层拒绝了系统音频录制。");
        println!("   若你是从终端 `cargo run` 启动，常见原因是终端宿主（例如 Warp）没有 `NSAudioCaptureUsageDescription`，");
        println!(
            "   TCC 会拒绝 `kTCCServiceAudioCapture`，但 Rust 侧仍可能拿到 `start() -> Ok(...)`。"
        );
        println!("   可用下面的命令核对最近日志：");
        println!("   /usr/bin/log show --last 10m --predicate 'subsystem == \"com.apple.TCC\"' | rg 'kTCCServiceAudioCapture|NSAudioCaptureUsageDescription'");
        println!("   如果看到 `Refusing authorization request ... without NSAudioCaptureUsageDescription key`，");
        println!(
            "   请改用带该 Info.plist 键的 .app 宿主运行，或直接用已有的 Tauri App bundle 调试。"
        );
    }

    println!("\n========== 完成 ==========");
}

fn test_capture_default() -> usize {
    let stop = Arc::new(AtomicBool::new(false));
    let rebuild = Arc::new(AtomicBool::new(false));
    let mut rx = match system_audio::start(stop.clone(), rebuild, None) {
        Ok((_tap, rx)) => rx,
        Err(e) => {
            println!("  启动失败: {e}");
            return 0;
        }
    };
    println!("  开始采集 6 秒...");
    capture_loop("默认设备", &mut rx, &stop)
}

fn capture_loop(
    label: &str,
    rx: &mut tokio::sync::mpsc::Receiver<Vec<f32>>,
    stop: &Arc<AtomicBool>,
) -> usize {
    let mut frames = 0usize;
    let mut max_rms = 0.0f32;
    for sec in 1..=6 {
        let deadline = Instant::now() + Duration::from_secs(1);
        while Instant::now() < deadline {
            while let Ok(buf) = rx.try_recv() {
                frames += 1;
                let r = rms(&buf);
                if r > max_rms {
                    max_rms = r;
                }
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        println!(
            "    [{label}] {sec}s: frames={frames} max_rms={max_rms:.4}{}",
            if max_rms > 0.001 {
                "  <-- 有声音!"
            } else {
                ""
            }
        );
    }
    stop.store(true, std::sync::atomic::Ordering::Release);
    frames
}

fn test_capture(name: &str, uid: &str) -> usize {
    let stop = Arc::new(AtomicBool::new(false));
    let rebuild = Arc::new(AtomicBool::new(false));
    let mut rx = match system_audio::start(stop.clone(), rebuild, Some(uid)) {
        Ok((_tap, rx)) => rx,
        Err(e) => {
            println!("  [{name:?}] 启动失败: {e}");
            return 0;
        }
    };
    capture_loop(name, &mut rx, &stop)
}

fn rms(s: &[f32]) -> f32 {
    if s.is_empty() {
        return 0.0;
    }
    let sum = s.iter().map(|x| x * x).sum::<f32>();
    (sum / s.len() as f32).sqrt()
}
