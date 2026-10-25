//! Linux 系统音频采集：`pactl` 枚举 monitor source，`parec` 抓 16kHz mono f32 流。
//!
//! 走子进程而不是直接接 ALSA/PipeWire API：monitor source 是 PulseAudio 层的
//! 概念，`parec` 同时兼容原生 PulseAudio 与 PipeWire 的 pulse 兼容层，并自带
//! 重采样（`--rate`/`--channels`），无需额外依赖。

use std::io::{BufReader, Read};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

use anyhow::{bail, Context, Result};
use serde_json::Value;
use tokio::sync::mpsc;

use super::{AudioDevice, FRAME_SAMPLES, TARGET_SAMPLE_RATE};

/// 采集器。持有 `parec` 子进程与读取线程，Drop 时全部回收。
/// 输入 source 与输出 monitor source 共用同一实现。
pub struct PulseCapture {
    child: Option<Child>,
    reader: Option<thread::JoinHandle<()>>,
    needs_rebuild: Arc<AtomicBool>,
}

impl PulseCapture {
    /// 启动采集。`source_uid` 为 `list_input_devices()` / `list_output_devices()`
    /// 返回的 uid；`None` 时自动解析系统默认输出的 monitor source。
    pub fn start(
        stop_flag: Arc<AtomicBool>,
        needs_rebuild: Arc<AtomicBool>,
        source_uid: Option<&str>,
    ) -> Result<(Self, mpsc::Receiver<Vec<f32>>)> {
        let source = match source_uid {
            Some(uid) if !uid.trim().is_empty() => uid.to_string(),
            _ => default_monitor_source()?,
        };

        let mut child = Command::new("parec")
            .arg(format!("--device={source}"))
            .arg("--format=float32le")
            .arg(format!("--rate={TARGET_SAMPLE_RATE}"))
            .arg("--channels=1")
            .arg("--latency-msec=50")
            .arg("--raw")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| {
                format!(
                    "启动 parec 失败（设备 {source}）：请确认已安装 pulseaudio-utils，且 PipeWire/PulseAudio 正在运行"
                )
            })?;

        let stdout = child.stdout.take().context("parec 未提供标准输出")?;
        let (tx, rx) = mpsc::channel(64);
        let rebuild_flag = needs_rebuild.clone();

        let reader = thread::Builder::new()
            .name("system-audio".to_string())
            .spawn(move || {
                let mut stdout = BufReader::with_capacity(64 * 1024, stdout);
                let mut acc: Vec<f32> = Vec::with_capacity(FRAME_SAMPLES * 2);
                let mut bytes = [0u8; 4];
                loop {
                    if stop_flag.load(Ordering::Acquire) {
                        return;
                    }
                    match stdout.read_exact(&mut bytes) {
                        Ok(()) => {
                            acc.push(f32::from_le_bytes(bytes));
                            if acc.len() >= FRAME_SAMPLES {
                                let frame: Vec<f32> = acc.drain(..FRAME_SAMPLES).collect();
                                if tx.blocking_send(frame).is_err() {
                                    return;
                                }
                            }
                        }
                        Err(err) if err.kind() == std::io::ErrorKind::Interrupted => continue,
                        Err(_) => {
                            // parec 退出（设备被移除 / 服务重启）：置位重建标记，
                            // 由上层决定是否重连。
                            if !stop_flag.load(Ordering::Acquire) {
                                rebuild_flag.store(true, Ordering::Release);
                            }
                            return;
                        }
                    }
                }
            })
            .context("启动系统音频读取线程失败")?;

        Ok((
            Self {
                child: Some(child),
                reader: Some(reader),
                needs_rebuild,
            },
            rx,
        ))
    }

    pub fn needs_rebuild_flag(&self) -> Arc<AtomicBool> {
        self.needs_rebuild.clone()
    }
}

impl Drop for PulseCapture {
    fn drop(&mut self) {
        // 先杀子进程让读取线程的 read 返回，再 join，避免卡住退出流程。
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        if let Some(handle) = self.reader.take() {
            let _ = handle.join();
        }
    }
}

/// 枚举所有输出设备（每个 sink 都自带一个 monitor source）。
pub(super) fn list_output_devices() -> Result<Vec<AudioDevice>> {
    let sinks = read_pactl_json(&["list", "sinks"])?;
    Ok(parse_sinks(&sinks))
}

/// 枚举所有输入端（PulseAudio sources，排除 `.monitor`）。
/// 系统默认 source 排在最前，让设备屏光标默认落在用户实际使用的麦克风上。
pub(super) fn list_input_devices() -> Result<Vec<AudioDevice>> {
    let sources = read_pactl_json(&["list", "sources"])?;
    let mut devices = parse_sources(&sources);
    let default_source = read_pactl_json(&["info"]).ok().and_then(|info| {
        info.get("default_source_name")
            .and_then(Value::as_str)
            .map(str::to_string)
    });
    promote_default(&mut devices, default_source.as_deref());
    Ok(devices)
}

/// 把 `default_uid` 对应的设备移到首位（不存在时保持原顺序）。
fn promote_default(devices: &mut Vec<AudioDevice>, default_uid: Option<&str>) {
    let Some(uid) = default_uid else {
        return;
    };
    if let Some(position) = devices.iter().position(|device| device.uid == uid) {
        let device = devices.remove(position);
        devices.insert(0, device);
    }
}

/// 解析默认输出设备对应的 monitor source 名。
fn default_monitor_source() -> Result<String> {
    let info = read_pactl_json(&["info"])?;
    let sink = info
        .get("default_sink_name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty() && *name != "(null)")
        .context("读取默认输出设备失败：PulseAudio/PipeWire 可能未运行")?;
    Ok(format!("{sink}.monitor"))
}

fn read_pactl_json(args: &[&str]) -> Result<Value> {
    let output = Command::new("pactl")
        .arg("-f")
        .arg("json")
        .args(args)
        .output()
        .with_context(|| "启动 pactl 失败：枚举系统音频设备需要 pulseaudio-utils（pactl）")?;
    if !output.status.success() {
        bail!(
            "pactl {:?} 执行失败: {}",
            args,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    serde_json::from_slice(&output.stdout).context("解析 pactl 的 JSON 输出失败")
}

/// 从 `pactl -f json list sinks` 提取输出设备：uid 取 `monitor_source`。
/// 没有该字段的条目（极老的 pactl）会被跳过。
fn parse_sinks(value: &Value) -> Vec<AudioDevice> {
    let Some(items) = value.as_array() else {
        return Vec::new();
    };
    items
        .iter()
        .filter_map(|item| {
            let uid = item.get("monitor_source")?.as_str()?.to_string();
            Some(device_from_item(item, uid))
        })
        .collect()
}

/// 从 `pactl -f json list sources` 提取输入端：uid 取 source 名，
/// 并过滤掉 `<sink>.monitor`（系统音频不是麦克风）。
fn parse_sources(value: &Value) -> Vec<AudioDevice> {
    let Some(items) = value.as_array() else {
        return Vec::new();
    };
    items
        .iter()
        .filter_map(|item| {
            let uid = item.get("name")?.as_str()?.to_string();
            if uid.ends_with(".monitor") {
                return None;
            }
            Some(device_from_item(item, uid))
        })
        .collect()
}

fn device_from_item(item: &Value, uid: String) -> AudioDevice {
    let id = item.get("index").and_then(Value::as_u64).unwrap_or(0) as u32;
    AudioDevice {
        id,
        name: device_display_name(item),
        uid,
    }
}

/// 设备的可读名字。PipeWire 常把 `description` 填成 `(null)`，
/// 此时退回 `device.description` / `node.nick`，最后才用原始设备名。
fn device_display_name(item: &Value) -> String {
    let description = item
        .get("description")
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty() && *text != "(null)");

    let properties = item.get("properties");
    let device_description = properties
        .and_then(|props| props.get("device.description"))
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty() && *text != "(null)");
    let node_nick = properties
        .and_then(|props| props.get("node.nick"))
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty() && *text != "(null)");

    if let Some(name) = description {
        return name.to_string();
    }
    match (device_description, node_nick) {
        (Some(device), Some(nick)) if device != nick => format!("{device} ({nick})"),
        (Some(device), _) => device.to_string(),
        (None, Some(nick)) => nick.to_string(),
        (None, None) => item
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("未知输出设备")
            .to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SINKS_JSON: &str = r#"[
        {
            "index": 49,
            "name": "alsa_output.pci-0000_2f_00.4.iec958-stereo",
            "description": "(null)",
            "monitor_source": "alsa_output.pci-0000_2f_00.4.iec958-stereo.monitor",
            "properties": {
                "device.description": "Starship/Matisse HD Audio Controller",
                "node.nick": "ALCS1200A Digital"
            }
        },
        {
            "index": 51,
            "name": "alsa_output.pci-0000_2d_00.1.hdmi-stereo",
            "description": "Navi 31 HDMI / DisplayPort",
            "monitor_source": "alsa_output.pci-0000_2d_00.1.hdmi-stereo.monitor",
            "properties": {}
        },
        {
            "index": 60,
            "name": "no-monitor-device",
            "description": "没有 monitor 的条目"
        }
    ]"#;

    #[test]
    fn parses_monitor_sources_from_pactl_json() {
        let value: Value = serde_json::from_str(SINKS_JSON).expect("fixture 应为合法 JSON");
        let devices = parse_sinks(&value);

        assert_eq!(devices.len(), 2, "缺少 monitor_source 的条目应被跳过");
        assert_eq!(devices[0].id, 49);
        assert_eq!(
            devices[0].uid,
            "alsa_output.pci-0000_2f_00.4.iec958-stereo.monitor"
        );
        // description 为 (null) 时用 device.description + node.nick 组合
        assert_eq!(
            devices[0].name,
            "Starship/Matisse HD Audio Controller (ALCS1200A Digital)"
        );
        assert_eq!(devices[1].name, "Navi 31 HDMI / DisplayPort");
    }

    #[test]
    fn parses_input_sources_and_skips_monitors() {
        const SOURCES_JSON: &str = r#"[
            {
                "index": 50,
                "name": "alsa_input.pci-0000_2f_00.4.analog-stereo",
                "description": "(null)",
                "properties": {
                    "device.description": "Starship/Matisse HD Audio Controller",
                    "node.nick": "ALCS1200A Analog"
                }
            },
            {
                "index": 52,
                "name": "alsa_input.usb-DJI_Wireless_Mic_Rx-01.analog-stereo",
                "description": "(null)",
                "properties": {
                    "device.description": "Wireless Mic Rx",
                    "node.nick": "Wireless Mic Rx"
                }
            },
            {
                "index": 49,
                "name": "alsa_output.pci-0000_2f_00.4.iec958-stereo.monitor",
                "description": "(null)",
                "properties": {}
            }
        ]"#;
        let value: Value = serde_json::from_str(SOURCES_JSON).expect("fixture 应为合法 JSON");
        let devices = parse_sources(&value);

        assert_eq!(devices.len(), 2, "monitor source 不应出现在输入端");
        assert_eq!(devices[0].uid, "alsa_input.pci-0000_2f_00.4.analog-stereo");
        assert_eq!(
            devices[0].name,
            "Starship/Matisse HD Audio Controller (ALCS1200A Analog)"
        );
        // device.description 与 node.nick 相同则只保留一份
        assert_eq!(devices[1].name, "Wireless Mic Rx");
    }

    #[test]
    fn tolerates_non_array_payload() {
        let value: Value = serde_json::from_str(r#"{"error": "no server"}"#).unwrap();
        assert!(parse_sinks(&value).is_empty());
        assert!(parse_sources(&value).is_empty());
    }

    #[test]
    fn promotes_default_source_to_front() {
        let mut devices = vec![
            AudioDevice {
                id: 1,
                name: "analog".into(),
                uid: "alsa_input.analog".into(),
            },
            AudioDevice {
                id: 2,
                name: "usb mic".into(),
                uid: "alsa_input.usb".into(),
            },
        ];

        promote_default(&mut devices, Some("alsa_input.usb"));
        assert_eq!(devices[0].uid, "alsa_input.usb");

        // 默认设备不在列表里时保持原顺序
        promote_default(&mut devices, Some("alsa_input.missing"));
        assert_eq!(devices[0].uid, "alsa_input.usb");

        promote_default(&mut devices, None);
        assert_eq!(devices[0].uid, "alsa_input.usb");
    }
}
