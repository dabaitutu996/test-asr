# TCC Audio Capture Check

这些命令都应当整行复制执行，不要在单引号 `'...'` 或 `-e` 参数中间手动换行。

## 1. 查最近 10 分钟的 TCC 音频捕获日志

最稳的版本，优先用这个：

```bash
/usr/bin/log show --last 10m --predicate 'subsystem == "com.apple.TCC"' | grep -E 'kTCCServiceAudioCapture|NSAudioCaptureUsageDescription'
```

带上下文：

```bash
/usr/bin/log show --last 10m --predicate 'subsystem == "com.apple.TCC"' | grep -n -C 3 -E 'kTCCServiceAudioCapture|NSAudioCaptureUsageDescription'
```

如果你就是想用 `rg`，必须整行执行：

```bash
/usr/bin/log show --last 10m --predicate 'subsystem == "com.apple.TCC"' | rg -n -C 3 -e 'kTCCServiceAudioCapture' -e 'NSAudioCaptureUsageDescription'
```

## 2. 只看拒绝原因

```bash
/usr/bin/log show --last 10m --predicate 'subsystem == "com.apple.TCC"' | grep -E 'Refusing authorization request|NSAudioCaptureUsageDescription|AUTHREQ_RESULT'
```

如果出现下面这句，就说明不是 Rust 代码问题，而是宿主 App 被 TCC 拒了：

```text
Refusing authorization request for service kTCCServiceAudioCapture ... without NSAudioCaptureUsageDescription key
```

## 3. 看 Warp 有没有系统音频录制用的 Info.plist 键

```bash
plutil -p '/Applications/Warp.app/Contents/Info.plist' | grep -E 'CFBundleIdentifier|NSAudioCaptureUsageDescription|NSMicrophoneUsageDescription'
```

如果只有 `NSMicrophoneUsageDescription`，没有 `NSAudioCaptureUsageDescription`，那 Warp 里跑 `cargo run` 时就会卡在系统音频录制权限这层。

## 4. 看 Terminal.app 有没有这个键

```bash
plutil -p '/System/Applications/Utilities/Terminal.app/Contents/Info.plist' | grep -E 'CFBundleIdentifier|NSAudioCaptureUsageDescription|NSMicrophoneUsageDescription'
```

## 5. 复现对照

当前仓库的诊断 example：

```bash
cargo run --example diag_devices
```

如果诊断 example 是 `frames=0`，同时第 1 步日志里又出现
`without NSAudioCaptureUsageDescription key`，就可以直接判定为宿主终端/TCC 问题。

## 6. 建议结论

- 从 Warp 里直接 `cargo run` 调试系统音频采集，不可靠。
- 优先改用带 `NSAudioCaptureUsageDescription` 的 `.app` 宿主运行。
- 你现有的 Tauri app 已经有这个键：`game-video/app-mac/src-tauri/Info.plist`。
