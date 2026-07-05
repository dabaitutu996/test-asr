# ASR Compare TUI

采集系统音频（BlackHole / 麦克风），实时对比多个 ASR 模型在同一音频流上的识别效果。
设计用途：在游戏实况 / FPS 解说等嘈杂场景下，横向对比不同模型的准确率、抗噪能力、
延迟和标点质量。

## 支持的模型

| 模型 | 类型 | 后端 | 说明 |
|---|---|---|---|
| Zipformer-zh | 真流式 transducer | sherpa-onnx OnlineRecognizer | 中文 |
| Zipformer-en | 真流式 transducer | sherpa-onnx OnlineRecognizer | 英文 |
| Nemotron-en | 真流式 transducer | sherpa-onnx OnlineRecognizer | 英文（NVIDIA Nemotron Speech Streaming 0.6B） |
| **Canary-180m-flash** | **离线 + VAD 触发** | sherpa-onnx OfflineRecognizer | 英文（NVIDIA Canary 180M，自带标点+大小写） |
| Silero-VAD | VAD 前置 | sherpa-onnx VAD | 轻量切句，供离线模型配合使用 |
| Parakeet-TDT-CTC-110M | 离线 + 外部 VAD | sherpa-onnx OfflineRecognizer | 英文（NVIDIA Parakeet 110M INT8） |

**流式 vs 离线的差异**：
- 流式模型逐帧增量喂音，自带 partial（实时显示正在识别的内容）和 endpoint（自动断句）。
- 离线模型（Canary）没有 partial/endpoint，靠内置 RMS 能量 VAD 检测静音句尾，
  0.4 秒静音触发一次性解码，结果直接作为 final 输出。

## 运行

```bash
# 1. 构建
cargo build

# 2. （可选）下载 Canary 离线模型（约 200MB）
./scripts/download-canary.sh

# 3. （可选）下载 Silero VAD（约 208KB）
./scripts/download-silero-vad.sh

# 4. （可选）下载 Parakeet 110M INT8（约 126MB）
./scripts/download-parakeet-tdt-ctc-110m.sh

# 5. 启动
cargo run
# 或 ./测试
```

启动后：
1. **选择引擎屏**：数字键 1-9 勾选要加载的模型。Canary 若未下载会显示为灰色不可选，
   并提示运行 `./scripts/download-canary.sh`。Enter 确认。
2. **设备选择屏**：选输入端（麦克风）或输出端（系统音频，如 BlackHole 2ch）。
3. **预览屏**：观察电平是否跳动，确认采集源正确。Enter 进入主屏。
4. **主屏**：多列并行显示各模型的 partial / finals。
   - `←/→` 切换激活列 · `↑/↓` 滚动 finals · `PgUp/PgDn` 快滚
   - `1-9` 切换某列启用/禁用 · `c` 清空 · `e` 导出 Markdown 报告到 `reports/`
   - `space` 播放/暂停测试音频 · `h/l` ±10s · `H/L` ±60s
   - `q` / `Esc` / `Ctrl-C` 退出

## VAD 参数调优

离线槽用 RMS 能量 VAD 触发解码，参数（`src/main.rs` 顶部常量）：

| 常量 | 默认值 | 含义 |
|---|---|---|
| `VAD_RMS_THRESHOLD` | 0.012 | RMS 低于此值视为静音。可用 `VAD_RMS_THRESHOLD=0.02 cargo run` 覆盖 |
| `VAD_SILENCE_SEC` | 0.4 | 说话后静音多少秒触发解码 |
| `VAD_MAX_BUFFER_SEC` | 8.0 | 缓冲上限强制触发（防止背景音乐持续高于阈值导致一直不触发） |

**游戏场景调优建议**：游戏背景音乐可能持续高于阈值，导致 VAD 一直判定"说话中"。
这时 `VAD_MAX_BUFFER_SEC` 的 8 秒兜底会强制触发，但延迟较高。可以把阈值调高
（如 `VAD_RMS_THRESHOLD=0.03`）或缩短最大缓冲（如 `VAD_MAX_BUFFER_SEC=4`）。

## 项目结构

```
src/main.rs          TUI 主程序（采集 → 喂音 → 渲染）
scripts/
  download-canary.sh 下载 Canary-180m-flash int8 模型（sherpa-onnx 官方 release）
  download-silero-vad.sh 下载 Silero VAD int8（sherpa-onnx 官方 release）
  download-parakeet-tdt-ctc-110m.sh 下载 Parakeet 110M INT8（sherpa-onnx 官方 release）
测试                  cargo run 的薄封装
Cargo.toml           依赖：arcvoice-core（流式槽）+ sherpa-onnx（离线槽）
```

模型默认路径：`../../game-video/engine/models/streaming/`（流式模型）和
`models/streaming/`（本项目下的离线模型）。用 `ENGINE_MODELS_DIR` 覆盖前者。

## 系统要求

- macOS 14.4+（系统音频采集依赖 ScreenCaptureKit / CoreAudio TAP）
- 终端宿主需有 `NSAudioCaptureUsageDescription` 权限（直接 `cargo run` 可能被 TCC 拒绝，
  可用 `cargo run --example diag_devices` 诊断）
- ffplay 在 PATH 中（用于播放测试音频，可选）

## 关于 Moonshine

调研过 `UsefulSensors/moonshine-streaming-medium`，结论是**当前无法接入**：
- sherpa-onnx 的 Moonshine v2 导出脚本只覆盖 tiny/base，不支持 streaming-medium
- 自行导出 ONNX 受阻于 optimum（锁 transformers 4.x）与 moonshine_streaming 架构
  （仅 transformers 5.x 支持）的版本冲突
- 即使强行导出，streaming encoder 的滑动窗口 ~50Hz 前端与 sherpa-onnx 内置的
  80 维 mel 预处理不兼容（实测 community 的 streaming-small ONNX 加载成功但识别报错）

代码里 `SlotKind::OfflineCanary` 已为未来扩展预留：若 sherpa-onnx 未来原生支持
streaming 系列，加一个枚举值 + build 函数即可。
