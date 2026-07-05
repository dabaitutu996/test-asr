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
- 离线模型（Canary / Parakeet）没有 partial/endpoint，默认共用同一个 Silero VAD
  segment；每次 VAD 完成切段后，所有启用的离线模型同时解码同一段音频，结果直接作为
  final 输出。

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
1. **选择引擎屏**：数字键 1-9 勾选要加载的模型。离线模型若未下载会显示为灰色不可选，
   并提示对应的下载脚本。Enter 确认。
2. **设备选择屏**：选输入端（麦克风）或输出端（系统音频，如 BlackHole 2ch）。
3. **预览屏**：观察电平是否跳动，确认采集源正确。Enter 进入主屏。
4. **主屏**：多列并行显示各模型的 partial / finals。
   - `←/→` 切换激活列 · `↑/↓` 滚动 finals · `PgUp/PgDn` 快滚
   - `1-9` 切换某列启用/禁用 · `c` 清空 · `e` 导出 Markdown 报告到 `reports/`
   - `space` 播放/暂停测试音频 · `h/l` ±10s · `H/L` ±60s
   - `q` / `Esc` / `Ctrl-C` 退出

## VAD 参数调优

离线槽默认使用 sherpa-onnx 的 Silero VAD。RMS 能量 VAD 仍保留为兜底/调试路径：

| 环境变量 | 默认值 | 含义 |
|---|---:|---|
| `VAD_BACKEND` | `silero` | 可设为 `silero` 或 `rms` |
| `SILERO_VAD_MODEL` | `models/vad/silero_vad/silero_vad.int8.onnx` | Silero 模型路径 |
| `SILERO_VAD_THRESHOLD` | `0.5` | Silero 语音概率阈值 |
| `SILERO_VAD_MIN_SILENCE_SEC` | `0.5` | 静音多久后结束 segment |
| `SILERO_VAD_MIN_SPEECH_SEC` | `0.25` | 最短语音段 |
| `SILERO_VAD_MAX_SPEECH_SEC` | `8.0` | 最长语音段，超出会强制切段 |

RMS 兜底：

```bash
VAD_BACKEND=rms VAD_RMS_THRESHOLD=0.02 cargo run
```

## 项目结构

```
src/main.rs          TUI 主程序（采集 → 喂音 → 渲染）
src/system_audio/    本仓库内置的 macOS Core Audio Process Tap 采集实现
scripts/
  download-canary.sh 下载 Canary-180m-flash int8 模型（sherpa-onnx 官方 release）
  download-silero-vad.sh 下载 Silero VAD int8（sherpa-onnx 官方 release）
  download-parakeet-tdt-ctc-110m.sh 下载 Parakeet 110M INT8（sherpa-onnx 官方 release）
测试                  cargo run 的薄封装
Cargo.toml           依赖：arcvoice-core（流式槽）+ sherpa-onnx（离线槽/VAD）
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

代码里 `OfflineFamily` 已承载离线模型扩展：不同离线模型只需要补充文件检测与
`OfflineRecognizerConfig` 构建逻辑，并继续共用同一份 VAD segment。
