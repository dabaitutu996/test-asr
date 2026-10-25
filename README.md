# ASR Compare TUI

采集系统音频（PipeWire / PulseAudio 的 monitor source）或麦克风，实时对比多个 ASR
模型在同一音频流上的识别效果。设计用途：在游戏实况 / FPS 解说等嘈杂场景下，横向
对比不同模型的准确率、抗噪能力、延迟和标点质量。

> 平台：**Linux**（已在 x86_64 / PipeWire 上验证）。原 macOS Core Audio Process Tap
> 实现已整体移除，不再做 macOS 兼容。

## 支持的模型

| 模型 | 类型 | 后端 | 说明 |
|---|---|---|---|
| Zipformer-zh | 真流式 transducer | sherpa-onnx OnlineRecognizer | 中文 |
| Zipformer-en | 真流式 transducer | sherpa-onnx OnlineRecognizer | 英文 |
| Zipformer-en (加强版) | 真流式 transducer | sherpa-onnx OnlineRecognizer | 英文，自研标点切句 + VAD 收尾 |
| Nemotron-en | 真流式 transducer | sherpa-onnx OnlineRecognizer | 英文（NVIDIA Nemotron Speech Streaming 0.6B） |
| **Canary-180m-flash** | **离线 + VAD 触发** | sherpa-onnx OfflineRecognizer | 英文（自带标点+大小写） |
| Parakeet-TDT-CTC-110M | 离线 + 外部 VAD | sherpa-onnx OfflineRecognizer | 英文（NVIDIA Parakeet 110M INT8） |
| Silero-VAD | VAD 前置 | sherpa-onnx VAD | 轻量切句，供离线模型配合使用 |

**流式 vs 离线的差异**：
- 流式模型逐帧增量喂音，自带 partial（实时显示正在识别的内容）和 endpoint（自动断句）。
- 离线模型（Canary / Parakeet）没有 partial/endpoint，默认共用同一个 Silero VAD
  segment；每次 VAD 完成切段后，所有启用的离线模型同时解码同一段音频，结果直接作为
  final 输出。

## 系统依赖

- Rust 1.75+（edition 2021）
- `pulseaudio-utils`：`pactl` 用于枚举输出设备，`parec` 用于抓 monitor source
- 运行中的 PipeWire 或 PulseAudio（PipeWire 走 pulse 兼容层即可）
- `ffmpeg`：把示例音频解码成 16kHz PCM；`ffplay` 可选，用于播放测试音频
- `pactl` / `parec`：输入（麦克风）与输出（monitor source）采集都走 PulseAudio 协议层
- `zstd`（可选）：从本仓库模型镜像下载时需要；没装则自动回退到上游 `.tar.bz2`

Debian / Ubuntu：

```bash
sudo apt install build-essential pulseaudio-utils ffmpeg zstd
```

首次 `cargo build` 时，`sherpa-onnx-sys` 会自动从 GitHub Releases 下载
`sherpa-onnx-v*-linux-x64-static-lib`（约 100MB，缓存在 cargo 的 OUT_DIR 里），
无需手工配置 `.cargo/config.toml`。

## 运行

```bash
# 1. 下载模型
#    默认（Zipformer 中/英 + Silero VAD，约 100MB）：
./scripts/download-models.sh
#    要跑全部 6 个模型对比（追加 Nemotron / Canary / Parakeet，约 750MB）：
./scripts/download-models.sh --all

# 2. 构建并启动
cargo run
# 或 ./测试
```

模型优先从本仓库的 [Release 镜像](https://github.com/dabaitutu996/test-asr/releases/tag/models-v1)
下载（`tar.zst` + `SHA256SUMS` 校验），镜像不可用时自动回退到 sherpa-onnx 官方
`asr-models` release。想强制走上游：

```bash
MODELS_SOURCE=upstream ./scripts/download-models.sh
```

启动后：
1. **选择引擎屏**：数字键 1-9 勾选要加载的模型。模型文件缺失的条目会显示为灰色不可选，
   并提示 `./scripts/download-models.sh`。Enter 确认。
2. **设备选择屏**：
   - 输入端 = 麦克风（PulseAudio source，已排除 monitor）
   - 输出端 = 系统音频，每个输出设备对应它的 monitor source
   - 示例文件 = 项目根目录的 `ARC_Raiders_is_getting_DARKER....mp3`（固定文件名，缺失时该项
     显示"未找到默认示例 mp3"）；它不随仓库分发，需要自行放入，放好后无需音频服务即可跑通链路
3. **预览屏**：观察电平是否跳动，确认采集源正确。Enter 进入主屏。
4. **主屏**：多列并行显示各模型的 partial / finals。
   - `←/→` 切换激活列 · `↑/↓` 滚动 finals · `PgUp/PgDn` 快滚
   - `1-9` 切换某列启用/禁用 · `c` 清空 · `e` 导出 Markdown 报告到 `reports/`
   - `space` 播放/暂停测试音频 · `h/l` ±10s · `H/L` ±60s
   - `q` / `Esc` / `Ctrl-C` 退出

## 模型目录

默认全部放在本仓库 `models/streaming/` 下（`models/` 已 gitignore）：

| 子目录 | 来源 |
|---|---|
| `zipformer-zh/` | `sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23` |
| `zipformer-en/` | `sherpa-onnx-streaming-zipformer-en-2023-06-26` |
| `zipformer-en-enhanced/` | 复用 `zipformer-en` 模型，仅切句策略不同 |
| `nemotron-en/` | `sherpa-onnx-nemotron-speech-streaming-en-0.6b-int8-2026-01-14` |
| `canary-180m-flash/` | `sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8` |
| `parakeet-tdt-ctc-110m/` | `sherpa-onnx-nemo-parakeet_tdt_ctc_110m-en-36000-int8` |
| `../vad/silero_vad/silero_vad.int8.onnx` | `silero_vad.onnx` |

镜像包（release `models-v1`）只含应用实际需要的权重文件（`encoder*.onnx` /
`decoder*.onnx` / `joiner*.onnx` / `model*.onnx` / `tokens.txt`），不含上游的
`test_wavs/`、README 与导出脚本，用来做冒烟测试请另外从上游 release 取 WAV。

不想跑脚本时也可以手动取（以中文模型为例）：

```bash
BASE=https://github.com/dabaitutu996/test-asr/releases/download/models-v1
curl -fLO "$BASE/asr-models-zipformer-zh.tar.zst" "$BASE/SHA256SUMS"
sha256sum -c SHA256SUMS --ignore-missing
tar --zstd -xf asr-models-zipformer-zh.tar.zst -C models/streaming
```

镜像里 zipformer 只带 `encoder.int8.onnx` + `decoder.onnx`（代码优先选这两个），
如需完整权重（float encoder、int8 decoder 等）仍可用
`MODELS_SOURCE=upstream ./scripts/download-models.sh` 从上游拿。

覆盖方式：

| 环境变量 | 作用 |
|---|---|
| `ENGINE_MODELS_DIR` | 流式模型根目录（默认 `models/streaming`） |
| `OFFLINE_MODELS_DIR` | 离线模型根目录（默认同上） |
| `SILERO_VAD_MODEL` | Silero VAD 模型完整路径 |

## VAD 参数调优

离线槽默认使用 sherpa-onnx 的 Silero VAD。RMS 能量 VAD 仍保留为兜底/调试路径：

| 环境变量 | 默认值 | 含义 |
|---|---:|---|
| `VAD_BACKEND` | `silero` | 可设为 `silero` 或 `rms` |
| `SILERO_VAD_THRESHOLD` | `0.5` | Silero 语音概率阈值 |
| `SILERO_VAD_MIN_SILENCE_SEC` | `0.5` | 静音多久后结束 segment |
| `SILERO_VAD_MIN_SPEECH_SEC` | `0.25` | 最短语音段 |
| `SILERO_VAD_MAX_SPEECH_SEC` | `8.0` | 最长语音段，超出会强制切段 |

RMS 兜底：

```bash
VAD_BACKEND=rms VAD_RMS_THRESHOLD=0.02 cargo run
```

流式 endpoint / 热词：

| 环境变量 | 默认值 | 含义 |
|---|---:|---|
| `ASR_ENDPOINT_RULE1` | `2.4` | 尾部静音（秒）触发断句 |
| `ASR_ENDPOINT_RULE2` | `1.2` | 含非静音后的尾部静音（秒） |
| `ASR_ENDPOINT_RULE3` | `20.0` | 最长 utterance（秒） |
| `ASR_HOTWORDS` | 空 | 热词，**逗号或换行**分隔；仅 Zipformer 生效 |

热词会以"每行一个"的形式写入 `hotwords_buf`，统一使用 `hotwords_score = 2.0`，
并自动把解码方式从 `greedy_search` 切到 `modified_beam_search`：

```bash
ASR_HOTWORDS="原神,显卡,RTX 5090" cargo run
```

## 诊断

```bash
# 列出麦克风与系统音频设备，并实测每个输出设备 2 秒采集
cargo run --example diag_devices -- --capture

# 用一段 16kHz 单声道 WAV 直接验证模型（不依赖音频设备）
cargo run --release --example transcribe_wav -- /path/to/16k.wav
```

## 项目结构

```
src/main.rs             TUI 主程序（采集 → 喂音 → 渲染）
src/config.rs           常量、环境变量解析、模型路径
src/models.rs           模型静态描述与文件探测
src/vad.rs              共享 VAD 状态机（Silero / RMS）
src/engine/online.rs    流式槽（OnlineRecognizer）
src/engine/offline.rs   离线槽（OfflineRecognizer + VAD）
src/capture.rs          统一采集抽象（系统音频 / 麦克风 / 示例文件）
src/system_audio/       音频采集（pactl 枚举 + parec 抓流，输入输出共用）
src/segments.rs         标点切句状态机（加强版模型）
src/ui/                 选择屏 / 预览屏 / 主画面
examples/diag_devices.rs   设备与采集诊断
examples/transcribe_wav.rs WAV 冒烟测试
scripts/download-models.sh 模型下载（Release 镜像 + 上游回退）
测试                     cargo run 的薄封装
```

## 已知限制

- 采集依赖 PulseAudio 协议层：纯 ALSA / 无音频服务时只能选示例文件（仍可验证模型与 UI）。
- 设备自身的默认采样率不是 16kHz 时会由 PulseAudio 负责重采样（`parec --rate=16000`）。
- 「示例文件」模式读取仓库根目录的 `ARC_Raiders_is_getting_DARKER....mp3`（及其 `.srt`），
  该媒体文件不随仓库分发，需要自行放入；没有它时请用麦克风或系统音频。
- 模型镜像走 `tar --zstd` 解压，机器上没有 `zstd` 时脚本会自动改用上游 `.tar.bz2`（下载更慢）。

## 关于 Moonshine

调研过 `UsefulSensors/moonshine-streaming-medium`，结论是**当前无法接入**：

- sherpa-onnx 的 Moonshine v2 导出脚本只覆盖 tiny/base，不支持 streaming-medium
- 自行导出 ONNX 受阻于 optimum（锁 transformers 4.x）与 moonshine_streaming 架构
  （仅 transformers 5.x 支持）的版本冲突
- 即使强行导出，streaming encoder 的滑动窗口 ~50Hz 前端与 sherpa-onnx 内置的
  80 维 mel 预处理不兼容（实测 community 的 streaming-small ONNX 加载成功但识别报错）

代码里 `OfflineFamily` 已承载离线模型扩展：不同离线模型只需要补充文件检测与
`OfflineRecognizerConfig` 构建逻辑，并继续共用同一份 VAD segment。
