# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ASR model comparison TUI — captures Linux system audio (PipeWire/PulseAudio monitor source)
or microphone input and compares real-time recognition results from multiple Sherpa streaming
ASR models side by side. **Linux-only** (x86_64 verified); the previous macOS Core Audio
Process Tap implementation has been removed.

## Build & Run

```bash
cargo build
cargo run
# or use the convenience script:
./测试
```

System prerequisites: `pulseaudio-utils` (`pactl`/`parec`), a running PipeWire or PulseAudio,
and `ffmpeg` (sample-audio decoding). `sherpa-onnx-sys`'s build script downloads the linux-x64
prebuilt static libraries from GitHub Releases on first build, so no `.cargo/config.toml`
tuning is required.

Models live in `models/streaming/<subdir>/` (gitignored); download them with
`./scripts/download-models.sh`. Override the roots with `ENGINE_MODELS_DIR` /
`OFFLINE_MODELS_DIR`, and the Silero VAD path with `SILERO_VAD_MODEL`.

## Architecture

TUI app with `src/main.rs` as entry point. Runtime phases:

1. **Selection screen** — user picks which ASR engines to load (1-9 toggle, Enter confirms, q quits).
   Engines whose model files are missing are greyed out (`ModelDesc::files_present`).
2. **Model loading** — builds the selected Sherpa online/offline models (C library stdout is
   suppressed via fd redirection in `util::with_stdout_suppressed`).
3. **Device screen → preview** — picks one of: microphone (PulseAudio source), system audio
   output (monitor source), or the bundled sample mp3.
4. **Main TUI loop** — feeds PCM frames to online engines, broadcasts shared VAD segments to
   offline engines, renders partial/final text per column with ratatui.

Key types:
- `ModelDesc` / `MODEL_DESCS` — static model metadata (name, subdir, backend kind, language)
- `OnlineSlot` / `OfflineSlot` — runtime state per engine
- `VadState` — shared Silero/RMS segmentation for offline models
- `Capture` / `DevicePick` — unified capture abstraction (system audio / mic / file)
- `App` — holds all slots, VAD state, log buffer, RMS level, start time

## Audio capture details

- `src/system_audio/linux.rs` enumerates devices with `pactl -f json list sinks|sources`
  (sinks expose a `monitor_source`; sources ending in `.monitor` are filtered out for input)
  and streams the chosen source through `parec --format=float32le --rate=16000 --channels=1`.
  The reader thread batches samples into 200 ms frames (`FRAME_SAMPLES = 3200`) and pushes them
  over a `tokio::sync::mpsc` channel. Dropping `PulseCapture` kills the child process.
- Input (microphone) and output (system audio) capture share the same `PulseCapture`
  implementation; `system_audio::start_source` handles both, `system_audio::start` resolves the
  default sink's monitor when no uid is given.
- `src/capture.rs` unifies all three sources (mic / system audio / sample file); `Capture::rx` is
  the single producer entry point for the main loop.

## Dependencies

- `sherpa-onnx` — online/offline recognizers, Silero VAD
- `serde_json` — parsing `pactl -f json` output
- `ratatui` + `crossterm` — TUI rendering
- `tokio` — `mpsc` channel (capture → main loop)

There is no `arcvoice-core` dependency anymore; the small wrappers the app needed
(`SherpaModelType`, `Language`) live in `src/models.rs`.
