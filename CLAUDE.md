# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ASR model comparison TUI tool — captures system audio and compares real-time recognition results from multiple Sherpa streaming ASR models side by side. macOS-only (`cfg(not(target_os = "windows"))`), requires macOS 14.4+ for system audio capture.

## Build & Run

```bash
cargo build
cargo run
# or use the convenience script:
./测试
```

The `.cargo/config.toml` automatically sets `SHERPA_ONNX_LIB_DIR` and `SHERPA_LIB_PATH` environment variables pointing to prebuilt sherpa-onnx libraries under `../../game-video/engine/target/`.

The model directory defaults to `../../game-video/engine/models/streaming/` (relative to Cargo.toml). Override with `ENGINE_MODELS_DIR` env var.

## Architecture

TUI app with `src/main.rs` as the entry point and `src/system_audio/` carrying the local macOS Process Tap capture code. Runtime phases:

1. **Selection screen** — user picks which ASR engines to load (1-9 toggle, Enter confirms, q quits)
2. **Model loading** — loads selected Sherpa online/offline models (suppresses C library stdout noise via fd redirect)
3. **Main TUI loop** — captures system audio via local `asr_compare_tui::system_audio`, feeds PCM frames to online engines, broadcasts shared VAD segments to offline engines, and renders partial/final results in columns using ratatui

Key types:
- `ModelDesc` — static model metadata (name, subdir, backend kind, language)
- `OnlineSlot` / `OfflineSlot` — runtime state per engine
- `VadState` — shared Silero/RMS segmentation for offline models
- `App` — holds all slots, VAD state, log buffer, RMS level, start time

## Dependencies

- `arcvoice_core` (local path dep from `game-video/engine/crates/core`) — Sherpa online ASR wrappers and microphone capture
- `sherpa-onnx` — offline ASR and Silero VAD
- `objc2` / `core-foundation` — local macOS system audio capture
- `ratatui` + `crossterm` — TUI rendering
- `tokio` — `mpsc` channel (capture → main loop)
