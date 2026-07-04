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

Single-file TUI app (`src/main.rs`) with three runtime phases:

1. **Selection screen** — user picks which ASR engines to load (1/2/3 toggle, Enter confirms, q quits)
2. **Model loading** — loads selected Sherpa models via `arcvoice_core` (suppresses C library stdout noise via fd redirect)
3. **Main TUI loop** — captures system audio via `arcvoice_core::audio::system_audio`, feeds PCM frames to each engine's stream, renders partial/final results in columns using ratatui

Key types:
- `ModelDesc` — static model metadata (name, subdir, SherpaModelType, language)
- `ModelSlot` — runtime state per engine (stream, partial text, finals history, enabled flag)
- `App` — holds all slots, log buffer, RMS level, start time

## Dependencies

- `arcvoice_core` (local path dep from `game-video/engine/crates/core`) — Sherpa ONNX bindings and system audio capture
- `ratatui` + `crossterm` — TUI rendering
- `tokio` — only used for `mpsc` channel (system audio → main loop)
