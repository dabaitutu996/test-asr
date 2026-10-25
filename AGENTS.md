# Repository Guidelines

## Project Structure & Module Organization

This repository is a small Rust TUI for comparing streaming ASR models side by side on Linux.
The main application lives in `src/main.rs`, split by responsibility: `config` (constants, env
parsing, model paths), `models` (static model descriptors and file resolution), `vad` (shared
Silero/RMS segmentation), `engine/` (online and offline recognizer slots), `capture` (unified
capture abstraction), `system_audio/` (PulseAudio enumeration and `parec` streaming), `ui/`
(screens and main render), plus `app`, `media`, `report`, `segments`, and `util`.

Diagnostic tooling lives in `examples/diag_devices.rs` (device enumeration and 2s capture test)
and `examples/transcribe_wav.rs` (WAV smoke test without audio hardware). Cargo metadata is in
`Cargo.toml`; the `测试` script is a thin convenience wrapper around `cargo run`.

Audio capture goes through PipeWire/PulseAudio: `pactl -f json` enumerates sources/sinks and
`parec` streams the selected source. Models default to `models/streaming/` and are downloaded
with `./scripts/download-models.sh`; override with `ENGINE_MODELS_DIR` / `OFFLINE_MODELS_DIR`,
and the Silero VAD path with `SILERO_VAD_MODEL`.

## Build, Test, and Development Commands

- `cargo build`: compile the TUI. The first build downloads the sherpa-onnx linux-x64 prebuilt
  static libraries via `sherpa-onnx-sys`'s build script (needs network access).
- `cargo run` / `./测试`: start the interactive ASR comparison UI.
- `cargo test`: run unit tests (model file resolution, `pactl` JSON parsing, segmenter state).
- `cargo fmt`: format Rust code before review.
- `cargo run --example diag_devices -- --capture`: list devices and measure 2s per source.
- `cargo run --release --example transcribe_wav -- <16k-mono.wav>`: verify model loading and
  recognition without touching the audio stack.

## Coding Style & Naming Conventions

Use standard Rust formatting with `cargo fmt`; prefer 4-space indentation and idiomatic Rust
naming: `snake_case` for functions/modules, `CamelCase` for types, `SCREAMING_SNAKE_CASE` for
constants. Keep `main.rs` readable by grouping related helpers with short section comments,
following the current layout. Favor `anyhow::Result` with contextual errors for runtime failures,
and keep user-facing messages in Chinese to match the existing TUI text.

## Testing Guidelines

Tests live in `#[cfg(test)] mod tests` blocks next to the code they cover. When changing capture,
model loading, or parsing logic, add focused tests where practical and verify manually with
`cargo run` and `cargo run --example diag_devices -- --capture`. Name tests by behavior, for
example `resolves_zipformer_prefers_float_decoder` or `parses_monitor_sources_from_pactl_json`.

## Commit & Pull Request Guidelines

Recent history uses short, descriptive Chinese commit subjects, sometimes with a full-width colon,
for example `简化启动：.cargo/config.toml 自动注入环境变量`. Keep commits concise and
outcome-focused.

PRs should explain user-visible behavior changes, note audio-capture prerequisites (PipeWire or
PulseAudio with `pactl`/`parec`), list commands run for verification, and include terminal
captures for TUI changes.

## Environment & Configuration Tips

- Linux only (x86_64 verified). Capture requires a running PipeWire/PulseAudio instance; without
  one, only the bundled sample-file mode works.
- Model filenames from sherpa-onnx releases (`encoder-epoch-99-avg-1.int8.onnx`) are normalized
  by the download script; the resolver also falls back to prefix matching for manual extracts.
- `models/`, `reports/`, and `target/` are gitignored.

