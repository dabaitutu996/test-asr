# Repository Guidelines

## Project Structure & Module Organization

This repository is a small Rust TUI for comparing streaming ASR models side by side on macOS. The main application lives in `src/main.rs`. Diagnostic tooling for audio device inspection lives in `examples/diag_devices.rs`. Cargo metadata is in `Cargo.toml`, and local build environment defaults are in `.cargo/config.toml`. The `测试` script is a thin convenience wrapper around `cargo run`.

The app depends on a local path crate, `arcvoice-core`, from `../../game-video/engine/crates/core`, and model assets default to `../../game-video/engine/models/streaming/`. Override the model path with `ENGINE_MODELS_DIR` when needed.

## Build, Test, and Development Commands

- `cargo build`: compile the TUI and validate local dependency wiring.
- `cargo run`: start the interactive ASR comparison UI.
- `./测试`: equivalent to `cargo run`.
- `cargo run --example diag_devices`: inspect input/output devices and verify system audio capture.
- `cargo fmt`: format Rust code before review.
- `cargo test`: run tests if present; this repo currently has little automated coverage, so keep it green when adding tests.

## Coding Style & Naming Conventions

Use standard Rust formatting with `cargo fmt`; prefer 4-space indentation and idiomatic Rust naming: `snake_case` for functions/modules, `CamelCase` for types, `SCREAMING_SNAKE_CASE` for constants. Keep `main.rs` readable by grouping related helpers with short section comments, following the current layout. Favor `anyhow::Result` with contextual errors for runtime failures.

## Testing Guidelines

There is no large test suite yet. When changing capture, model loading, or TUI behavior, add focused tests where practical and verify manually with `cargo run` and `cargo run --example diag_devices`. Name new tests by behavior, for example `loads_selected_models` or `ignores_disabled_slot`.

## Commit & Pull Request Guidelines

Recent history uses short, descriptive Chinese commit subjects, sometimes with a full-width colon, for example `简化启动：.cargo/config.toml 自动注入环境变量`. Keep commits concise and outcome-focused.

PRs should explain user-visible behavior changes, note any macOS or audio-capture prerequisites, list commands run for verification, and include screenshots or terminal captures for TUI changes.

## Environment & Configuration Tips

System audio capture is macOS-only and currently expects macOS 14.4+ behavior. `.cargo/config.toml` injects sherpa library paths automatically; update it carefully if the sibling `game-video` build output moves.
