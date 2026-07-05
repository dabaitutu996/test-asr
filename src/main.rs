//! ASR 模型对比 TUI 工具：采集系统音频，实时对比多个 Sherpa 流式模型的识别效果。
//!
//! 启动时进入选择界面，用 1-9 勾选要加载的引擎，Enter 确认，q 退出。
//! 运行中：q / Esc / Ctrl-C 退出；c 清空历史；1-9 切换引擎启用/禁用。
//!
//! 模型路径指向 game-video/engine/models/streaming/。
//! 运行：cargo run
//!
//! 代码按职责拆分为多个模块：
//! - `config`  — 常量、环境变量解析、模型路径
//! - `util`    — stdout 屏蔽、RMS、日志等小工具
//! - `models`  — 模型静态描述（ModelDesc / MODEL_DESCS）
//! - `vad`     — 离线槽共享的 VAD 状态机
//! - `engine`  — 流式 / 离线引擎槽位（AnySlot / SlotView）
//! - `capture` — 设备选择与采集抽象（Capture / DevicePick）
//! - `media`   — 媒体播放与字幕（MediaState）
//! - `report`  — Markdown 报告导出
//! - `app`     — 应用运行态与键盘交互（App）
//! - `ui`      — 各交互屏与主画面渲染

#![cfg(not(target_os = "windows"))]

mod app;
mod capture;
mod config;
mod engine;
mod media;
mod models;
mod report;
mod segments;
mod ui;
mod util;
mod vad;

use std::io;
use std::time::Instant;

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Terminal;

use crate::app::App;
use crate::capture::{Capture, DevicePick};
use crate::config::POLL_INTERVAL;
use crate::engine::{decode_offline_segment, feed_online_frame, AnySlot};
use crate::media::MediaState;
use crate::models::MODEL_DESCS;
use crate::ui::{draw, run_device_screen, run_preview_screen, run_selection_screen};
use crate::util::{humantime_elapsed, push_log, rms};
use crate::vad::VadState;

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
        prev_hook(info);
    }));

    // 阶段 1：选择引擎 + 切句配置。
    let (indices, seg_config) = match run_selection_screen(&mut terminal)? {
        Some((idx, cfg)) => (idx, cfg),
        None => {
            disable_raw_mode()?;
            execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
            terminal.show_cursor()?;
            println!("已取消");
            return Ok(());
        }
    };

    // 阶段 2：加载选中的模型。
    terminal.draw(|frame| {
        let area = frame.area();
        let names: Vec<&str> = indices.iter().map(|&i| MODEL_DESCS[i].name).collect();
        let cfg_tag = if seg_config.is_some() { " · 加强版" } else { "" };
        let msg = format!(" 正在加载: {} ... {cfg_tag}", names.join(", "));
        let loading = Paragraph::new(Line::from(Span::styled(
            msg,
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        )))
        .block(Block::default().borders(Borders::ALL).title("加载中"));
        frame.render_widget(loading, area);
    })?;

    let mut slots = Vec::with_capacity(indices.len());
    for &i in &indices {
        // 只有流式模型（且选了加强配置的）才传 seg_config
        let model_config = if matches!(MODEL_DESCS[i].kind, crate::models::SlotKind::Online(_)) {
            seg_config.as_ref()
        } else {
            None
        };
        slots.push(AnySlot::build_with_config(&MODEL_DESCS[i], model_config)?);
    }
    let has_offline_slots = slots.iter().any(|slot| !slot.is_online());
    let vad = VadState::new(has_offline_slots)?;

    // 阶段 3 + 4：设备选择 → 预览。两屏之间可往返（预览 Esc 回设备选择）。
    let (device_pick, mut capture, source_label) = loop {
        let (pick, inputs, outputs) = match run_device_screen(&mut terminal)? {
            Some(v) => v,
            None => {
                // 取消：退回模型选择。这里直接退出（简化），与原 q 取消语义一致。
                disable_raw_mode()?;
                execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
                terminal.show_cursor()?;
                println!("已取消");
                return Ok(());
            }
        };
        match run_preview_screen(&mut terminal, &pick, &inputs, &outputs)? {
            Some(capture) => {
                let label = pick.label(&inputs, &outputs);
                break (pick, capture, label);
            }
            None => continue, // Esc：回设备选择屏重选
        }
    };
    let sample_file_mode = matches!(device_pick, DevicePick::SampleFile(_));

    let media = MediaState::load_default()?;
    let mut app = App {
        slots,
        vad,
        log: Vec::new(),
        last_rms: 0.0,
        started_at: Instant::now(),
        source_label,
        active_slot: 0,
        media,
        sample_file_mode,
    };
    if let Some(media) = &mut app.media {
        let start_result = if capture.is_sample_file() {
            media.start_clock_only();
            Ok(())
        } else {
            media.start()
        };
        match start_result {
            Ok(()) => {
                let action = if capture.is_sample_file() {
                    "已开始示例文件回放"
                } else {
                    "已开始播放"
                };
                app.log.push(format!(
                    "{action}: {}",
                    media
                        .audio_path
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                ));
            }
            Err(e) => app.log.push(format!("音频播放启动失败: {e:#}")),
        }
    }

    // 阶段 5：主 TUI 循环。
    let result = run_loop(&mut terminal, &mut app, &mut capture);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    println!("已退出（运行 {}）", humantime_elapsed(app.started_at));
    result
}

// ─── 主循环 ────────────────────────────────────────────────────────────

fn run_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
    capture: &mut Capture,
) -> Result<()> {
    loop {
        while let Ok(frame) = capture.rx.try_recv() {
            app.last_rms = rms(&frame);

            // 1) 流式槽：增量喂当前帧（沿用原有 feed 逻辑）。
            for slot in &mut app.slots {
                if let AnySlot::Online(s) = slot {
                    if let Some(final_text) = feed_online_frame(s, &frame) {
                        push_log(&mut app.log, format!("[{}] final: {}", s.name, final_text));
                    }
                }
            }

            // 2) 离线槽：喂 VAD 状态机；触发时把整段广播给所有离线槽。
            for segment in app.vad.push(&frame, &mut app.log) {
                for slot in &mut app.slots {
                    if let AnySlot::Offline(s) = slot {
                        if let Some(final_text) = decode_offline_segment(s, &segment) {
                            push_log(&mut app.log, format!("[{}] final: {}", s.name, final_text));
                        }
                    }
                }
            }
        }

        if event::poll(POLL_INTERVAL)? {
            if let Event::Key(key) = event::read()? {
                if should_quit(&key) {
                    return Ok(());
                }
                if matches!(key.code, KeyCode::Char('c')) {
                    app.clear();
                }
                app.handle_media_key(&key, capture);
                match key.code {
                    KeyCode::Left => app.move_active_slot(-1),
                    KeyCode::Right => app.move_active_slot(1),
                    KeyCode::Up => app.scroll_active_slot(-1),
                    KeyCode::Down => app.scroll_active_slot(1),
                    KeyCode::PageUp => app.scroll_active_slot(-6),
                    KeyCode::PageDown => app.scroll_active_slot(6),
                    KeyCode::Home => app.reset_active_scroll(),
                    KeyCode::Char(c @ '1'..='9') => {
                        let idx = (c as u8 - b'1') as usize;
                        app.toggle(idx);
                    }
                    _ => {}
                }
            }
        }

        app.refresh_media();

        // 样本文件自然播完后，同步暂停 media 时钟（否则时钟继续走秒）。
        if let Some(media) = &mut app.media {
            if capture.is_sample_file() && media.playing && !capture.is_sample_file_playing() {
                media.pause_clock();
                push_log(&mut app.log, "示例文件播放完毕");
            }
        }

        terminal.draw(|frame| draw(app, frame))?;
    }
}

fn should_quit(key: &KeyEvent) -> bool {
    matches!(key.code, KeyCode::Char('q'))
        || (key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL))
        || matches!(key.code, KeyCode::Esc)
}
