//! ASR 模型对比 TUI 工具：采集系统音频，实时对比多个 Sherpa 流式模型的识别效果。
//!
//! 启动时进入选择界面，用 1/2/3 勾选要加载的引擎，Enter 确认，q 退出。
//! 运行中：q / Esc / Ctrl-C 退出；c 清空历史；1/2/3 切换引擎启用/禁用。
//!
//! 模型路径指向 game-video/engine/models/streaming/。
//! 运行：cargo run

#![cfg(not(target_os = "windows"))]

use std::io;
use std::os::unix::io::RawFd;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Terminal;

use arcvoice_core::asr::streaming::{
    OnlineAsrEngine, OnlineAsrStream, SherpaConfig, SherpaModelType, SherpaOnlineAsrEngine,
};
use arcvoice_core::asr::Language;
use arcvoice_core::audio::system_audio;

/// 顶部 partial 之外，每个 slot 最多保留多少条 final 历史。
const FINALS_RETAIN: usize = 6;
/// TUI 轮询周期：终端事件与重绘节奏。
const POLL_INTERVAL: Duration = Duration::from_millis(50);

// ─── 工具函数 ──────────────────────────────────────────────────────────

/// 在闭包执行期间临时将 stdout（fd 1）重定向到 /dev/null，屏蔽 C 库的
/// 调试输出（sherpa-onnx 加载时会 dump 大量 encoder/decoder 配置）。
fn with_stdout_suppressed<F: FnOnce() -> R, R>(f: F) -> R {
    extern "C" {
        fn dup(fd: RawFd) -> RawFd;
        fn dup2(oldfd: RawFd, newfd: RawFd) -> RawFd;
        fn close(fd: RawFd) -> RawFd;
        fn open(path: *const u8, oflag: RawFd, ...) -> RawFd;
    }
    const O_WRONLY: RawFd = 1;

    unsafe {
        let saved_stdout = dup(1);
        let dev_null = open(b"/dev/null\0".as_ptr(), O_WRONLY);
        if dev_null >= 0 {
            dup2(dev_null, 1);
            close(dev_null);
        }

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));

        if saved_stdout >= 0 {
            dup2(saved_stdout, 1);
            close(saved_stdout);
        }

        match result {
            Ok(val) => val,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }
}

/// engine 模型基础目录：从本项目 Cargo.toml 向上两级到 Desktop，
/// 再进入 game-video/engine/models/streaming/。
fn engine_models_dir() -> PathBuf {
    // 优先用环境变量覆盖，方便灵活部署。
    if let Ok(dir) = std::env::var("ENGINE_MODELS_DIR") {
        return PathBuf::from(dir);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../game-video/engine/models/streaming")
}

// ─── 模型静态描述 ─────────────────────────────────────────────────────

struct ModelDesc {
    name: &'static str,
    subdir: &'static str,
    model_type: SherpaModelType,
    language: Language,
}

const MODEL_DESCS: &[ModelDesc] = &[
    ModelDesc {
        name: "Zipformer-zh",
        subdir: "zipformer-zh",
        model_type: SherpaModelType::Zipformer,
        language: Language::Chinese,
    },
    ModelDesc {
        name: "Zipformer-en",
        subdir: "zipformer-en",
        model_type: SherpaModelType::Zipformer,
        language: Language::English,
    },
    ModelDesc {
        name: "Nemotron-en",
        subdir: "nemotron-en",
        model_type: SherpaModelType::NemotronStreaming,
        language: Language::English,
    },
];

// ─── 运行态 ───────────────────────────────────────────────────────────

struct ModelSlot {
    name: &'static str,
    #[allow(dead_code)]
    engine: SherpaOnlineAsrEngine,
    stream: Box<dyn OnlineAsrStream>,
    partial: String,
    finals: Vec<String>,
    enabled: bool,
}

fn build_slot(desc: &ModelDesc) -> Result<ModelSlot> {
    let model_dir = engine_models_dir().join(desc.subdir);
    let cfg = SherpaConfig::new(model_dir.clone(), desc.model_type, desc.language);
    let engine = with_stdout_suppressed(|| SherpaOnlineAsrEngine::load(cfg))
        .with_context(|| format!("加载模型 {} 失败，目录 {:?} 是否存在", desc.name, model_dir))?;
    let mut stream = engine.create_stream()?;
    stream.prepare();
    Ok(ModelSlot {
        name: desc.name,
        engine,
        stream,
        partial: String::new(),
        finals: Vec::new(),
        enabled: true,
    })
}

fn feed_frame(slot: &mut ModelSlot, pcm: &[f32]) -> Option<String> {
    if !slot.enabled {
        return None;
    }
    slot.stream.accept_waveform(pcm);
    slot.stream.decode();
    slot.partial = slot.stream.current_partial();
    if slot.stream.is_endpoint() {
        let final_text = slot.stream.take_final();
        slot.partial.clear();
        if !final_text.trim().is_empty() {
            slot.finals.push(final_text.clone());
            if slot.finals.len() > FINALS_RETAIN {
                slot.finals.remove(0);
            }
            return Some(final_text);
        }
    }
    None
}

// ─── 应用状态 ─────────────────────────────────────────────────────────

struct App {
    slots: Vec<ModelSlot>,
    log: Vec<String>,
    last_rms: f32,
    started_at: Instant,
}

impl App {
    fn clear(&mut self) {
        for slot in &mut self.slots {
            slot.partial.clear();
            slot.finals.clear();
        }
        self.log.clear();
    }

    fn toggle(&mut self, index: usize) {
        if let Some(slot) = self.slots.get_mut(index) {
            slot.enabled = !slot.enabled;
            if !slot.enabled {
                slot.partial.clear();
            }
            let state = if slot.enabled { "启用" } else { "禁用" };
            self.log.push(format!("[{}] 已{}", slot.name, state));
            if self.log.len() > 50 {
                self.log.remove(0);
            }
        }
    }
}

// ─── 启动选择界面 ─────────────────────────────────────────────────────

fn run_selection_screen(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
) -> Result<Option<Vec<usize>>> {
    let mut selected = vec![true; MODEL_DESCS.len()];

    loop {
        terminal.draw(|frame| {
            let area = frame.area();
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(3),
                    Constraint::Min(6),
                    Constraint::Length(3),
                ])
                .split(area);

            let title = Paragraph::new(Line::from(vec![Span::styled(
                " ASR Compare — 选择要加载的引擎 ",
                Style::default().add_modifier(Modifier::BOLD),
            )]))
            .block(Block::default().borders(Borders::ALL));
            frame.render_widget(title, chunks[0]);

            let mut lines: Vec<Line> = Vec::new();
            for (i, desc) in MODEL_DESCS.iter().enumerate() {
                let check = if selected[i] { "[x]" } else { "[ ]" };
                let lang = match desc.language {
                    Language::Chinese => "中文",
                    Language::English => "英文",
                };
                let style = if selected[i] {
                    Style::default().fg(Color::Green)
                } else {
                    Style::default().fg(Color::DarkGray)
                };
                lines.push(Line::from(vec![
                    Span::styled(format!(" {} ", i + 1), Style::default().fg(Color::Yellow)),
                    Span::styled(format!("{check} {} ({lang})", desc.name), style),
                ]));
            }
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                " 按数字键切换勾选 · Enter 确认 · q 退出 ",
                Style::default().fg(Color::DarkGray),
            )));
            let list = Paragraph::new(lines)
                .block(Block::default().borders(Borders::ALL).title("选择引擎"));
            frame.render_widget(list, chunks[1]);

            let count = selected.iter().filter(|&&s| s).count();
            let hint = if count == 0 {
                " 请至少选择一个引擎".to_string()
            } else {
                format!(" 将加载 {count} 个引擎（模型较大，加载需几秒）")
            };
            let hint_style = if count == 0 {
                Style::default().fg(Color::Red)
            } else {
                Style::default().fg(Color::Green)
            };
            let footer = Paragraph::new(Line::from(Span::styled(hint, hint_style)))
                .block(Block::default().borders(Borders::ALL));
            frame.render_widget(footer, chunks[2]);
        })?;

        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => return Ok(None),
                    KeyCode::Char('1') => selected[0] = !selected[0],
                    KeyCode::Char('2') => selected[1] = !selected[1],
                    KeyCode::Char('3') => selected[2] = !selected[2],
                    KeyCode::Enter | KeyCode::Char('\n') | KeyCode::Char('\r') => {
                        let indices: Vec<usize> = selected
                            .iter()
                            .enumerate()
                            .filter(|(_, &s)| s)
                            .map(|(i, _)| i)
                            .collect();
                        if !indices.is_empty() {
                            return Ok(Some(indices));
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}

// ─── 主函数 ────────────────────────────────────────────────────────────

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

    // 阶段 1：选择引擎。
    let indices = match run_selection_screen(&mut terminal)? {
        Some(i) => i,
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
        let msg = format!(" 正在加载: {} ... ", names.join(", "));
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
        slots.push(build_slot(&MODEL_DESCS[i])?);
    }

    // 系统音频采集：Process Tap 自动采集全部系统音频输出。
    let stop_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let rebuild_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let (_tap, mut rx) = system_audio::start(stop_flag, rebuild_flag, None)
        .context("启动系统音频采集失败，需要 macOS 14.4 或更高版本")?;

    let mut app = App {
        slots,
        log: Vec::new(),
        last_rms: 0.0,
        started_at: Instant::now(),
    };

    // 阶段 3：主 TUI 循环。
    let result = run_loop(&mut terminal, &mut app, &mut rx);

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
    rx: &mut tokio::sync::mpsc::Receiver<Vec<f32>>,
) -> Result<()> {
    loop {
        while let Ok(frame) = rx.try_recv() {
            app.last_rms = rms(&frame);
            for slot in &mut app.slots {
                if let Some(final_text) = feed_frame(slot, &frame) {
                    app.log
                        .push(format!("[{}] final: {}", slot.name, final_text));
                    if app.log.len() > 50 {
                        app.log.remove(0);
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
                match key.code {
                    KeyCode::Char('1') => app.toggle(0),
                    KeyCode::Char('2') => app.toggle(1),
                    KeyCode::Char('3') => app.toggle(2),
                    _ => {}
                }
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

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum = samples.iter().map(|s| s * s).sum::<f32>();
    (sum / samples.len() as f32).sqrt()
}

fn humantime_elapsed(start: Instant) -> String {
    let s = start.elapsed().as_secs();
    format!("{}分{}秒", s / 60, s % 60)
}

// ─── 渲染 ─────────────────────────────────────────────────────────────

fn draw(app: &mut App, frame: &mut ratatui::Frame) {
    let area = frame.area();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(8),
            Constraint::Length(6),
        ])
        .split(area);

    let title = Line::from(vec![
        Span::styled(
            " ASR Compare ",
            Style::default().add_modifier(Modifier::BOLD),
        ),
        Span::raw("· "),
        Span::styled("q", Style::default().fg(Color::Yellow)),
        Span::raw(" 退出 · "),
        Span::styled("c", Style::default().fg(Color::Yellow)),
        Span::raw(" 清空 · "),
        Span::styled("1/2/3", Style::default().fg(Color::Yellow)),
        Span::raw(" 切换引擎 · "),
        Span::raw(format!("音频 RMS {:.4}", app.last_rms)),
    ]);
    let top = Paragraph::new(title).block(Block::default().borders(Borders::ALL).title("实时对比"));
    frame.render_widget(top, chunks[0]);

    let count = app.slots.len().max(1);
    let col_constraints: Vec<Constraint> = (0..count)
        .map(|_| Constraint::Ratio(1, count as u32))
        .collect();
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints(col_constraints)
        .split(chunks[1]);

    for (slot, col_area) in app.slots.iter_mut().zip(columns.iter()) {
        let dim = !slot.enabled;
        let text_style = if dim {
            Style::default().fg(Color::DarkGray)
        } else {
            Style::default()
        };

        let mut lines: Vec<Line> = Vec::new();
        let partial_label = if dim { "[已关闭]" } else { &slot.partial };
        lines.push(Line::from(vec![
            Span::styled("partial: ", Style::default().fg(Color::Cyan)),
            Span::styled(partial_label.to_string(), text_style),
        ]));
        lines.push(Line::from(Span::styled(
            "— finals —",
            Style::default().fg(Color::DarkGray),
        )));
        for f in &slot.finals {
            lines.push(Line::from(Span::styled(format!("· {f}"), text_style)));
        }
        let title_suffix = if dim { " [关]" } else { "" };
        let title_style = if dim {
            Style::default().fg(Color::DarkGray)
        } else {
            Style::default().add_modifier(Modifier::BOLD)
        };
        let para = Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(
            Span::styled(format!(" {}{} ", slot.name, title_suffix), title_style),
        ));
        frame.render_widget(para, *col_area);
    }

    let log_lines: Vec<Line> = app
        .log
        .iter()
        .rev()
        .take(5)
        .map(|s| Line::from(s.clone()))
        .collect();
    let log_para = Paragraph::new(log_lines).block(
        Block::default()
            .borders(Borders::ALL)
            .title("日志（最近 endpoint）"),
    );
    frame.render_widget(log_para, chunks[2]);
}
