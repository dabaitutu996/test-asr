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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Terminal;
use tokio::sync::mpsc;

use arcvoice_core::asr::streaming::{
    OnlineAsrEngine, OnlineAsrStream, SherpaConfig, SherpaModelType, SherpaOnlineAsrEngine,
};
use arcvoice_core::asr::Language;
use arcvoice_core::audio::mic::MicInput;
use arcvoice_core::audio::system_audio::{self, OutputDevice};

/// 顶部 partial 之外，每个 slot 最多保留多少条 final 历史。
const FINALS_RETAIN: usize = 6;
/// TUI 轮询周期：终端事件与重绘节奏。
const POLL_INTERVAL: Duration = Duration::from_millis(50);

// ─── 工具函数 ──────────────────────────────────────────────────────────

/// 在闭包执行期间临时将 stdout（fd 1）和 stderr（fd 2）重定向到 /dev/null，
/// 屏蔽 C 库的调试输出（sherpa-onnx / onnxruntime 加载时会通过 stderr dump
/// 大量 encoder/decoder 配置与 CoreML 警告，这些原始字节会与 TUI 画面混在一起）。
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
        let saved_stderr = dup(2);
        let dev_null = open(b"/dev/null\0".as_ptr(), O_WRONLY);
        if dev_null >= 0 {
            dup2(dev_null, 1);
            dup2(dev_null, 2);
            close(dev_null);
        }

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));

        if saved_stdout >= 0 {
            dup2(saved_stdout, 1);
            close(saved_stdout);
        }
        if saved_stderr >= 0 {
            dup2(saved_stderr, 2);
            close(saved_stderr);
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

// ─── 设备选择与采集抽象 ──────────────────────────────────────────────

/// 用户在设备选择屏挑中的采集源。
/// Input 存 cpal 用的设备名，Output 存 CoreAudio 的设备 UID。
enum DevicePick {
    Input(String),
    Output(String),
}

impl DevicePick {
    /// 给 UI 显示用的标签，如「输入端 / MacBook 麦克风」。
    fn label(&self, _inputs: &[String], outputs: &[OutputDevice]) -> String {
        match self {
            DevicePick::Input(name) => {
                if name == "default" {
                    "输入端 / 默认麦克风".to_string()
                } else {
                    format!("输入端 / {name}")
                }
            }
            DevicePick::Output(uid) => {
                let dev_name = outputs
                    .iter()
                    .find(|d| &d.uid == uid)
                    .map(|d| d.name.as_str())
                    .unwrap_or(uid);
                format!("输出端 / {dev_name}")
            }
        }
    }
}

/// 把 system_audio（Process Tap）和 mic（cpal）两条采集路径收拢成统一接口。
/// 预览屏和主屏都只跟 Capture 打交道，不再 if 分支。
struct Capture {
    rx: mpsc::Receiver<Vec<f32>>,
    stop_flag: Arc<AtomicBool>,
    /// 持有底层 stream/tap 保活；drop 即停止采集。
    _guard: CaptureGuard,
}

enum CaptureGuard {
    #[allow(dead_code)]
    System(system_audio::SystemAudioTap),
    #[allow(dead_code)]
    Mic(MicInput),
}

impl Capture {
    fn start(pick: &DevicePick) -> Result<Capture> {
        let stop_flag = Arc::new(AtomicBool::new(false));
        match pick {
            DevicePick::Output(uid) => {
                let rebuild = Arc::new(AtomicBool::new(false));
                let (tap, rx) =
                    system_audio::start(stop_flag.clone(), rebuild, Some(uid.as_str()))?;
                Ok(Capture {
                    rx,
                    stop_flag,
                    _guard: CaptureGuard::System(tap),
                })
            }
            DevicePick::Input(name) => {
                let resolved = if name.is_empty() {
                    "default"
                } else {
                    name.as_str()
                };
                let (mic, rx) = MicInput::start_selected_with_stop(resolved, stop_flag.clone())?;
                Ok(Capture {
                    rx,
                    stop_flag,
                    _guard: CaptureGuard::Mic(mic),
                })
            }
        }
    }

    fn stop(&self) {
        self.stop_flag.store(true, Ordering::Release);
    }
}

/// 枚举输入端（麦克风）。cpal 失败时返回空 Vec，由调用方处理。
fn list_inputs() -> Vec<String> {
    let host = cpal::default_host();
    match host.input_devices() {
        Ok(devs) => devs
            .filter_map(|d| d.name().ok())
            .filter(|n| !n.trim().is_empty())
            .collect(),
        Err(_) => Vec::new(),
    }
}

/// 枚举输出端（扬声器/系统音频）。
/// core 的 `list_output_devices()` 已按 Output scope 通道数过滤掉麦克风，
/// 这里只做同名去重（虚拟声卡多实例场景）。
fn list_outputs() -> Vec<OutputDevice> {
    let mut raw = system_audio::list_output_devices().unwrap_or_default();
    raw.dedup_by(|a, b| a.name == b.name && a.uid == b.uid);
    raw
}

// ─── 应用状态 ─────────────────────────────────────────────────────────

struct App {
    slots: Vec<ModelSlot>,
    log: Vec<String>,
    last_rms: f32,
    started_at: Instant,
    source_label: String,
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

// ─── 设备选择屏 ───────────────────────────────────────────────────────

/// 上下两区列出输入/输出设备，方向键跨区扁平导航，Enter 选中。
/// 返回 Some(pick) 表示用户已选定；None 表示按 q/Esc 取消（退回模型选择屏）。
fn run_device_screen(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
) -> Result<Option<(DevicePick, Vec<String>, Vec<OutputDevice>)>> {
    // 进入时一次性枚举。失败用空列表，UI 会显示「无可用设备」。
    let inputs = list_inputs();
    let outputs = list_outputs();

    // 扁平游标：把 (区, 索引) 视为一个一维序列。
    // region: 0=输入区，1=输出区。
    let total = inputs.len() + outputs.len();
    let mut cursor: usize = 0;

    // 初始光标落在第一个非空区的第一项，避免停在空区。
    if inputs.is_empty() && !outputs.is_empty() {
        cursor = inputs.len(); // 跳到输出区第一项
    }

    loop {
        terminal.draw(|frame| {
            let area = frame.area();
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(3),
                    Constraint::Min(8),
                    Constraint::Length(3),
                ])
                .split(area);

            let title = Paragraph::new(Line::from(vec![Span::styled(
                " 选择采集源 ",
                Style::default().add_modifier(Modifier::BOLD),
            )]))
            .block(Block::default().borders(Borders::ALL));
            frame.render_widget(title, chunks[0]);

            // 上下两区按内容高度分配。中间区域竖直拆分。
            let input_h = (inputs.len() + 2).max(3) as u16;
            let output_h = (outputs.len() + 2).max(3) as u16;
            let body = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(input_h as u16),
                    Constraint::Min(output_h as u16),
                ])
                .split(chunks[1]);

            // 输入区
            let mut in_lines: Vec<Line> = Vec::new();
            if inputs.is_empty() {
                in_lines.push(Line::from(Span::styled(
                    " （无可用输入设备）",
                    Style::default().fg(Color::DarkGray),
                )));
            }
            for (i, name) in inputs.iter().enumerate() {
                let flat = i;
                let mark = if flat == cursor { "▸" } else { " " };
                let style = if flat == cursor {
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default()
                };
                in_lines.push(Line::from(vec![
                    Span::raw(format!(" {mark} ")),
                    Span::styled(format!("[I{}] ", i + 1), Style::default().fg(Color::Cyan)),
                    Span::styled(name.clone(), style),
                ]));
            }
            let in_para = Paragraph::new(in_lines).block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" 输入端（麦克风） "),
            );
            frame.render_widget(in_para, body[0]);

            // 输出区
            let mut out_lines: Vec<Line> = Vec::new();
            if outputs.is_empty() {
                out_lines.push(Line::from(Span::styled(
                    " （无可用输出设备）",
                    Style::default().fg(Color::DarkGray),
                )));
            }
            for (i, dev) in outputs.iter().enumerate() {
                let flat = inputs.len() + i;
                let mark = if flat == cursor { "▸" } else { " " };
                let style = if flat == cursor {
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default()
                };
                out_lines.push(Line::from(vec![
                    Span::raw(format!(" {mark} ")),
                    Span::styled(format!("[O{}] ", i + 1), Style::default().fg(Color::Cyan)),
                    Span::styled(dev.name.clone(), style),
                ]));
            }
            let out_para = Paragraph::new(out_lines).block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" 输出端（系统音频） "),
            );
            frame.render_widget(out_para, body[1]);

            let hint = if total == 0 {
                Span::styled(
                    " 无可用设备，请检查麦克风/扬声器 · q 返回",
                    Style::default().fg(Color::Red),
                )
            } else {
                Span::styled(
                    " ↑/↓ 移动 · Enter 选中并预览 · q 返回 ",
                    Style::default().fg(Color::DarkGray),
                )
            };
            let footer =
                Paragraph::new(Line::from(hint)).block(Block::default().borders(Borders::ALL));
            frame.render_widget(footer, chunks[2]);
        })?;

        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => return Ok(None),
                    KeyCode::Up => {
                        if cursor > 0 {
                            cursor -= 1;
                        }
                    }
                    KeyCode::Down => {
                        if cursor + 1 < total {
                            cursor += 1;
                        }
                    }
                    KeyCode::Enter | KeyCode::Char('\n') | KeyCode::Char('\r') => {
                        if cursor < inputs.len() {
                            let name = inputs[cursor].clone();
                            return Ok(Some((
                                DevicePick::Input(name),
                                inputs.clone(),
                                outputs.clone(),
                            )));
                        } else if cursor < total {
                            let dev = &outputs[cursor - inputs.len()];
                            let uid = dev.uid.clone();
                            return Ok(Some((
                                DevicePick::Output(uid),
                                inputs.clone(),
                                outputs.clone(),
                            )));
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}

// ─── 预览屏 ───────────────────────────────────────────────────────────

/// 用选中的设备立即启动采集，只算 RMS 不喂 ASR，让用户确认「这个设备有声音进来」。
/// 返回 Ok(Some(capture))：Enter 进入主屏（采集对象移交，不重启）。
/// 返回 Ok(None)：Esc 返回设备选择屏（停止采集）。
fn run_preview_screen(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    pick: &DevicePick,
    inputs: &[String],
    outputs: &[OutputDevice],
) -> Result<Option<Capture>> {
    let label = pick.label(inputs, outputs);
    let mut capture = match Capture::start(pick) {
        Ok(c) => c,
        Err(e) => {
            // 启动失败：短暂展示错误，等用户按键返回。
            show_error_screen(terminal, &label, &format!("{e:#}"))?;
            return Ok(None);
        }
    };
    let mut rms_disp: f32 = 0.0;
    let preview_started_at = Instant::now();
    let mut received_frames = 0usize;

    loop {
        // 消费已有帧算 RMS（不喂 ASR）。
        while let Ok(frame) = capture.rx.try_recv() {
            received_frames += 1;
            rms_disp = rms(&frame).max(rms_disp * 0.9);
        }

        let show_output_tcc_hint = matches!(pick, DevicePick::Output(_))
            && received_frames == 0
            && preview_started_at.elapsed() >= Duration::from_secs(3);

        terminal.draw(|frame| {
            let area = frame.area();
            let block = Block::default().borders(Borders::ALL).title(" 采集预览 ");
            let inner = block.inner(area);
            frame.render_widget(block, area);

            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(7), Constraint::Length(3), Constraint::Min(1)])
                .split(inner);

            let mut info_lines = vec![
                Line::from(vec![
                    Span::styled(" 选中源：", Style::default().fg(Color::Cyan)),
                    Span::raw(label.clone()),
                ]),
                Line::from(""),
                Line::from(Span::styled(
                    " 对着麦克风说话 / 播放声音，观察下方电平是否跳动",
                    Style::default().fg(Color::DarkGray),
                )),
            ];
            if show_output_tcc_hint {
                info_lines.push(Line::from(""));
                info_lines.push(Line::from(Span::styled(
                    " 3 秒内仍未收到任何系统音频帧：若你是从终端 cargo run，macOS 可能在 TCC 层拒绝了 AudioCapture。",
                    Style::default().fg(Color::Yellow),
                )));
                info_lines.push(Line::from(Span::styled(
                    " 常见场景是终端宿主缺少 NSAudioCaptureUsageDescription；请改用带该键的 .app 宿主或先跑 diag_devices 核对。",
                    Style::default().fg(Color::Yellow),
                )));
            }
            frame.render_widget(Paragraph::new(info_lines), chunks[0]);

            // 电平条：用 ASCII block 字符铺 30 格。
            let bars = 30;
            let filled = ((rms_disp * 3.0 * bars as f32).round() as i64).clamp(0, bars as i64) as usize;
            let bar: String = "█".repeat(filled);
            let rest: String = "░".repeat(bars - filled);
            let level = Paragraph::new(Line::from(vec![
                Span::raw(" "),
                Span::styled(bar, Style::default().fg(Color::Green)),
                Span::styled(rest, Style::default().fg(Color::DarkGray)),
                Span::raw(format!("  RMS {:.4}", rms_disp)),
            ]));
            frame.render_widget(level, chunks[1]);

            let hint = Paragraph::new(Line::from(Span::styled(
                " Enter 开始采集并进入对比 · Esc 返回重选 ",
                Style::default().fg(Color::DarkGray),
            )));
            frame.render_widget(hint, chunks[2]);
        })?;

        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Esc | KeyCode::Char('q') => {
                        capture.stop();
                        return Ok(None);
                    }
                    KeyCode::Enter | KeyCode::Char('\n') | KeyCode::Char('\r') => {
                        return Ok(Some(capture));
                    }
                    _ => {}
                }
            }
        }
    }
}

/// 启动采集失败时展示一行错误，按任意键返回。
fn show_error_screen(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    label: &str,
    err: &str,
) -> Result<()> {
    loop {
        terminal.draw(|frame| {
            let para = Paragraph::new(vec![
                Line::from(Span::styled(
                    format!(" 启动采集失败：{label} "),
                    Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
                )),
                Line::from(""),
                Line::from(err.to_string()),
                Line::from(""),
                Line::from(Span::styled(
                    " 按任意键返回重选 ",
                    Style::default().fg(Color::DarkGray),
                )),
            ])
            .block(Block::default().borders(Borders::ALL).title("错误"));
            frame.render_widget(para, frame.area());
        })?;
        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(_) = event::read()? {
                return Ok(());
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

    // 阶段 3 + 4：设备选择 → 预览。两屏之间可往返（预览 Esc 回设备选择）。
    let (mut rx, source_label) = loop {
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
                break (capture.rx, label);
            }
            None => continue, // Esc：回设备选择屏重选
        }
    };

    let mut app = App {
        slots,
        log: Vec::new(),
        last_rms: 0.0,
        started_at: Instant::now(),
        source_label,
    };

    // 阶段 5：主 TUI 循环。
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
        Span::raw(" · 源:"),
        Span::styled(app.source_label.clone(), Style::default().fg(Color::Cyan)),
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
        let para = Paragraph::new(lines).wrap(Wrap { trim: false }).block(
            Block::default().borders(Borders::ALL).title(Span::styled(
                format!(" {}{} ", slot.name, title_suffix),
                title_style,
            )),
        );
        frame.render_widget(para, *col_area);
    }

    let log_lines: Vec<Line> = app
        .log
        .iter()
        .rev()
        .take(5)
        .map(|s| Line::from(s.clone()))
        .collect();
    let log_para = Paragraph::new(log_lines).wrap(Wrap { trim: false }).block(
        Block::default()
            .borders(Borders::ALL)
            .title("日志（最近 endpoint）"),
    );
    frame.render_widget(log_para, chunks[2]);
}
