//! 启动流程的各个交互屏：模型选择 → 设备选择 → 采集预览 → 错误提示。

use std::io;
use std::time::{Duration, Instant};

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Terminal;

use arcvoice_core::asr::Language;
use asr_compare_tui::system_audio::OutputDevice;

use crate::capture::{list_inputs, list_outputs, list_sample_files, Capture, DevicePick};
use crate::models::{SlotKind, MODEL_DESCS};
use crate::segments::{SegConfig, SEG_CONFIG_ENHANCED_ZIPFORMER_EN};
use crate::util::rms;

// ─── 启动选择界面 ─────────────────────────────────────────────────────

pub(crate) fn run_selection_screen(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
) -> Result<Option<(Vec<usize>, Option<SegConfig>)>> {
    // 每个模型是否就绪：流式模型恒 true，离线模型检测文件是否齐全。
    // 不就绪的模型在选择屏打灰、不可勾选（优雅降级）。
    let available: Vec<bool> = MODEL_DESCS.iter().map(|d| d.files_present()).collect();
    let mut selected: Vec<bool> = MODEL_DESCS
        .iter()
        .enumerate()
        .map(|(i, desc)| available[i] && matches!(desc.kind, SlotKind::Offline(_)))
        .collect();
    if !selected.iter().any(|&s| s) {
        selected = available.clone();
    }

    // 配置选择：默认 = None（Sherpa 内置端点），增强 = 加强版Zipformer-en
    let has_any_online = MODEL_DESCS.iter().any(|d| matches!(d.kind, SlotKind::Online(_)));
    let mut use_enhanced_config = false;
    /// 配置面板固定行数（标题+border 各 1 行，内容 6 行）
    const CONFIG_PANEL_LINES: u16 = 8;

    loop {
        let config_panel_height = if has_any_online { CONFIG_PANEL_LINES } else { 0 };
        terminal.draw(|frame| {
            let area = frame.area();
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(3),
                    Constraint::Min(6 + config_panel_height),
                    Constraint::Length(3),
                ])
                .split(area);

            let title = Paragraph::new(Line::from(vec![Span::styled(
                " ASR Compare — 选择要加载的引擎 ",
                Style::default().add_modifier(Modifier::BOLD),
            )]))
            .block(Block::default().borders(Borders::ALL));
            frame.render_widget(title, chunks[0]);

            // 中部：模型列表（上）+ 配置面板（下，仅当有流式模型时显示）
            let body_chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(6), Constraint::Length(config_panel_height)])
                .split(chunks[1]);

            // ── 模型列表 ──
            let mut lines: Vec<Line> = Vec::new();
            for (i, desc) in MODEL_DESCS.iter().enumerate() {
                let ok = available[i];
                let check = if !ok {
                    "    " // 缺文件：不可勾选
                } else if selected[i] {
                    "[x] "
                } else {
                    "[ ] "
                };
                let lang = match desc.language {
                    Language::Chinese => "中文",
                    Language::English => "英文",
                };
                let kind_tag = match desc.kind {
                    SlotKind::Online(_) => "流式",
                    SlotKind::Offline(_) => "离线+VAD",
                };
                let style = if !ok {
                    Style::default().fg(Color::DarkGray)
                } else if selected[i] {
                    Style::default().fg(Color::Green)
                } else {
                    Style::default().fg(Color::Gray)
                };
                let key_span = if ok {
                    Span::styled(format!(" {} ", i + 1), Style::default().fg(Color::Yellow))
                } else {
                    Span::raw(format!(" {} ", i + 1))
                };
                let suffix = if !ok {
                    desc.missing_files_hint()
                        .map(|script| format!("  (未下载: {script})"))
                        .unwrap_or_default()
                } else {
                    String::new()
                };
                lines.push(Line::from(vec![
                    key_span,
                    Span::styled(
                        format!("{check}{} ({lang}/{kind_tag}){}", desc.name, suffix),
                        style,
                    ),
                ]));
            }
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                " 1-9 勾选模型 · e 切换切句配置 · Enter 确认 · q 退出 ",
                Style::default().fg(Color::DarkGray),
            )));
            let list = Paragraph::new(lines)
                .block(Block::default().borders(Borders::ALL).title("选择引擎"));
            frame.render_widget(list, body_chunks[0]);

            // ── 配置面板 ──
            if has_any_online {
                let selected_style = Style::default().fg(Color::Green).add_modifier(Modifier::BOLD);
                let unselected_style = Style::default().fg(Color::Gray);
                let (default_style, enhanced_style) = if use_enhanced_config {
                    (unselected_style, selected_style)
                } else {
                    (selected_style, unselected_style)
                };
                let (default_mark, enhanced_mark) = if use_enhanced_config {
                    ("[ ]", "[x]")
                } else {
                    ("[x]", "[ ]")
                };
                let config_lines = vec![
                    Line::from(vec![
                        Span::styled(
                            format!(" {default_mark} 默认（Sherpa 内置端点）"),
                            default_style,
                        ),
                    ]),
                    Line::from(vec![
                        Span::styled(
                            format!(" {enhanced_mark} 加强版Zipformer-en"),
                            enhanced_style,
                        ),
                    ]),
                    Line::from(""),
                    Line::from(Span::styled(
                        if use_enhanced_config {
                            "    标点稳定700ms · 冷却1.5s · 兜底10s · 强切25s · VAD收尾300ms"
                        } else {
                            "    Sherpa rule1=2.4s rule2=1.2s rule3=20s"
                        },
                        Style::default().fg(Color::DarkGray),
                    )),
                    Line::from(Span::styled(
                        "    按 e 切换（仅对流式模型 Zipformer/Nemotron 生效）",
                        Style::default().fg(Color::DarkGray),
                    )),
                ];
                let config_para = Paragraph::new(config_lines)
                    .block(Block::default().borders(Borders::ALL).title(" 切句配置 "));
                frame.render_widget(config_para, body_chunks[1]);
            }

            let count = selected.iter().filter(|&&s| s).count();
            let config_tag = if use_enhanced_config && has_any_online {
                " · 加强版Zipformer-en"
            } else {
                ""
            };
            let hint = if count == 0 {
                " 请至少选择一个引擎".to_string()
            } else {
                format!(" 将加载 {count} 个引擎{config_tag}（模型较大，加载需几秒）")
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
                    KeyCode::Char('e') if has_any_online => {
                        use_enhanced_config = !use_enhanced_config;
                    }
                    KeyCode::Char(c @ '1'..='9') => {
                        let idx = (c as u8 - b'1') as usize;
                        if idx < selected.len() && available[idx] {
                            selected[idx] = !selected[idx];
                        }
                    }
                    KeyCode::Enter | KeyCode::Char('\n') | KeyCode::Char('\r') => {
                        let indices: Vec<usize> = selected
                            .iter()
                            .enumerate()
                            .filter(|(_, &s)| s)
                            .map(|(i, _)| i)
                            .collect();
                        if !indices.is_empty() {
                            let config = if use_enhanced_config {
                                Some(SEG_CONFIG_ENHANCED_ZIPFORMER_EN.clone())
                            } else {
                                None
                            };
                            return Ok(Some((indices, config)));
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
pub(crate) fn run_device_screen(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
) -> Result<Option<(DevicePick, Vec<String>, Vec<OutputDevice>)>> {
    // 进入时一次性枚举。失败用空列表，UI 会显示「无可用设备」。
    let inputs = list_inputs();
    let outputs = list_outputs();
    let sample_files = list_sample_files();

    // 扁平游标：把 (区, 索引) 视为一个一维序列。
    // region: 0=输入区，1=输出区，2=示例文件区。
    let total = inputs.len() + outputs.len() + sample_files.len();
    let mut cursor: usize = 0;

    // 初始光标落在第一个非空区的第一项，避免停在空区。
    if inputs.is_empty() && !outputs.is_empty() {
        cursor = inputs.len(); // 跳到输出区第一项
    } else if inputs.is_empty() && outputs.is_empty() && !sample_files.is_empty() {
        cursor = inputs.len() + outputs.len(); // 跳到示例文件第一项
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
            let sample_h = (sample_files.len() + 2).max(3) as u16;
            let body = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(input_h as u16),
                    Constraint::Length(output_h as u16),
                    Constraint::Min(sample_h as u16),
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

            // 示例文件区
            let mut sample_lines: Vec<Line> = Vec::new();
            if sample_files.is_empty() {
                sample_lines.push(Line::from(Span::styled(
                    " （未找到默认示例 mp3）",
                    Style::default().fg(Color::DarkGray),
                )));
            }
            for (i, path) in sample_files.iter().enumerate() {
                let flat = inputs.len() + outputs.len() + i;
                let mark = if flat == cursor { "▸" } else { " " };
                let style = if flat == cursor {
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default()
                };
                let name = path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("<unknown>");
                sample_lines.push(Line::from(vec![
                    Span::raw(format!(" {mark} ")),
                    Span::styled(format!("[F{}] ", i + 1), Style::default().fg(Color::Cyan)),
                    Span::styled(name.to_string(), style),
                    Span::styled("  直接送入 ASR", Style::default().fg(Color::DarkGray)),
                ]));
            }
            let sample_para = Paragraph::new(sample_lines).block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" 示例音频文件（绕过系统音频采集） "),
            );
            frame.render_widget(sample_para, body[2]);

            let hint = if total == 0 {
                Span::styled(
                    " 无可用采集源，请检查麦克风/扬声器或默认 mp3 · q 返回",
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
                        } else if cursor < inputs.len() + outputs.len() {
                            let dev = &outputs[cursor - inputs.len()];
                            let uid = dev.uid.clone();
                            return Ok(Some((
                                DevicePick::Output(uid),
                                inputs.clone(),
                                outputs.clone(),
                            )));
                        } else if cursor < total {
                            let path = sample_files[cursor - inputs.len() - outputs.len()].clone();
                            return Ok(Some((
                                DevicePick::SampleFile(path),
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
pub(crate) fn run_preview_screen(
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
                Line::from(Span::styled(preview_hint_for_pick(pick), Style::default().fg(Color::DarkGray))),
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
                        if matches!(pick, DevicePick::SampleFile(_)) {
                            capture.stop();
                            return match Capture::start(pick) {
                                Ok(fresh_capture) => Ok(Some(fresh_capture)),
                                Err(e) => {
                                    show_error_screen(terminal, &label, &format!("{e:#}"))?;
                                    Ok(None)
                                }
                            };
                        }
                        return Ok(Some(capture));
                    }
                    _ => {}
                }
            }
        }
    }
}

fn preview_hint_for_pick(pick: &DevicePick) -> &'static str {
    match pick {
        DevicePick::SampleFile(_) => " 正在把示例 mp3 解码为 16kHz mono PCM，并按实时速度送入 ASR",
        DevicePick::Input(_) | DevicePick::Output(_) => {
            " 对着麦克风说话 / 播放声音，观察下方电平是否跳动"
        }
    }
}

/// 启动采集失败时展示一行错误，按任意键返回。
pub(crate) fn show_error_screen(
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
