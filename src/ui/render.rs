//! 主 TUI 画面渲染：顶部标题栏 + 各引擎列 + 底部日志，以及左侧媒体/字幕面板。

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

use crate::app::App;
use crate::engine::SlotView;
use crate::media::{format_media_time, media_progress_bar, MediaState};

pub(crate) fn draw(app: &App, frame: &mut ratatui::Frame) {
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
        Span::styled("←/→", Style::default().fg(Color::Yellow)),
        Span::raw(" 选列 · "),
        Span::styled("↑/↓", Style::default().fg(Color::Yellow)),
        Span::raw(" 滚动 · "),
        Span::styled("PgUp/PgDn", Style::default().fg(Color::Yellow)),
        Span::raw(" 快速滚动 · "),
        Span::styled("q", Style::default().fg(Color::Yellow)),
        Span::raw(" 退出 · "),
        Span::styled("c", Style::default().fg(Color::Yellow)),
        Span::raw(" 清空 · "),
        Span::styled("1-9", Style::default().fg(Color::Yellow)),
        Span::raw(" 切换引擎 · "),
        Span::raw(format!("音频 RMS {:.4}", app.last_rms)),
        Span::raw(" · 源:"),
        Span::styled(app.source_label.clone(), Style::default().fg(Color::Cyan)),
    ]);
    let top = Paragraph::new(title).block(Block::default().borders(Borders::ALL).title("实时对比"));
    frame.render_widget(top, chunks[0]);

    let asr_area = if let Some(media) = &app.media {
        let body = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(34), Constraint::Percentage(66)])
            .split(chunks[1]);
        render_media_panel(media, frame, body[0], app.sample_file_mode);
        body[1]
    } else {
        chunks[1]
    };

    let count = app.slots.len().max(1);
    let col_constraints: Vec<Constraint> = (0..count)
        .map(|_| Constraint::Ratio(1, count as u32))
        .collect();
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints(col_constraints)
        .split(asr_area);

    for (index, (slot, col_area)) in app.slots.iter().zip(columns.iter()).enumerate() {
        let is_active = index == app.active_slot;
        let dim = !slot.enabled();
        let is_offline = !slot.is_online();
        let text_style = if dim {
            Style::default().fg(Color::DarkGray)
        } else {
            Style::default()
        };

        // partial 区：流式槽显示实时 partial；离线槽显示 VAD 等待状态。
        let partial_label: String = if dim {
            "[已关闭]".to_string()
        } else if is_offline {
            app.vad.status_label()
        } else {
            slot.partial().to_string()
        };
        let partial_caption = if is_offline && !dim {
            "vad: "
        } else {
            "partial: "
        };
        let partial_lines = vec![Line::from(vec![
            Span::styled(partial_caption, Style::default().fg(Color::Cyan)),
            Span::styled(partial_label, text_style),
        ])];
        let title_suffix = if dim { " [关]" } else { "" };
        let title_style = if dim {
            Style::default().fg(Color::DarkGray)
        } else if is_active {
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().add_modifier(Modifier::BOLD)
        };
        let col_layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(5), Constraint::Min(5)])
            .split(*col_area);
        let partial_para = Paragraph::new(partial_lines)
            .wrap(Wrap { trim: false })
            .block(Block::default().borders(Borders::ALL).title(Span::styled(
                format!(" {}{} ", slot.name(), title_suffix),
                title_style,
            )));
        frame.render_widget(partial_para, col_layout[0]);

        let mut finals_lines: Vec<Line> = Vec::new();
        for (idx, f) in slot.finals().iter().enumerate() {
            if idx > 0 {
                finals_lines.push(Line::from(""));
            }
            finals_lines.push(Line::from(Span::styled(f.clone(), text_style)));
        }
        if finals_lines.is_empty() {
            finals_lines.push(Line::from(Span::styled(
                "（暂无 final）",
                Style::default().fg(Color::DarkGray),
            )));
        }
        let finals_title = if is_active {
            " finals (active) "
        } else {
            " finals "
        };
        let finals_para = Paragraph::new(finals_lines)
            .wrap(Wrap { trim: false })
            .scroll((slot.finals_scroll(), 0))
            .block(Block::default().borders(Borders::ALL).title(finals_title));
        frame.render_widget(finals_para, col_layout[1]);
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

fn render_media_panel(
    media: &MediaState,
    frame: &mut ratatui::Frame,
    area: Rect,
    sample_file_mode: bool,
) {
    let active = media.active_index();
    let title = if sample_file_mode && media.playing {
        " subtitles / sample file running "
    } else if sample_file_mode {
        " subtitles / sample file paused "
    } else if media.playing {
        " subtitles playing "
    } else {
        " subtitles paused "
    };
    let mut lines = vec![
        Line::from(vec![
            Span::styled("time: ", Style::default().fg(Color::Cyan)),
            Span::styled(
                format!(
                    "{} / {}",
                    format_media_time(media.position_ms),
                    format_media_time(media.duration_ms)
                ),
                Style::default().fg(Color::Yellow),
            ),
        ]),
        Line::from(Span::styled(
            media_progress_bar(media.position_ms, media.duration_ms, 28),
            Style::default().fg(Color::Green),
        )),
        Line::from(vec![
            Span::styled("seek: ", Style::default().fg(Color::Cyan)),
            Span::raw(if sample_file_mode {
                "h/l 10s · H/L 60s · space pause sample input · e report"
            } else {
                "h/l 10s · H/L 60s · space pause audio · e report"
            }),
        ]),
        Line::from(""),
    ];

    if media.cues.is_empty() {
        lines.push(Line::from(Span::styled(
            "未解析到字幕",
            Style::default().fg(Color::Red),
        )));
    } else {
        let start = active.saturating_sub(8);
        let end = (active + 10).min(media.cues.len());
        for cue in &media.cues[start..end] {
            let is_active = media.position_ms >= cue.start_ms && media.position_ms <= cue.end_ms;
            let style = if is_active {
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            let marker = if is_active { ">" } else { " " };
            lines.push(Line::from(Span::styled(
                format!(
                    "{marker} {} {}",
                    format_media_time(cue.start_ms),
                    cue.text.replace('\n', " ")
                ),
                style,
            )));
        }
    }

    if let Some(path) = &media.last_report {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            format!("report: {}", path.display()),
            Style::default().fg(Color::Green),
        )));
    }

    let panel = Paragraph::new(lines)
        .wrap(Wrap { trim: false })
        .block(Block::default().borders(Borders::ALL).title(title));
    frame.render_widget(panel, area);
}
