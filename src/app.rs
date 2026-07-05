//! 应用运行态：持有所有槽位、共享 VAD、日志、媒体状态，以及键盘交互。

use std::time::Instant;

use crossterm::event::{KeyCode, KeyEvent};

use crate::engine::{AnySlot, SlotView};
use crate::media::MediaState;
use crate::report::export_report;
use crate::util::push_log;
use crate::vad::VadState;

pub(crate) struct App {
    pub(crate) slots: Vec<AnySlot>,
    /// 离线槽共享的 VAD 状态机。喂音时 Online 槽增量吃帧，离线槽等 VAD 触发。
    pub(crate) vad: VadState,
    pub(crate) log: Vec<String>,
    pub(crate) last_rms: f32,
    pub(crate) started_at: Instant,
    pub(crate) source_label: String,
    pub(crate) active_slot: usize,
    pub(crate) media: Option<MediaState>,
}

impl App {
    pub(crate) fn clear(&mut self) {
        for slot in &mut self.slots {
            slot.clear();
        }
        self.vad.reset();
        self.log.clear();
    }

    pub(crate) fn toggle(&mut self, index: usize) {
        if let Some(slot) = self.slots.get_mut(index) {
            let new_enabled = !slot.enabled();
            slot.set_enabled(new_enabled);
            let name = slot.name().to_string();
            let state = if new_enabled { "启用" } else { "禁用" };
            push_log(&mut self.log, format!("[{name}] 已{state}"));
        }
    }

    fn active_slot_mut(&mut self) -> Option<&mut AnySlot> {
        self.slots.get_mut(self.active_slot)
    }

    pub(crate) fn move_active_slot(&mut self, delta: isize) {
        if self.slots.is_empty() {
            self.active_slot = 0;
            return;
        }
        let last = self.slots.len().saturating_sub(1) as isize;
        let next = (self.active_slot as isize + delta).clamp(0, last);
        self.active_slot = next as usize;
    }

    pub(crate) fn scroll_active_slot(&mut self, delta: i16) {
        if let Some(slot) = self.active_slot_mut() {
            let s = slot.finals_scroll_mut();
            if delta.is_negative() {
                *s = s.saturating_sub(delta.unsigned_abs());
            } else {
                *s = s.saturating_add(delta as u16);
            }
        }
    }

    pub(crate) fn reset_active_scroll(&mut self) {
        if let Some(slot) = self.active_slot_mut() {
            *slot.finals_scroll_mut() = 0;
        }
    }

    pub(crate) fn refresh_media(&mut self) {
        if let Some(media) = &mut self.media {
            media.refresh_position();
        }
    }

    /// 处理媒体相关按键。所有失败都记入日志而非上抛——播放/导出出错
    /// 不应掀翻整个 TUI（与主循环启动播放时的 catch+log 策略一致）。
    pub(crate) fn handle_media_key(&mut self, key: &KeyEvent) {
        match key.code {
            KeyCode::Char(' ') => {
                if let Some(media) = &mut self.media {
                    if let Err(e) = media.toggle_play() {
                        push_log(&mut self.log, format!("播放切换失败: {e:#}"));
                    }
                }
            }
            KeyCode::Char('h') => self.media_seek(-10_000),
            KeyCode::Char('l') => self.media_seek(10_000),
            KeyCode::Char('H') => self.media_seek(-60_000),
            KeyCode::Char('L') => self.media_seek(60_000),
            KeyCode::Char('e') => match export_report(self) {
                Ok(path) => {
                    if let Some(media) = &mut self.media {
                        media.last_report = Some(path.clone());
                    }
                    push_log(&mut self.log, format!("报告已导出: {}", path.display()));
                }
                Err(e) => push_log(&mut self.log, format!("报告导出失败: {e:#}")),
            },
            _ => {}
        }
    }

    fn media_seek(&mut self, delta_ms: i64) {
        if let Some(media) = &mut self.media {
            if let Err(e) = media.seek_by(delta_ms) {
                push_log(&mut self.log, format!("跳转失败: {e:#}"));
            }
        }
    }
}
