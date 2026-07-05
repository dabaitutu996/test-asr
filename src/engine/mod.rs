//! ASR 引擎槽位：流式（online）与离线（offline）两类，统一到 `AnySlot`。
//!
//! `SlotView` trait 给 `AnySlot` 的字段访问提供统一接口，渲染 / 报告 / toggle 共用。

pub(crate) mod offline;
pub(crate) mod online;

use anyhow::Result;

use crate::models::{ModelDesc, SlotKind};

pub(crate) use offline::{build_offline_slot, decode_offline_segment, OfflineSlot};
pub(crate) use online::{build_online_slot, feed_online_frame, OnlineSlot};

pub(crate) enum AnySlot {
    Online(OnlineSlot),
    Offline(OfflineSlot),
}

impl AnySlot {
    pub(crate) fn name(&self) -> &str {
        match self {
            AnySlot::Online(s) => s.name,
            AnySlot::Offline(s) => s.name,
        }
    }

    pub(crate) fn is_online(&self) -> bool {
        matches!(self, AnySlot::Online(_))
    }

    pub(crate) fn build(desc: &ModelDesc) -> Result<Self> {
        match desc.kind {
            SlotKind::Online(_) => Ok(Self::Online(build_online_slot(desc)?)),
            SlotKind::Offline(_) => Ok(Self::Offline(build_offline_slot(desc)?)),
        }
    }
}

/// 给 AnySlot 的字段访问提供统一接口（渲染/报告/toggle 共用）。
pub(crate) trait SlotView {
    fn partial(&self) -> &str;
    fn finals(&self) -> &[String];
    fn all_finals(&self) -> &[String];
    fn enabled(&self) -> bool;
    fn finals_scroll(&self) -> u16;
    fn finals_scroll_mut(&mut self) -> &mut u16;
    fn set_enabled(&mut self, enabled: bool);
    fn clear(&mut self);
}

impl SlotView for OnlineSlot {
    fn partial(&self) -> &str {
        &self.partial
    }
    fn finals(&self) -> &[String] {
        &self.finals
    }
    fn all_finals(&self) -> &[String] {
        &self.all_finals
    }
    fn enabled(&self) -> bool {
        self.enabled
    }
    fn finals_scroll(&self) -> u16 {
        self.finals_scroll
    }
    fn finals_scroll_mut(&mut self) -> &mut u16 {
        &mut self.finals_scroll
    }
    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if !enabled {
            self.partial.clear();
        }
    }
    fn clear(&mut self) {
        self.partial.clear();
        self.all_partials.clear();
        self.finals.clear();
        self.all_finals.clear();
        self.finals_scroll = 0;
    }
}

impl SlotView for OfflineSlot {
    fn partial(&self) -> &str {
        &self.partial
    }
    fn finals(&self) -> &[String] {
        &self.finals
    }
    fn all_finals(&self) -> &[String] {
        &self.all_finals
    }
    fn enabled(&self) -> bool {
        self.enabled
    }
    fn finals_scroll(&self) -> u16 {
        self.finals_scroll
    }
    fn finals_scroll_mut(&mut self) -> &mut u16 {
        &mut self.finals_scroll
    }
    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if !enabled {
            self.partial.clear();
        }
    }
    fn clear(&mut self) {
        self.partial.clear();
        self.finals.clear();
        self.all_finals.clear();
        self.finals_scroll = 0;
        self.segments_decoded = 0;
        self.last_segment_samples = 0;
    }
}

impl SlotView for AnySlot {
    fn partial(&self) -> &str {
        match self {
            AnySlot::Online(s) => s.partial(),
            AnySlot::Offline(s) => s.partial(),
        }
    }
    fn finals(&self) -> &[String] {
        match self {
            AnySlot::Online(s) => s.finals(),
            AnySlot::Offline(s) => s.finals(),
        }
    }
    fn all_finals(&self) -> &[String] {
        match self {
            AnySlot::Online(s) => s.all_finals(),
            AnySlot::Offline(s) => s.all_finals(),
        }
    }
    fn enabled(&self) -> bool {
        match self {
            AnySlot::Online(s) => s.enabled(),
            AnySlot::Offline(s) => s.enabled(),
        }
    }
    fn finals_scroll(&self) -> u16 {
        match self {
            AnySlot::Online(s) => s.finals_scroll(),
            AnySlot::Offline(s) => s.finals_scroll(),
        }
    }
    fn finals_scroll_mut(&mut self) -> &mut u16 {
        match self {
            AnySlot::Online(s) => s.finals_scroll_mut(),
            AnySlot::Offline(s) => s.finals_scroll_mut(),
        }
    }
    fn set_enabled(&mut self, enabled: bool) {
        match self {
            AnySlot::Online(s) => s.set_enabled(enabled),
            AnySlot::Offline(s) => s.set_enabled(enabled),
        }
    }
    fn clear(&mut self) {
        match self {
            AnySlot::Online(s) => s.clear(),
            AnySlot::Offline(s) => s.clear(),
        }
    }
}
