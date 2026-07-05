//! 媒体播放与字幕：默认 mp3 + srt，用 ffplay 播放，跟踪播放位置与字幕对齐。

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::Instant;

use anyhow::{Context, Result};

use crate::config::{DEFAULT_MEDIA_MP3, DEFAULT_MEDIA_SRT};

#[derive(Clone)]
pub(crate) struct SubtitleCue {
    pub(crate) index: usize,
    pub(crate) start_ms: u64,
    pub(crate) end_ms: u64,
    pub(crate) text: String,
}

pub(crate) struct MediaState {
    pub(crate) audio_path: PathBuf,
    pub(crate) srt_path: PathBuf,
    pub(crate) cues: Vec<SubtitleCue>,
    pub(crate) duration_ms: u64,
    pub(crate) position_ms: u64,
    anchor_ms: u64,
    anchor_started_at: Instant,
    pub(crate) playing: bool,
    child: Option<Child>,
    pub(crate) last_report: Option<PathBuf>,
}

impl MediaState {
    pub(crate) fn load_default() -> Result<Option<Self>> {
        let audio_path = PathBuf::from(DEFAULT_MEDIA_MP3);
        let srt_path = PathBuf::from(DEFAULT_MEDIA_SRT);
        if !audio_path.exists() || !srt_path.exists() {
            return Ok(None);
        }
        let cues = parse_srt(&srt_path)?;
        let duration_ms = cues.last().map(|cue| cue.end_ms).unwrap_or(0);
        Ok(Some(Self {
            audio_path,
            srt_path,
            cues,
            duration_ms,
            position_ms: 0,
            anchor_ms: 0,
            anchor_started_at: Instant::now(),
            playing: false,
            child: None,
            last_report: None,
        }))
    }

    pub(crate) fn start(&mut self) -> Result<()> {
        self.stop_child();
        self.child = Some(spawn_ffplay(&self.audio_path, self.position_ms)?);
        self.resume_clock();
        Ok(())
    }

    pub(crate) fn start_clock_only(&mut self) {
        self.stop_child();
        self.resume_clock();
    }

    fn stop_child(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }

    fn pause(&mut self) {
        self.pause_clock();
    }

    pub(crate) fn toggle_play(&mut self) -> Result<()> {
        if self.playing {
            self.pause();
            Ok(())
        } else {
            self.start()
        }
    }

    pub(crate) fn refresh_position(&mut self) {
        if self.playing {
            self.position_ms = self.anchor_ms + self.anchor_started_at.elapsed().as_millis() as u64;
        }
    }

    pub(crate) fn pause_clock(&mut self) {
        self.refresh_position();
        self.playing = false;
        self.stop_child();
    }

    pub(crate) fn resume_clock(&mut self) {
        self.anchor_ms = self.position_ms;
        self.anchor_started_at = Instant::now();
        self.playing = true;
    }

    pub(crate) fn set_position_ms(&mut self, position_ms: u64) {
        self.position_ms = if self.duration_ms > 0 {
            position_ms.min(self.duration_ms)
        } else {
            position_ms
        };
        self.anchor_ms = self.position_ms;
        self.anchor_started_at = Instant::now();
    }

    pub(crate) fn seek_by(&mut self, delta_ms: i64) -> Result<()> {
        self.refresh_position();
        let next_position = if delta_ms.is_negative() {
            self.position_ms.saturating_sub(delta_ms.unsigned_abs())
        } else {
            self.position_ms.saturating_add(delta_ms as u64)
        };
        self.set_position_ms(next_position);
        if self.playing {
            self.start()?;
        }
        Ok(())
    }

    pub(crate) fn active_index(&self) -> usize {
        if self.cues.is_empty() {
            return 0;
        }
        match self.cues.binary_search_by(|cue| {
            if self.position_ms < cue.start_ms {
                std::cmp::Ordering::Greater
            } else if self.position_ms > cue.end_ms {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Equal
            }
        }) {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1),
        }
    }
}

impl Drop for MediaState {
    fn drop(&mut self) {
        self.stop_child();
    }
}

fn parse_srt(path: &Path) -> Result<Vec<SubtitleCue>> {
    let raw =
        fs::read_to_string(path).with_context(|| format!("读取字幕失败: {}", path.display()))?;
    let normalized = raw.replace("\r\n", "\n");
    let mut cues = Vec::new();

    for block in normalized.split("\n\n") {
        let mut lines = block.lines().filter(|line| !line.trim().is_empty());
        let Some(index_line) = lines.next() else {
            continue;
        };
        let Ok(index) = index_line.trim().parse::<usize>() else {
            continue;
        };
        let Some(time_line) = lines.next() else {
            continue;
        };
        let Some((start, end)) = time_line.split_once("-->") else {
            continue;
        };
        let text = lines.collect::<Vec<_>>().join(" ").trim().to_string();
        if text.is_empty() {
            continue;
        }
        cues.push(SubtitleCue {
            index,
            start_ms: parse_srt_time(start.trim())?,
            end_ms: parse_srt_time(end.trim())?,
            text,
        });
    }

    Ok(cues)
}

fn parse_srt_time(input: &str) -> Result<u64> {
    let Some((hms, millis)) = input.split_once(',') else {
        anyhow::bail!("无效字幕时间: {input}");
    };
    let parts = hms
        .split(':')
        .map(str::parse::<u64>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if parts.len() != 3 {
        anyhow::bail!("无效字幕时间: {input}");
    }
    let millis = millis.trim().parse::<u64>()?;
    Ok(((parts[0] * 3600 + parts[1] * 60 + parts[2]) * 1000) + millis)
}

fn spawn_ffplay(audio_path: &Path, position_ms: u64) -> Result<Child> {
    let seek_seconds = format!("{:.3}", position_ms as f64 / 1000.0);
    Command::new("ffplay")
        .arg("-nodisp")
        .arg("-autoexit")
        .arg("-loglevel")
        .arg("quiet")
        .arg("-ss")
        .arg(seek_seconds)
        .arg(audio_path)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .with_context(|| "启动 ffplay 失败，请确认 ffplay 在 PATH 中")
}

pub(crate) fn format_media_time(ms: u64) -> String {
    let total = ms / 1000;
    let h = total / 3600;
    let m = (total % 3600) / 60;
    let s = total % 60;
    if h > 0 {
        format!("{h:02}:{m:02}:{s:02}")
    } else {
        format!("{m:02}:{s:02}")
    }
}

pub(crate) fn media_progress_bar(position_ms: u64, duration_ms: u64, width: usize) -> String {
    if duration_ms == 0 || width == 0 {
        return " ".repeat(width);
    }
    let filled = ((position_ms.min(duration_ms) as f64 / duration_ms as f64) * width as f64)
        .round()
        .clamp(0.0, width as f64) as usize;
    format!("{}{}", "█".repeat(filled), "░".repeat(width - filled))
}
