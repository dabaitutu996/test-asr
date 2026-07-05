//! Markdown 报告导出：汇总 source / runtime / 字幕上下文 / VAD / 各引擎结果。

use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};

use crate::app::App;
use crate::config::VAD_SAMPLE_RATE;
use crate::engine::{AnySlot, SlotView};
use crate::media::format_media_time;
use crate::util::humantime_elapsed;

pub(crate) fn export_report(app: &App) -> Result<PathBuf> {
    fs::create_dir_all("reports").context("创建 reports 目录失败")?;
    let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let path = PathBuf::from(format!("reports/asr_report_{now}.md"));
    let mut out = String::new();

    out.push_str("# ASR Compare Report\n\n");
    out.push_str(&format!("- source: {}\n", app.source_label));
    out.push_str(&format!(
        "- runtime: {}\n",
        humantime_elapsed(app.started_at)
    ));

    if let Some(media) = &app.media {
        out.push_str(&format!("- audio: {}\n", media.audio_path.display()));
        out.push_str(&format!("- subtitle: {}\n", media.srt_path.display()));
        out.push_str(&format!(
            "- position: {} ({:.3}s)\n",
            format_media_time(media.position_ms),
            media.position_ms as f64 / 1000.0
        ));
        out.push_str("\n## Subtitle Context\n\n");
        if media.cues.is_empty() {
            out.push_str("<no subtitles parsed>\n");
        } else {
            let active = media.active_index();
            let start = active.saturating_sub(3);
            let end = (active + 4).min(media.cues.len());
            for cue in &media.cues[start..end] {
                let marker = if cue.index == media.cues[active].index {
                    ">"
                } else {
                    "-"
                };
                out.push_str(&format!(
                    "{marker} [{} - {}] {}\n",
                    format_media_time(cue.start_ms),
                    format_media_time(cue.end_ms),
                    cue.text
                ));
            }
        }
    }

    app.vad.append_report(&mut out);

    // ── Segmenter 统计 ──
    for slot in &app.slots {
        if let AnySlot::Online(s) = slot {
            if let Some(ref seg) = s.segmenter {
                out.push_str("\n## Segmentation Stats\n\n");
                out.push_str(&format!("- config: {}\n", seg.config.name));
                out.push_str(&format!(
                    "- punct_candidate_count: {}\n",
                    seg.stats.punct_candidate_count
                ));
                out.push_str(&format!(
                    "- punct_success_count: {}\n",
                    seg.stats.punct_success_count
                ));
                out.push_str(&format!(
                    "- punct_regret_count: {}\n",
                    seg.stats.punct_regret_count
                ));
                out.push_str(&format!(
                    "- no_punct_fallback_count: {}\n",
                    seg.stats.no_punct_fallback_count
                ));
                out.push_str(&format!(
                    "- extreme_cut_count: {}\n",
                    seg.stats.extreme_cut_count
                ));
                out.push_str(&format!(
                    "- avg_cut_latency_ms: {:.1}\n",
                    seg.stats.avg_cut_latency_ms()
                ));
                out.push_str(&format!(
                    "- avg_words_per_segment: {:.1}\n",
                    seg.stats.avg_words_per_segment()
                ));
                out.push_str(&format!(
                    "- repeat_risk_count: {}\n",
                    seg.stats.repeat_risk_count
                ));
                out.push_str(&format!(
                    "- missing_word_risk_count: {}\n",
                    seg.stats.missing_word_risk_count
                ));
                out.push_str(&format!(
                    "- total_segment_count: {}\n",
                    seg.stats.total_segment_count
                ));
                out.push('\n');
                break; // 目前只支持一个 segmenter
            }
        }
    }

    out.push_str("\n## ASR Results\n");
    for slot in &app.slots {
        out.push_str(&format!("\n### {}\n\n", slot.name()));
        if !slot.partial().trim().is_empty() {
            out.push_str(&format!("partial: {}\n\n", slot.partial().trim()));
        }
        if let AnySlot::Online(s) = slot {
            if !s.all_partials.is_empty() {
                out.push_str(&format!(
                    "partial_history_count: {}\n\n",
                    s.all_partials.len()
                ));
                out.push_str("partial_history:\n\n");
                for (idx, partial) in s.all_partials.iter().enumerate() {
                    out.push_str(&format!("{}. {}\n", idx + 1, partial.trim()));
                }
                out.push('\n');
            }
        }
        if let AnySlot::Offline(s) = slot {
            out.push_str(&format!("- family: {}\n", s.family.as_str()));
            out.push_str(&format!("- decoded_segments: {}\n", s.segments_decoded));
            if s.last_segment_samples > 0 {
                out.push_str(&format!(
                    "- last_decoded_segment_sec: {:.3}\n",
                    s.last_segment_samples as f32 / VAD_SAMPLE_RATE as f32
                ));
            }
            out.push('\n');
        }
        if slot.all_finals().is_empty() {
            out.push_str("finals: <none>\n");
        } else {
            out.push_str(&format!("final_count: {}\n\n", slot.all_finals().len()));
            for (idx, final_text) in slot.all_finals().iter().enumerate() {
                out.push_str(&format!("{}. {}\n", idx + 1, final_text.trim()));
            }
        }
    }

    fs::write(&path, out).with_context(|| format!("写入报告失败: {}", path.display()))?;
    Ok(path)
}
