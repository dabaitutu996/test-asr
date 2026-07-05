//! 离线槽（OfflineRecognizer：离线模型 + 共享 VAD）。
//!
//! 离线模型没有 partial / endpoint 概念。主循环把同一个 VAD segment 广播给
//! 所有启用的离线槽，保证 Canary / Parakeet 在完全相同的音频片段上对比。

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use sherpa_onnx::{OfflineRecognizer, OfflineRecognizerConfig};

use crate::config::{offline_models_dir, FINALS_RETAIN, VAD_SAMPLE_RATE};
use crate::models::{ModelDesc, OfflineFamily, SlotKind};
use crate::util::with_stdout_suppressed;
use crate::vad::VadSegment;

pub(crate) struct OfflineSlot {
    pub(crate) name: &'static str,
    pub(crate) family: OfflineFamily,
    pub(crate) recognizer: OfflineRecognizer,
    pub(crate) partial: String, // 离线模型恒为空，UI 显示"等待 VAD 触发"
    pub(crate) finals: Vec<String>,
    pub(crate) all_finals: Vec<String>,
    pub(crate) enabled: bool,
    pub(crate) finals_scroll: u16,
    pub(crate) segments_decoded: usize,
    pub(crate) last_segment_samples: usize,
}

pub(crate) fn build_offline_slot(desc: &ModelDesc) -> Result<OfflineSlot> {
    let family = match desc.kind {
        SlotKind::Offline(family) => family,
        SlotKind::Online(_) => bail!("build_offline_slot 收到流式模型"),
    };
    let mut cfg = OfflineRecognizerConfig::default();

    match family {
        OfflineFamily::Canary => {
            let encoder = require_offline_file(desc, family, "encoder", "encoder.int8.onnx")?;
            let decoder = require_offline_file(desc, family, "decoder", "decoder.int8.onnx")?;
            let tokens = require_offline_file(desc, family, "tokens", "tokens.txt")?;

            // Canary 用 128 维 mel filterbank（sherpa-onnx 默认 80，必须显式设置）。
            cfg.feat_config.feature_dim = 128;
            cfg.model_config.canary.encoder = Some(encoder.to_string_lossy().into_owned());
            cfg.model_config.canary.decoder = Some(decoder.to_string_lossy().into_owned());
            cfg.model_config.canary.src_lang = Some("en".into());
            cfg.model_config.canary.tgt_lang = Some("en".into());
            cfg.model_config.canary.use_pnc = true; // 标点 + 大小写
            cfg.model_config.tokens = Some(tokens.to_string_lossy().into_owned());
        }
        OfflineFamily::ParakeetNemoCtc => {
            let model = require_offline_file(desc, family, "model", "model.int8.onnx")?;
            let tokens = require_offline_file(desc, family, "tokens", "tokens.txt")?;

            cfg.model_config.nemo_ctc.model = Some(model.to_string_lossy().into_owned());
            cfg.model_config.tokens = Some(tokens.to_string_lossy().into_owned());
        }
    }

    cfg.model_config.num_threads = 1;
    cfg.model_config.provider = Some("cpu".into());
    cfg.model_config.debug = false;

    let recognizer = with_stdout_suppressed(|| {
        OfflineRecognizer::create(&cfg)
            .with_context(|| format!("创建 {} OfflineRecognizer 失败", desc.name))
    })?;

    Ok(OfflineSlot {
        name: desc.name,
        family,
        recognizer,
        partial: String::new(),
        finals: Vec::new(),
        all_finals: Vec::new(),
        enabled: true,
        finals_scroll: 0,
        segments_decoded: 0,
        last_segment_samples: 0,
    })
}

fn require_offline_file(
    desc: &ModelDesc,
    family: OfflineFamily,
    label: &str,
    filename: &str,
) -> Result<PathBuf> {
    let path = offline_models_dir().join(desc.subdir).join(filename);
    if path.exists() {
        return Ok(path);
    }
    bail!(
        "{} 模型文件缺失: {} ({})\n请先运行 {} 下载。",
        desc.name,
        label,
        path.display(),
        family.download_script()
    );
}

/// 把一段已经 VAD 切好的音频喂给离线模型解码，返回识别文本。
pub(crate) fn decode_offline_segment(
    slot: &mut OfflineSlot,
    segment: &VadSegment,
) -> Option<String> {
    if !slot.enabled || segment.samples.is_empty() {
        slot.last_segment_samples = 0;
        return None;
    }
    slot.segments_decoded += 1;
    slot.last_segment_samples = segment.samples.len();

    let stream = slot.recognizer.create_stream();
    stream.accept_waveform(VAD_SAMPLE_RATE as i32, &segment.samples);
    slot.recognizer.decode(&stream);
    let text = stream
        .get_result()
        .map(|r| r.text)
        .unwrap_or_default()
        .trim()
        .to_string();
    if text.is_empty() {
        return None;
    }
    slot.finals.push(text.clone());
    slot.all_finals.push(text.clone());
    if slot.finals.len() > FINALS_RETAIN {
        slot.finals.remove(0);
    }
    Some(text)
}
