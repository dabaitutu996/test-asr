//! 流式槽（OnlineRecognizer：Zipformer / Nemotron）：真流式 transducer，增量吃帧。

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use sherpa_onnx::{OnlineRecognizer, OnlineRecognizerConfig, OnlineStream};

use arcvoice_core::asr::streaming::SherpaModelType;

use crate::config::{
    endpoint_rules_from_env, engine_models_dir, hotwords_from_env, FINALS_RETAIN,
    PARTIAL_HISTORY_RETAIN, VAD_SAMPLE_RATE,
};
use crate::models::{ModelDesc, SlotKind};
use crate::util::with_stdout_suppressed;

pub(crate) struct OnlineSlot {
    pub(crate) name: &'static str,
    pub(crate) recognizer: Arc<OnlineRecognizer>,
    pub(crate) stream: OnlineStream,
    pub(crate) partial: String,
    pub(crate) all_partials: VecDeque<String>,
    pub(crate) finals: Vec<String>,
    pub(crate) all_finals: Vec<String>,
    pub(crate) enabled: bool,
    pub(crate) finals_scroll: u16,
}

pub(crate) fn build_online_slot(desc: &ModelDesc) -> Result<OnlineSlot> {
    let model_dir = engine_models_dir().join(desc.subdir);
    let model_type = match desc.kind {
        SlotKind::Online(mt) => mt,
        SlotKind::Offline(_) => bail!("build_online_slot 收到离线模型"),
    };
    let (rule1, rule2, rule3) = endpoint_rules_from_env();
    let files = online_model_files(desc.name, &model_dir, model_type)?;
    let hotwords = if matches!(model_type, SherpaModelType::Zipformer) {
        hotwords_from_env()
    } else {
        Vec::new()
    };
    let mut cfg = OnlineRecognizerConfig::default();
    cfg.feat_config.feature_dim = online_feature_dim(model_type);
    cfg.model_config.transducer.encoder = Some(files.encoder.to_string_lossy().into_owned());
    cfg.model_config.transducer.decoder = Some(files.decoder.to_string_lossy().into_owned());
    cfg.model_config.transducer.joiner = Some(files.joiner.to_string_lossy().into_owned());
    cfg.model_config.tokens = Some(files.tokens.to_string_lossy().into_owned());
    cfg.model_config.num_threads = 1;
    cfg.model_config.provider = Some("cpu".into());
    cfg.model_config.debug = false;
    cfg.decoding_method = Some(
        match (model_type, hotwords.is_empty()) {
            (SherpaModelType::Zipformer, false) => "modified_beam_search",
            _ => "greedy_search",
        }
        .into(),
    );
    cfg.max_active_paths = 4;
    cfg.enable_endpoint = true;
    cfg.rule1_min_trailing_silence = rule1;
    cfg.rule2_min_trailing_silence = rule2;
    cfg.rule3_min_utterance_length = rule3;
    if !hotwords.is_empty() {
        cfg.hotwords_score = 2.0;
        cfg.hotwords_buf = Some(hotwords.join("\n").into_bytes());
    }

    let recognizer = Arc::new(
        with_stdout_suppressed(|| OnlineRecognizer::create(&cfg)).with_context(|| {
            format!("加载模型 {} 失败，目录 {:?} 是否存在", desc.name, model_dir)
        })?,
    );
    let stream = if hotwords.is_empty() {
        recognizer.create_stream()
    } else {
        recognizer.create_stream_with_hotwords(&hotwords.join("\n"))
    };
    Ok(OnlineSlot {
        name: desc.name,
        recognizer,
        stream,
        partial: String::new(),
        all_partials: VecDeque::new(),
        finals: Vec::new(),
        all_finals: Vec::new(),
        enabled: true,
        finals_scroll: 0,
    })
}

pub(crate) fn feed_online_frame(slot: &mut OnlineSlot, pcm: &[f32]) -> Option<String> {
    if !slot.enabled {
        return None;
    }
    slot.stream.accept_waveform(VAD_SAMPLE_RATE as i32, pcm);
    while slot.recognizer.is_ready(&slot.stream) {
        slot.recognizer.decode(&slot.stream);
    }
    let next_partial = slot
        .recognizer
        .get_result(&slot.stream)
        .map(|result| result.text)
        .unwrap_or_default();
    let trimmed_partial = next_partial.trim();
    if !trimmed_partial.is_empty()
        && slot
            .all_partials
            .back()
            .is_none_or(|last| last.trim() != trimmed_partial)
    {
        slot.all_partials.push_back(trimmed_partial.to_string());
        if slot.all_partials.len() > PARTIAL_HISTORY_RETAIN {
            slot.all_partials.pop_front();
        }
    }
    slot.partial = next_partial;
    if slot.recognizer.is_endpoint(&slot.stream) {
        let final_text = slot
            .recognizer
            .get_result(&slot.stream)
            .map(|result| result.text)
            .unwrap_or_default();
        slot.recognizer.reset(&slot.stream);
        slot.partial.clear();
        if !final_text.trim().is_empty() {
            slot.finals.push(final_text.clone());
            slot.all_finals.push(final_text.clone());
            if slot.finals.len() > FINALS_RETAIN {
                slot.finals.remove(0);
            }
            return Some(final_text);
        }
    }
    None
}

struct OnlineModelFiles {
    encoder: PathBuf,
    decoder: PathBuf,
    joiner: PathBuf,
    tokens: PathBuf,
}

fn online_model_files(
    name: &str,
    model_dir: &Path,
    model_type: SherpaModelType,
) -> Result<OnlineModelFiles> {
    Ok(OnlineModelFiles {
        encoder: first_existing_model_file(
            name,
            model_dir,
            &["encoder.int8.onnx", "encoder.onnx"],
        )?,
        decoder: first_existing_model_file(
            name,
            model_dir,
            match model_type {
                SherpaModelType::NemotronStreaming => &["decoder.int8.onnx", "decoder.onnx"],
                SherpaModelType::Zipformer => &["decoder.onnx", "decoder.int8.onnx"],
            },
        )?,
        joiner: first_existing_model_file(name, model_dir, &["joiner.int8.onnx", "joiner.onnx"])?,
        tokens: first_existing_model_file(name, model_dir, &["tokens.txt"])?,
    })
}

fn first_existing_model_file(name: &str, model_dir: &Path, candidates: &[&str]) -> Result<PathBuf> {
    candidates
        .iter()
        .map(|filename| model_dir.join(filename))
        .find(|path| path.exists())
        .with_context(|| {
            format!(
                "{} 模型文件缺失: 需要 {} 之一，目录 {}",
                name,
                candidates.join(" / "),
                model_dir.display()
            )
        })
}

fn online_feature_dim(model_type: SherpaModelType) -> i32 {
    match model_type {
        SherpaModelType::NemotronStreaming => 128,
        SherpaModelType::Zipformer => 80,
    }
}
