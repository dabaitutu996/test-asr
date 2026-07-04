//! 直接把本地 wav 文件喂给流式 ASR，对比多个模型识别结果。
//!
//! 用途：
//! - 先验证 ASR 模型本身是否工作正常
//! - 完全绕开 macOS 系统音频捕获 / TCC
//!
//! 用法：
//! cargo run --example transcribe_wav -- /path/to/file.wav

#![cfg(not(target_os = "windows"))]

use std::env;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use arcvoice_core::asr::streaming::{
    OnlineAsrEngine, OnlineAsrStream, SherpaConfig, SherpaModelType, SherpaOnlineAsrEngine,
};
use arcvoice_core::asr::Language;

const TARGET_SAMPLE_RATE: u32 = 16_000;
const CHUNK_SIZE: usize = 3_200;
const FLUSH_CHUNKS: usize = 15;
const FINALS_RETAIN: usize = 32;

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

struct ModelSlot {
    name: &'static str,
    #[allow(dead_code)]
    engine: SherpaOnlineAsrEngine,
    stream: Box<dyn OnlineAsrStream>,
    partial: String,
    finals: Vec<String>,
}

fn engine_models_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("ENGINE_MODELS_DIR") {
        return PathBuf::from(dir);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../game-video/engine/models/streaming")
}

fn build_slot(desc: &ModelDesc) -> Result<ModelSlot> {
    let model_dir = engine_models_dir().join(desc.subdir);
    let cfg = SherpaConfig::new(model_dir.clone(), desc.model_type, desc.language);
    let engine = SherpaOnlineAsrEngine::load(cfg)
        .with_context(|| format!("加载模型 {} 失败，目录 {:?} 是否存在", desc.name, model_dir))?;
    let mut stream = engine.create_stream()?;
    stream.prepare();
    Ok(ModelSlot {
        name: desc.name,
        engine,
        stream,
        partial: String::new(),
        finals: Vec::new(),
    })
}

fn feed_frame(slot: &mut ModelSlot, pcm: &[f32]) -> Option<String> {
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

fn max_amplitude(bits_per_sample: u16) -> f32 {
    let bits = bits_per_sample.clamp(1, 31) as u32;
    ((1_i64 << (bits - 1)) - 1) as f32
}

fn decode_wav(path: &Path) -> Result<(Vec<f32>, hound::WavSpec)> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("打开 wav 失败: {}", path.display()))?;
    let spec = reader.spec();

    let samples = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .map(|s| s.context("读取 f32 wav 样本失败"))
            .collect::<Result<Vec<_>, _>>()?,
        (hound::SampleFormat::Int, bits) if bits <= 16 => reader
            .samples::<i16>()
            .map(|s| {
                s.map(|v| v as f32 / i16::MAX as f32)
                    .context("读取 i16 wav 样本失败")
            })
            .collect::<Result<Vec<_>, _>>()?,
        (hound::SampleFormat::Int, bits) => {
            let scale = max_amplitude(bits);
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / scale).context("读取 i32 wav 样本失败"))
                .collect::<Result<Vec<_>, _>>()?
        }
        (fmt, bits) => {
            bail!("暂不支持的 wav 格式: {:?}, {} bits", fmt, bits);
        }
    };

    Ok((samples, spec))
}

fn downmix_to_mono(samples: &[f32], channels: usize) -> Result<Vec<f32>> {
    if channels == 0 {
        bail!("wav 声道数为 0");
    }
    if channels == 1 {
        return Ok(samples.to_vec());
    }

    let mut mono = Vec::with_capacity(samples.len() / channels);
    for frame in samples.chunks_exact(channels) {
        let sum: f32 = frame.iter().copied().sum();
        mono.push(sum / channels as f32);
    }
    Ok(mono)
}

fn resample_linear(samples: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if src_rate == dst_rate || samples.is_empty() {
        return samples.to_vec();
    }

    let dst_len = ((samples.len() as f64) * dst_rate as f64 / src_rate as f64).round() as usize;
    let mut out = Vec::with_capacity(dst_len);

    for i in 0..dst_len {
        let pos = i as f64 * src_rate as f64 / dst_rate as f64;
        let left = pos.floor() as usize;
        let frac = (pos - left as f64) as f32;
        let a = samples[left.min(samples.len() - 1)];
        let b = samples[(left + 1).min(samples.len() - 1)];
        out.push(a + (b - a) * frac);
    }

    out
}

fn flush_slots(slots: &mut [ModelSlot]) {
    let silence = vec![0.0_f32; CHUNK_SIZE];
    for _ in 0..FLUSH_CHUNKS {
        for slot in slots.iter_mut() {
            let _ = feed_frame(slot, &silence);
        }
    }
}

fn main() -> Result<()> {
    let wav_path = env::args()
        .nth(1)
        .context("用法: cargo run --example transcribe_wav -- /path/to/file.wav")?;
    let wav_path = PathBuf::from(wav_path);

    if wav_path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.eq_ignore_ascii_case("wav"))
        != Some(true)
    {
        bail!("当前 example 只支持 wav 文件");
    }

    let (raw_samples, spec) = decode_wav(&wav_path)?;
    let mono = downmix_to_mono(&raw_samples, spec.channels as usize)?;
    let audio = resample_linear(&mono, spec.sample_rate, TARGET_SAMPLE_RATE);

    println!("== File ASR Compare ==");
    println!("file: {}", wav_path.display());
    println!(
        "source: {} Hz, {} ch, {:?}, {} bits",
        spec.sample_rate, spec.channels, spec.sample_format, spec.bits_per_sample
    );
    println!(
        "prepared: {} Hz mono, {:.2} sec",
        TARGET_SAMPLE_RATE,
        audio.len() as f32 / TARGET_SAMPLE_RATE as f32
    );
    println!();

    let mut slots = MODEL_DESCS
        .iter()
        .map(build_slot)
        .collect::<Result<Vec<_>>>()?;

    for chunk in audio.chunks(CHUNK_SIZE) {
        for slot in &mut slots {
            if let Some(final_text) = feed_frame(slot, chunk) {
                println!("[{}] final: {}", slot.name, final_text);
            }
        }
    }

    flush_slots(&mut slots);

    println!();
    println!("== Summary ==");
    for slot in &slots {
        println!();
        println!("[{}]", slot.name);
        if slot.finals.is_empty() {
            println!("finals: <none>");
        } else {
            for line in &slot.finals {
                println!("final: {}", line);
            }
        }
        if !slot.partial.trim().is_empty() {
            println!("partial: {}", slot.partial);
        }
    }

    Ok(())
}
