//! 模型校验示例（Linux）：把一段 16kHz mono WAV 依次喂给各流式模型，
//! 打印识别结果与耗时。无需音频设备，适合在装模型后先做冒烟测试。
//!
//! 用法：
//! ```text
//! cargo run --release --example transcribe_wav -- path/to/16k.wav
//! ENGINE_MODELS_DIR=/data/models/streaming cargo run --example transcribe_wav -- a.wav
//! ```
//!
//! WAV 必须是 16kHz 单声道，转换示例：
//! `ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav`

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use sherpa_onnx::{OnlineRecognizer, OnlineRecognizerConfig};

const SAMPLE_RATE: u32 = 16_000;
/// 每次喂 200ms，模拟真实流式节奏。
const CHUNK_SAMPLES: usize = 3_200;

struct ModelDesc {
    name: &'static str,
    subdir: &'static str,
    feature_dim: i32,
}

const MODELS: &[ModelDesc] = &[
    ModelDesc {
        name: "Zipformer-zh",
        subdir: "zipformer-zh",
        feature_dim: 80,
    },
    ModelDesc {
        name: "Zipformer-en",
        subdir: "zipformer-en",
        feature_dim: 80,
    },
    ModelDesc {
        name: "Nemotron-en",
        subdir: "nemotron-en",
        feature_dim: 128,
    },
];

fn main() -> Result<()> {
    let wav = std::env::args()
        .nth(1)
        .context("用法: cargo run --example transcribe_wav -- <16kHz mono wav>")?;
    let samples = read_wav(Path::new(&wav))?;
    println!(
        "音频: {wav} ({:.2}s, {} 样本)",
        samples.len() as f32 / SAMPLE_RATE as f32,
        samples.len()
    );

    let models_dir = std::env::var("ENGINE_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/streaming"));

    for desc in MODELS {
        let dir = models_dir.join(desc.subdir);
        let Some(files) = resolve_files(&dir) else {
            println!(
                "\n[{}] 跳过：{} 缺少模型文件（先运行 ./scripts/download-models.sh）",
                desc.name,
                dir.display()
            );
            continue;
        };

        let mut cfg = OnlineRecognizerConfig::default();
        cfg.feat_config.feature_dim = desc.feature_dim;
        cfg.model_config.transducer.encoder = Some(files.encoder.to_string_lossy().into_owned());
        cfg.model_config.transducer.decoder = Some(files.decoder.to_string_lossy().into_owned());
        cfg.model_config.transducer.joiner = Some(files.joiner.to_string_lossy().into_owned());
        cfg.model_config.tokens = Some(files.tokens.to_string_lossy().into_owned());
        cfg.model_config.num_threads = 1;
        cfg.model_config.provider = Some("cpu".into());
        cfg.model_config.debug = false;
        cfg.decoding_method = Some("greedy_search".into());
        cfg.max_active_paths = 4;
        cfg.enable_endpoint = true;
        cfg.rule1_min_trailing_silence = 0.8;
        cfg.rule2_min_trailing_silence = 1.2;
        cfg.rule3_min_utterance_length = 20.0;

        let recognizer = OnlineRecognizer::create(&cfg)
            .with_context(|| format!("加载模型 {} 失败（目录 {}）", desc.name, dir.display()))?;
        let stream = recognizer.create_stream();

        let started = Instant::now();
        for chunk in samples.chunks(CHUNK_SAMPLES) {
            stream.accept_waveform(SAMPLE_RATE as i32, chunk);
            while recognizer.is_ready(&stream) {
                recognizer.decode(&stream);
            }
        }
        stream.input_finished();
        while recognizer.is_ready(&stream) {
            recognizer.decode(&stream);
        }
        let text = recognizer
            .get_result(&stream)
            .map(|result| result.text)
            .unwrap_or_default();

        println!(
            "\n[{}] 耗时 {:.2}s\n  {}",
            desc.name,
            started.elapsed().as_secs_f32(),
            text.trim()
        );
    }

    Ok(())
}

struct ModelFiles {
    encoder: PathBuf,
    decoder: PathBuf,
    joiner: PathBuf,
    tokens: PathBuf,
}

fn resolve_files(dir: &Path) -> Option<ModelFiles> {
    Some(ModelFiles {
        encoder: first_existing(dir, &["encoder.int8.onnx", "encoder.onnx"])?,
        decoder: first_existing(dir, &["decoder.int8.onnx", "decoder.onnx"])?,
        joiner: first_existing(dir, &["joiner.int8.onnx", "joiner.onnx"])?,
        tokens: first_existing(dir, &["tokens.txt"])?,
    })
}

fn first_existing(dir: &Path, candidates: &[&str]) -> Option<PathBuf> {
    candidates
        .iter()
        .map(|name| dir.join(name))
        .find(|path| path.exists())
}

fn read_wav(path: &Path) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("打开 WAV 失败: {}", path.display()))?;
    let spec = reader.spec();
    if spec.channels != 1 || spec.sample_rate != SAMPLE_RATE {
        bail!(
            "需要 16kHz 单声道 WAV（当前 {}Hz / {} 声道）。先转码：ffmpeg -i 输入 -ar 16000 -ac 1 out.wav",
            spec.sample_rate,
            spec.channels
        );
    }
    match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .context("读取浮点 WAV 样本失败"),
        hound::SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|sample| sample.map(|value| value as f32 / scale))
                .collect::<Result<Vec<_>, _>>()
                .context("读取整型 WAV 样本失败")
        }
    }
}
