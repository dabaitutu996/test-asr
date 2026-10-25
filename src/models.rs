//! 模型静态描述：区分流式 / 离线两类后端及其文件要求。

use std::path::{Path, PathBuf};

use crate::config::{engine_models_dir, offline_models_dir};

/// 统一的模型下载脚本（Linux 版，位于本仓库 scripts/ 下）。
pub(crate) const DOWNLOAD_SCRIPT: &str = "./scripts/download-models.sh";

/// 流式模型的后端族。原本由 arcvoice-core 提供，随 macOS 依赖一起内联到本仓库。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum SherpaModelType {
    /// Zipformer transducer（80 维 fbank）。
    Zipformer,
    /// NVIDIA Nemotron Speech Streaming transducer（128 维 fbank）。
    NemotronStreaming,
}

/// 模型语言，仅用于选择屏展示。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Language {
    Chinese,
    English,
}

impl Language {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Chinese => "中文",
            Self::English => "英文",
        }
    }
}

/// 区分流式（OnlineRecognizer，真流式 transducer）和离线（OfflineRecognizer，
/// 需要外部 VAD 触发解码）两类后端。
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum SlotKind {
    /// sherpa-onnx OnlineRecognizer：Zipformer / Nemotron 等真流式 transducer。
    Online(SherpaModelType),
    /// sherpa-onnx OfflineRecognizer：离线模型 + 共享 VAD 触发。
    Offline(OfflineFamily),
}

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum OfflineFamily {
    Canary,
    ParakeetNemoCtc,
}

impl OfflineFamily {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Canary => "canary",
            Self::ParakeetNemoCtc => "parakeet_nemo_ctc",
        }
    }

    pub(crate) fn required_files(self) -> &'static [&'static str] {
        match self {
            Self::Canary => &["encoder.int8.onnx", "decoder.int8.onnx", "tokens.txt"],
            Self::ParakeetNemoCtc => &["model.int8.onnx", "tokens.txt"],
        }
    }
}

pub(crate) struct ModelDesc {
    pub(crate) name: &'static str,
    pub(crate) subdir: &'static str,
    pub(crate) kind: SlotKind,
    pub(crate) language: Language,
}

/// 流式模型实际解析到的文件集合。
pub(crate) struct OnlineModelFiles {
    pub(crate) encoder: PathBuf,
    pub(crate) decoder: PathBuf,
    pub(crate) joiner: PathBuf,
    pub(crate) tokens: PathBuf,
}

impl ModelDesc {
    /// 是否为"加强版"模型（名称含"加强版"）。
    /// 用于自动注入标点切句配置，避免在多处重复字符串匹配。
    pub(crate) fn is_enhanced(&self) -> bool {
        self.name.contains("加强版")
    }

    /// 模型文件是否齐全。不齐全时调用方从选择屏过滤并提示下载脚本。
    /// Linux 版对流式模型也做文件探测，避免选到缺失模型后直接崩在加载阶段。
    pub(crate) fn files_present(&self) -> bool {
        match self.kind {
            SlotKind::Online(model_type) => {
                let dir = engine_models_dir().join(self.subdir);
                online_model_files(&dir, model_type).is_some()
            }
            SlotKind::Offline(family) => {
                let dir = offline_models_dir().join(self.subdir);
                family
                    .required_files()
                    .iter()
                    .all(|filename| dir.join(filename).exists())
            }
        }
    }

    pub(crate) fn missing_files_hint(&self) -> Option<&'static str> {
        if self.files_present() {
            None
        } else {
            Some(DOWNLOAD_SCRIPT)
        }
    }
}

/// 依次尝试候选文件名，返回第一个存在的。
/// 兜底：sherpa 官方发布包用的是 `encoder-epoch-99-avg-1.int8.onnx` 这类名字，
/// 若用户手工解包、没跑归一化脚本，就按前缀匹配（优先 int8）。
fn first_existing(dir: &Path, candidates: &[&str]) -> Option<PathBuf> {
    if let Some(path) = candidates
        .iter()
        .map(|name| dir.join(name))
        .find(|path| path.exists())
    {
        return Some(path);
    }

    let prefix = candidates.first()?.split('.').next()?;
    let mut matches: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.starts_with(prefix) && name.ends_with(".onnx"))
                .unwrap_or(false)
        })
        .collect();
    matches.sort_by_key(|path| {
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        (!name.contains("int8"), name.len())
    });
    matches.into_iter().next()
}

/// 按模型类型解析流式模型的四个必需文件。任一缺失即返回 None。
pub(crate) fn online_model_files(
    model_dir: &Path,
    model_type: SherpaModelType,
) -> Option<OnlineModelFiles> {
    let decoder_candidates: &[&str] = match model_type {
        SherpaModelType::NemotronStreaming => &["decoder.int8.onnx", "decoder.onnx"],
        SherpaModelType::Zipformer => &["decoder.onnx", "decoder.int8.onnx"],
    };
    Some(OnlineModelFiles {
        encoder: first_existing(model_dir, &["encoder.int8.onnx", "encoder.onnx"])?,
        decoder: first_existing(model_dir, decoder_candidates)?,
        joiner: first_existing(model_dir, &["joiner.int8.onnx", "joiner.onnx"])?,
        tokens: first_existing(model_dir, &["tokens.txt"])?,
    })
}

/// 流式模型的特征维度：Nemotron 用 128 维 mel，Zipformer 用 80 维。
pub(crate) fn online_feature_dim(model_type: SherpaModelType) -> i32 {
    match model_type {
        SherpaModelType::NemotronStreaming => 128,
        SherpaModelType::Zipformer => 80,
    }
}

pub(crate) const MODEL_DESCS: &[ModelDesc] = &[
    ModelDesc {
        name: "Zipformer-zh",
        subdir: "zipformer-zh",
        kind: SlotKind::Online(SherpaModelType::Zipformer),
        language: Language::Chinese,
    },
    ModelDesc {
        name: "Zipformer-en",
        subdir: "zipformer-en",
        kind: SlotKind::Online(SherpaModelType::Zipformer),
        language: Language::English,
    },
    ModelDesc {
        name: "Zipformer-en (加强版)",
        subdir: "zipformer-en-enhanced",
        kind: SlotKind::Online(SherpaModelType::Zipformer),
        language: Language::English,
    },
    ModelDesc {
        name: "Nemotron-en",
        subdir: "nemotron-en",
        kind: SlotKind::Online(SherpaModelType::NemotronStreaming),
        language: Language::English,
    },
    ModelDesc {
        name: "Canary-180m-flash",
        subdir: "canary-180m-flash",
        kind: SlotKind::Offline(OfflineFamily::Canary),
        language: Language::English,
    },
    ModelDesc {
        name: "Parakeet-TDT-CTC-110M",
        subdir: "parakeet-tdt-ctc-110m",
        kind: SlotKind::Offline(OfflineFamily::ParakeetNemoCtc),
        language: Language::English,
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("asr-compare-models-{tag}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("创建临时模型目录");
        dir
    }

    fn touch(dir: &Path, name: &str) {
        std::fs::write(dir.join(name), b"stub").expect("写入占位文件");
    }

    #[test]
    fn resolves_zipformer_prefers_float_decoder() {
        let dir = temp_dir("zipformer");
        touch(&dir, "encoder.int8.onnx");
        touch(&dir, "decoder.onnx");
        touch(&dir, "decoder.int8.onnx");
        touch(&dir, "joiner.int8.onnx");
        touch(&dir, "tokens.txt");

        let files = online_model_files(&dir, SherpaModelType::Zipformer).expect("应解析成功");
        assert_eq!(files.encoder.file_name().unwrap(), "encoder.int8.onnx");
        // Zipformer 优先 decoder.onnx（非 int8）
        assert_eq!(files.decoder.file_name().unwrap(), "decoder.onnx");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn resolves_nemotron_prefers_int8_decoder() {
        let dir = temp_dir("nemotron");
        touch(&dir, "encoder.onnx");
        touch(&dir, "decoder.onnx");
        touch(&dir, "decoder.int8.onnx");
        touch(&dir, "joiner.onnx");
        touch(&dir, "tokens.txt");

        let files =
            online_model_files(&dir, SherpaModelType::NemotronStreaming).expect("应解析成功");
        assert_eq!(files.decoder.file_name().unwrap(), "decoder.int8.onnx");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn missing_joiner_yields_none() {
        let dir = temp_dir("missing-joiner");
        touch(&dir, "encoder.onnx");
        touch(&dir, "decoder.onnx");
        touch(&dir, "tokens.txt");

        assert!(online_model_files(&dir, SherpaModelType::Zipformer).is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn feature_dim_differs_by_backend() {
        assert_eq!(online_feature_dim(SherpaModelType::Zipformer), 80);
        assert_eq!(online_feature_dim(SherpaModelType::NemotronStreaming), 128);
    }

    #[test]
    fn falls_back_to_epoch_named_release_files() {
        let dir = temp_dir("epoch-names");
        touch(&dir, "encoder-epoch-99-avg-1.int8.onnx");
        touch(&dir, "decoder-epoch-99-avg-1.onnx");
        touch(&dir, "joiner-epoch-99-avg-1.int8.onnx");
        touch(&dir, "tokens.txt");

        let files = online_model_files(&dir, SherpaModelType::Zipformer)
            .expect("未归一化的官方发布包也应能解析");
        assert_eq!(
            files.encoder.file_name().unwrap(),
            "encoder-epoch-99-avg-1.int8.onnx"
        );
        assert_eq!(
            files.decoder.file_name().unwrap(),
            "decoder-epoch-99-avg-1.onnx"
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
