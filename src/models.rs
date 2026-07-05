//! 模型静态描述：区分流式 / 离线两类后端及其文件要求。

use arcvoice_core::asr::streaming::SherpaModelType;
use arcvoice_core::asr::Language;

use crate::config::offline_models_dir;

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

    pub(crate) fn download_script(self) -> &'static str {
        match self {
            Self::Canary => "./scripts/download-canary.sh",
            Self::ParakeetNemoCtc => "./scripts/download-parakeet-tdt-ctc-110m.sh",
        }
    }
}

pub(crate) struct ModelDesc {
    pub(crate) name: &'static str,
    pub(crate) subdir: &'static str,
    pub(crate) kind: SlotKind,
    pub(crate) language: Language,
}

impl ModelDesc {
    /// 是否为"加强版"模型（名称含"加强版"）。
    /// 用于自动注入标点切句配置，避免在多处重复字符串匹配。
    pub(crate) fn is_enhanced(&self) -> bool {
        self.name.contains("加强版")
    }

    /// 离线模型要求目录存在且文件齐全；不齐全时返回 false，调用方据此从选择屏过滤。
    /// 流式模型（Online）一律返回 true（缺文件会在 build_slot 阶段报错，那是另一条路径）。
    pub(crate) fn files_present(&self) -> bool {
        match self.kind {
            SlotKind::Online(_) => true,
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
        match self.kind {
            SlotKind::Online(_) => None,
            SlotKind::Offline(family) => Some(family.download_script()),
        }
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
