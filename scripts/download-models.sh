#!/usr/bin/env bash
# 下载 test-asr 在 Linux 上需要的模型（流式 / 离线 / Silero VAD）。
#
# 用法：
#   ./scripts/download-models.sh                # Zipformer 中/英 + Silero VAD
#   ./scripts/download-models.sh --all          # 上述 + Nemotron + Canary + Parakeet（体积大）
#   ./scripts/download-models.sh --offline      # 仅离线模型（Canary + Parakeet）
#   ./scripts/download-models.sh --vad          # 仅 Silero VAD
#   ./scripts/download-models.sh zipformer-zh   # 只下载单个模型
#
# 目标目录默认是 <repo>/models/streaming（受 ENGINE_MODELS_DIR 影响）。
#
# 下载源：
#   默认优先使用本仓库自己的 release 镜像（models-v1），失败时自动回退到
#   sherpa-onnx 官方 asr-models release。可用环境变量覆盖：
#     MODELS_SOURCE=upstream        直接走上游，不用镜像
#     MODELS_RELEASE_TAG=<tag>      换一个镜像 release 标签（默认 models-v1）
#     MODELS_BASE_URL=<url>         完全自定义镜像地址
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${ENGINE_MODELS_DIR:-$ROOT/models/streaming}"
VAD_DIR="$ROOT/models/vad/silero_vad"
BASE_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
MODELS_SOURCE="${MODELS_SOURCE:-mirror}"
MODELS_RELEASE_TAG="${MODELS_RELEASE_TAG:-models-v1}"
MIRROR_BASE_URL="${MODELS_BASE_URL:-https://github.com/dabaitutu996/test-asr/releases/download/$MODELS_RELEASE_TAG}"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

log() { printf '\033[36m==>\033[0m %s\n' "$*"; }

need() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "缺少命令: $1" >&2
        exit 1
    }
}
need curl
need tar

# 目标目录已存在且非空时视为完成。
dir_ready() {
    [ -d "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null)" ]
}

# ─── 镜像下载（本仓库 release，tar.zst + SHA256SUMS） ────────────────────
ZSTD_OK=1
command -v zstd >/dev/null 2>&1 || ZSTD_OK=0
MIRROR_READY="$([ "$MODELS_SOURCE" = "upstream" ] && echo 0 || echo 1)"
SHA_FILE="$TMP_DIR/SHA256SUMS"

# 校验镜像包；清单拉不到时只告警，不阻塞。
verify_asset() {
    local archive="$1" name want got
    name="$(basename "$archive")"
    if [ ! -s "$SHA_FILE" ]; then
        curl -fsL --retry 2 -o "$SHA_FILE" "$MIRROR_BASE_URL/SHA256SUMS" 2>/dev/null || true
    fi
    if [ ! -s "$SHA_FILE" ]; then
        log "未取到 SHA256SUMS，跳过校验: $name"
        return 0
    fi
    want="$(awk -v n="$name" '$2 == n {print $1; exit}' "$SHA_FILE")"
    if [ -z "$want" ]; then
        log "清单中没有 $name，跳过校验"
        return 0
    fi
    got="$(sha256sum "$archive" | awk '{print $1}')"
    if [ "$got" != "$want" ]; then
        echo "    校验失败: $name" >&2
        return 1
    fi
    log "校验通过: $name"
}

# fetch_mirror <镜像 asset 名(不含扩展名)> <目标目录>
# 返回非 0 表示镜像不可用，由调用方回退上游。
fetch_mirror() {
    local name="$1" target="$2"
    local archive="$TMP_DIR/$name.tar.zst" unpack="$TMP_DIR/unpack-$name" inner

    if dir_ready "$target"; then
        log "已存在，跳过: $target"
        return 0
    fi
    if [ "$MIRROR_READY" = "0" ]; then
        return 1
    fi
    if [ "$ZSTD_OK" = "0" ]; then
        log "未安装 zstd，改用上游 .tar.bz2（可 apt install zstd 后走镜像）"
        MIRROR_READY=0
        return 1
    fi

    log "下载镜像 $name.tar.zst"
    if ! curl -fL --retry 3 --progress-bar -o "$archive" "$MIRROR_BASE_URL/$name.tar.zst"; then
        log "镜像不可用，回退上游"
        MIRROR_READY=0
        return 1
    fi
    if ! verify_asset "$archive"; then
        rm -f "$archive"
        return 1
    fi

    rm -rf "$unpack"
    mkdir -p "$unpack"
    if ! tar --zstd -xf "$archive" -C "$unpack"; then
        echo "    解压失败: $name.tar.zst" >&2
        rm -rf "$unpack" "$archive"
        return 1
    fi
    # 镜像内层通常带一个同名目录（zipformer-zh/...），flatten 到目标目录。
    inner="$unpack"
    if [ -d "$unpack/$(basename "$target")" ]; then
        inner="$unpack/$(basename "$target")"
    fi
    mkdir -p "$target"
    cp -a "$inner"/. "$target"/
    rm -rf "$unpack" "$archive"
    normalize_names "$target"
    log "完成: $target"
    return 0
}

# fetch_model <asset 名> <目标目录> [optional=0|1]
fetch_model() {
    local asset="$1" target="$2" optional="${3:-0}"
    local archive="$TMP_DIR/$asset.tar.bz2"

    if [ -d "$target" ] && [ -n "$(ls -A "$target" 2>/dev/null)" ]; then
        log "已存在，跳过: $target"
        return 0
    fi

    log "下载 $asset"
    if ! curl -fL --retry 3 --progress-bar -o "$archive" "$BASE_URL/$asset.tar.bz2"; then
        if [ "$optional" = "1" ]; then
            echo "    下载失败（可选模型，已跳过）: $asset" >&2
            return 0
        fi
        echo "    下载失败: $asset" >&2
        return 1
    fi

    mkdir -p "$target" 2>/dev/null || true
    tar -xjf "$archive" -C "$TMP_DIR"
    local inner="$TMP_DIR/$asset"
    if [ ! -d "$inner" ]; then
        inner="$(find "$TMP_DIR" -maxdepth 1 -mindepth 1 -type d ! -path "$TMP_DIR" | head -1)"
    fi
    mkdir -p "$target"
    cp -a "$inner"/. "$target"/
    rm -rf "$inner" "$archive"
    normalize_names "$target"
    log "完成: $target"
}

# sherpa 发布包里的模型名带 epoch 前缀（如 encoder-epoch-99-avg-1.int8.onnx），
# 统一归一化成应用约定的 encoder.int8.onnx / decoder.onnx / joiner.int8.onnx。
normalize_names() {
    local dir="$1"
    [ -d "$dir" ] || return 0
    local kind file base name
    for kind in encoder decoder joiner model; do
        for file in "$dir/$kind"*.onnx; do
            [ -e "$file" ] || continue
            name="$(basename "$file")"
            case "$name" in
                "$kind.int8.onnx" | "$kind.onnx") continue ;;
            esac
            if [[ "$name" == *int8* ]]; then
                base="$kind.int8.onnx"
            else
                base="$kind.onnx"
            fi
            mv -f "$file" "$dir/$base"
        done
    done
}

download_vad() {
    mkdir -p "$VAD_DIR"
    local target="$VAD_DIR/silero_vad.int8.onnx"
    if [ -f "$target" ]; then
        log "已存在，跳过: $target"
        return 0
    fi
    if fetch_mirror asr-models-vad "$VAD_DIR" && [ -f "$target" ]; then
        return 0
    fi
    log "下载 silero_vad.onnx"
    curl -fL --retry 3 --progress-bar -o "$target" "$BASE_URL/silero_vad.onnx"
}

download_zipformer_zh() {
    fetch_mirror asr-models-zipformer-zh "$MODELS_DIR/zipformer-zh" ||
        fetch_model sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23 "$MODELS_DIR/zipformer-zh"
}

download_zipformer_en() {
    fetch_mirror asr-models-zipformer-en "$MODELS_DIR/zipformer-en" ||
        fetch_model sherpa-onnx-streaming-zipformer-en-2023-06-26 "$MODELS_DIR/zipformer-en"
    # 「加强版」与普通版是同一套声学模型，区别只在切句策略（src/segments.rs）。
    mkdir -p "$MODELS_DIR/zipformer-en-enhanced"
    if [ -z "$(ls -A "$MODELS_DIR/zipformer-en-enhanced" 2>/dev/null)" ]; then
        cp -a "$MODELS_DIR/zipformer-en/." "$MODELS_DIR/zipformer-en-enhanced/"
        log "完成: $MODELS_DIR/zipformer-en-enhanced（复用 zipformer-en 模型）"
    fi
}

download_nemotron() {
    fetch_mirror asr-models-nemotron-en "$MODELS_DIR/nemotron-en" ||
        fetch_model sherpa-onnx-nemotron-speech-streaming-en-0.6b-int8-2026-01-14 \
            "$MODELS_DIR/nemotron-en" 1
}

download_canary() {
    fetch_mirror asr-models-canary-180m-flash "$MODELS_DIR/canary-180m-flash" ||
        fetch_model sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8 \
            "$MODELS_DIR/canary-180m-flash"
}

download_parakeet() {
    fetch_mirror asr-models-parakeet-tdt-ctc-110m "$MODELS_DIR/parakeet-tdt-ctc-110m" ||
        fetch_model sherpa-onnx-nemo-parakeet_tdt_ctc_110m-en-36000-int8 \
            "$MODELS_DIR/parakeet-tdt-ctc-110m"
}

download_offline() {
    download_canary
    download_parakeet
}

download_default() {
    download_zipformer_zh
    download_zipformer_en
    download_vad
}

case "${1:-}" in
    "" | --streaming)
        download_default
        ;;
    --all)
        download_default
        download_nemotron
        download_offline
        ;;
    --offline)
        download_offline
        download_vad
        ;;
    --vad)
        download_vad
        ;;
    zipformer-zh)
        download_zipformer_zh
        ;;
    zipformer-en)
        download_zipformer_en
        ;;
    nemotron-en)
        download_nemotron
        ;;
    canary-180m-flash)
        download_canary
        ;;
    parakeet-tdt-ctc-110m)
        download_parakeet
        ;;
    *)
        cat <<EOF
用法: $0 [选项]

   (无参数)            下载 Zipformer 中/英 + Silero VAD
  --all               上述 + Nemotron + Canary + Parakeet（体积较大）
  --offline           仅离线模型（Canary + Parakeet）
  --vad               仅 Silero VAD
  zipformer-zh        只下载中文 Zipformer
  zipformer-en        只下载英文 Zipformer（含"加强版"）
  nemotron-en         只下载 Nemotron 流式模型
  canary-180m-flash   只下载 Canary 离线模型
  parakeet-tdt-ctc-110m  只下载 Parakeet 离线模型

环境变量:
  MODELS_SOURCE=upstream       跳过镜像，直接用 sherpa-onnx 上游 release
  MODELS_RELEASE_TAG=<tag>     镜像 release 标签（默认 models-v1）
  MODELS_BASE_URL=<url>        自定义镜像地址
EOF
        exit 1
        ;;
esac

log "模型目录: $MODELS_DIR"
