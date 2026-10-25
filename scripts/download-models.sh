#!/usr/bin/env bash
# 下载 test-asr 在 Linux 上需要的模型（流式 / 离线 / Silero VAD）。
#
# 用法：
#   ./scripts/download-models.sh                # Zipformer 中/英 + Silero VAD
#   ./scripts/download-models.sh --all          # 上述 + Nemotron + Canary（体积大）
#   ./scripts/download-models.sh --offline      # 仅离线模型（Canary）
#   ./scripts/download-models.sh --vad          # 仅 Silero VAD
#   ./scripts/download-models.sh zipformer-zh   # 只下载单个模型
#
# 目标目录默认是 <repo>/models/streaming（受 ENGINE_MODELS_DIR 影响）。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${ENGINE_MODELS_DIR:-$ROOT/models/streaming}"
VAD_DIR="$ROOT/models/vad/silero_vad"
BASE_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
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
    log "下载 silero_vad.onnx"
    curl -fL --retry 3 --progress-bar -o "$target" "$BASE_URL/silero_vad.onnx"
}

download_zipformer_zh() {
    fetch_model sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23 "$MODELS_DIR/zipformer-zh"
}

download_zipformer_en() {
    fetch_model sherpa-onnx-streaming-zipformer-en-2023-06-26 "$MODELS_DIR/zipformer-en"
    # 「加强版」与普通版是同一套声学模型，区别只在切句策略（src/segments.rs）。
    mkdir -p "$MODELS_DIR/zipformer-en-enhanced"
    if [ -z "$(ls -A "$MODELS_DIR/zipformer-en-enhanced" 2>/dev/null)" ]; then
        cp -a "$MODELS_DIR/zipformer-en/." "$MODELS_DIR/zipformer-en-enhanced/"
        log "完成: $MODELS_DIR/zipformer-en-enhanced（复用 zipformer-en 模型）"
    fi
}

download_nemotron() {
    fetch_model sherpa-onnx-nemotron-speech-streaming-en-0.6b-int8-2026-01-14 \
        "$MODELS_DIR/nemotron-en" 1
}

download_offline() {
    fetch_model sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8 \
        "$MODELS_DIR/canary-180m-flash"
    cat <<'EOF'
Parakeet-TDT-CTC-110M 未在脚本内固化地址：
  到 https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models 找到
  parakeet-tdt-ctc-110m 的 int8 包，解包到 models/streaming/parakeet-tdt-ctc-110m/
EOF
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
        fetch_model sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8 \
            "$MODELS_DIR/canary-180m-flash"
        ;;
    *)
        cat <<EOF
用法: $0 [选项]

  (无参数)            下载 Zipformer 中/英 + Silero VAD
  --all               上述 + Nemotron + Canary（体积较大）
  --offline           仅离线模型（Canary）
  --vad               仅 Silero VAD
  zipformer-zh        只下载中文 Zipformer
  zipformer-en        只下载英文 Zipformer（含"加强版"）
  nemotron-en         只下载 Nemotron 流式模型
  canary-180m-flash   只下载 Canary 离线模型
EOF
        exit 1
        ;;
esac

log "模型目录: $MODELS_DIR"
