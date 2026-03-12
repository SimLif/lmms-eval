#!/usr/bin/env bash
# ==============================================================================
# Environment setup for lmms-eval medical evaluation tasks.
#
# Usage:
#   bash scripts/setup_env.sh           # Base setup (required)
#   bash scripts/setup_env.sh --all     # Base + optional dependencies
#
# What this script does:
#   1. Creates/syncs the venv via uv (from uv.lock)
#   2. Downloads NLTK data for METEOR metric
#   3. Sets LD_LIBRARY_PATH for CUDA/cuDNN
#   4. (--all) Installs RaTEScore + medspacy for medical report evaluation
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_DIR"

# ---------- Color helpers ----------
info()  { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()    { echo -e "\033[1;32m[OK]\033[0m    $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
err()   { echo -e "\033[1;31m[ERROR]\033[0m $*"; }

# ---------- Check prerequisites ----------
if ! command -v uv &>/dev/null; then
    err "uv not found. Install it first: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

# ---------- Parse arguments ----------
INSTALL_ALL=false
for arg in "$@"; do
    case "$arg" in
        --all) INSTALL_ALL=true ;;
        -h|--help)
            echo "Usage: bash scripts/setup_env.sh [--all]"
            echo ""
            echo "Options:"
            echo "  --all    Install optional deps (RaTEScore + medspacy)"
            echo "  -h       Show this help"
            exit 0
            ;;
        *)
            warn "Unknown argument: $arg"
            ;;
    esac
done

# ---------- Step 1: Sync environment ----------
info "Syncing environment from uv.lock..."
uv sync
ok "Environment synced."

# ---------- Step 2: NLTK data ----------
info "Downloading NLTK data (wordnet, punkt_tab, omw-1.4)..."
uv run python3 -c "
import nltk
for res in ['wordnet', 'omw-1.4', 'punkt_tab']:
    nltk.download(res, quiet=True)
print('NLTK data ready.')
"
ok "NLTK data downloaded."

# ---------- Step 3: LD_LIBRARY_PATH ----------
CUDNN_LIB="$PROJECT_DIR/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib"
if [ -d "$CUDNN_LIB" ]; then
    export LD_LIBRARY_PATH="$CUDNN_LIB:${LD_LIBRARY_PATH:-}"
    ok "LD_LIBRARY_PATH set: $CUDNN_LIB"
else
    warn "cuDNN lib not found at $CUDNN_LIB (may be fine if using system CUDA)."
fi

# ---------- Step 4: Flash Attention (prebuilt wheel) ----------
info "Checking flash-attn..."
if uv run python3 -c "import flash_attn" 2>/dev/null; then
    ok "flash-attn already installed ($(uv run python3 -c 'import flash_attn; print(flash_attn.__version__)' 2>/dev/null))."
else
    info "Installing flash-attn from prebuilt wheel..."
    # Prebuilt wheels from https://github.com/mjun0812/flash-attention-prebuild-wheels
    # Avoids 30+ min source compilation. Wheel must match PyTorch + CUDA + Python versions.
    FA_VERSION="2.8.3"
    TORCH_VER="2.8"
    CUDA_VER="cu128"
    PY_VER="cp310"
    WHEEL_NAME="flash_attn-${FA_VERSION}+${CUDA_VER}torch${TORCH_VER}-${PY_VER}-${PY_VER}-linux_x86_64.whl"
    WHEEL_URL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/${WHEEL_NAME}"
    WHEEL_PATH="/tmp/${WHEEL_NAME}"

    if [ ! -f "$WHEEL_PATH" ]; then
        wget -q --show-progress -O "$WHEEL_PATH" "$WHEEL_URL" || {
            err "Failed to download flash-attn wheel. Check network/proxy settings."
            warn "Continuing without flash-attn (InternVL will fall back to naive attention)."
            WHEEL_PATH=""
        }
    fi

    if [ -n "$WHEEL_PATH" ] && [ -f "$WHEEL_PATH" ]; then
        uv pip install "$WHEEL_PATH" || {
            err "Failed to install flash-attn wheel."
            warn "Continuing without flash-attn."
        }
    fi
fi

# ---------- Step 5: Optional - RaTEScore + medspacy ----------
if [ "$INSTALL_ALL" = true ]; then
    info "Installing optional: RaTEScore..."
    uv sync --extra medical-report
    ok "RaTEScore installed."

    info "Installing medspacy (requires uv pip due to broken build metadata)..."
    uv pip install medspacy
    ok "medspacy installed."

    info "Verifying RaTEScore..."
    uv run python3 -c "from RaTEScore import RaTEScore; print('RaTEScore import: OK')" || {
        warn "RaTEScore import failed. Check error above."
    }
fi

# ---------- Step 6: Verify core ----------
info "Verifying core imports..."
uv run python3 -c "
from lmms_eval.tasks._task_utils.report_metrics import (
    calculate_bleu_4, calculate_rouge_l, calculate_meteor,
)
from lmms_eval.tasks._task_utils.answer_utils import (
    parse_multi_choice_response, parse_reasoning_answer,
)
print('Core imports: OK')
"

# ---------- Done ----------
echo ""
ok "Setup complete. Run evaluations with:"
echo ""
echo "  export LD_LIBRARY_PATH=$CUDNN_LIB:\$LD_LIBRARY_PATH"
echo "  uv run python -m lmms_eval --model qwen2_5_vl \\"
echo "    --tasks med_eval --batch_size 1"
echo ""
