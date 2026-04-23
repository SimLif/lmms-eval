#!/usr/bin/env bash
# ==============================================================================
# lmms-eval environment setup.
#
# Usage:
#   bash scripts/setup_env.sh              # Incremental sync + install missing
#   bash scripts/setup_env.sh --fresh      # Delete .venv + full recreate
#   bash scripts/setup_env.sh --all        # + optional deps (RaTEScore + medspacy)
#
# Prebuilt wheels (afs):
#   flash-attn:      /afs_guohaoqiang/wheels/flash_attn-*cp310*.whl
#   causal-conv1d:   /afs_guohaoqiang/wheels/causal_conv1d-*cp310*.whl
#   mamba-ssm:       /afs_guohaoqiang/wheels/mamba_ssm-*cp310*.whl
#   flash-linear-attention: compiled from source (lightweight triton kernels)
#
# Wheel 安装坑位：必须用 .venv/bin/python -m pip（而非 uv pip）装本地 whl。
# uv pip 会 resolve 到 PyPI 最新版并尝试源码编译，忽略传入的 whl 路径。
# 另外：uv sync 可能从 uv.lock 拉取同版本号但 ABI 不匹配的 whl 覆盖我们的
# AFS whl（元数据一致导致 pip 拒绝覆盖），因此必须 --force-reinstall。
#
# B卡 (sm_100, CUDA 13.x) 支持:
#   当前 pyproject.toml 锁定 torch==2.8.0+cu128，不兼容 CUDA 13.x。
#   B卡需要升级到 torch>=2.11.0+cu130 后重新运行此脚本。
#   参考: https://download.pytorch.org/whl/cu130/
# ==============================================================================

set -euo pipefail

# Load credentials (proxy, tokens)
SECRETS_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.secrets/env"
[ -f "$SECRETS_ENV" ] && source "$SECRETS_ENV"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

info()  { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()    { echo -e "\033[1;32m[OK]\033[0m    $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
err()   { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; }

# ---------- Parse arguments ----------
FRESH=false
INSTALL_ALL=false
for arg in "$@"; do
    case "$arg" in
        --fresh) FRESH=true ;;
        --all) INSTALL_ALL=true ;;
        -h|--help)
            echo "Usage: bash scripts/setup_env.sh [--fresh] [--all]"
            echo "  --fresh    Delete .venv and recreate from scratch"
            echo "  --all      Install optional deps (RaTEScore + medspacy)"
            exit 0 ;;
    esac
done

# CUDA compilation config
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export MAX_JOBS="${MAX_JOBS:-8}"

# Prebuilt wheel directory
AFS_WHEELS="/afs_guohaoqiang/wheels"

# ==============================================================================
# Step 1: uv sync
# ==============================================================================
if [ "$FRESH" = true ]; then
    info "Step 1: 清理旧 venv + uv sync..."
    rm -rf .venv
    uv sync 2>&1 | tail -5
else
    info "Step 1: uv sync --inexact (保留额外安装的包)..."
    uv sync --inexact 2>&1 | tail -5
fi
ok "uv sync 完成"

# ==============================================================================
# Step 1.5: [B卡 ONLY] Patch torch CUDA version check
# B卡 (sm_100) 系统 CUDA 13.x 与 PyTorch cu128 major 版本不匹配，
# torch.utils.cpp_extension._check_cuda_version() 会报错拒绝编译 CUDA 扩展。
# 取消注释以下代码块启用 patch（编译完成后会自动还原）。
# 更优方案：升级到 torch>=2.11.0+cu130 后无需 patch。
# ==============================================================================
# CPP_EXT=".venv/lib/python3.10/site-packages/torch/utils/cpp_extension.py"
# if [ -f "$CPP_EXT" ] && grep -q "_check_cuda_version" "$CPP_EXT"; then
#     info "Patch torch CUDA 版本检查 (B卡)..."
#     cp "$CPP_EXT" "${CPP_EXT}.bak"
#     sed -i 's/            _check_cuda_version(compiler_name, compiler_version)/            pass  # _check_cuda_version PATCHED/' "$CPP_EXT"
#     CUDA_PATCHED=true
# fi

# ==============================================================================
# Step 1.7: bootstrap pip into .venv
# uv 默认不往 .venv 装 pip，而 Step 2/3/4 的 whl install 必须走
# `.venv/bin/python -m pip`（uv pip 会 resolve 到 PyPI 最新版、忽略本地 whl）
# ==============================================================================
if ! .venv/bin/python -c "import pip" 2>/dev/null; then
    info "Step 1.7: bootstrap pip..."
    uv pip install pip 2>&1 | tail -2
fi

# ==============================================================================
# Step 2: flash-attn (prebuilt wheel from afs, fallback to download)
# ==============================================================================
info "Step 2: flash-attn..."
if uv run --no-sync python3 -c "import flash_attn; print(f'flash-attn {flash_attn.__version__}')" 2>/dev/null; then
    ok "flash-attn 已安装"
else
    FA_WHL=$(ls ${AFS_WHEELS}/flash_attn-*cp310*linux_x86_64.whl 2>/dev/null | head -1)
    if [ -n "$FA_WHL" ]; then
        info "从 afs 安装: $(basename $FA_WHL)"
        .venv/bin/python -m pip install --force-reinstall --no-deps --no-build-isolation --no-index "$FA_WHL" 2>&1 | tail -3
    else
        FA_VERSION="2.8.3"
        TORCH_VER="2.8"
        WHEEL_NAME="flash_attn-${FA_VERSION}+cu128torch${TORCH_VER}-cp310-cp310-linux_x86_64.whl"
        WHEEL_URL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/${WHEEL_NAME}"
        WHEEL_PATH="/tmp/${WHEEL_NAME}"
        [ ! -f "$WHEEL_PATH" ] && wget -q --show-progress -O "$WHEEL_PATH" "$WHEEL_URL"
        .venv/bin/python -m pip install --force-reinstall --no-deps --no-build-isolation --no-index "$WHEEL_PATH" 2>&1 | tail -3
    fi
    uv run --no-sync python3 -c "import flash_attn; print(f'flash-attn {flash_attn.__version__}')" 2>/dev/null \
        && ok "flash-attn 安装完成" || warn "flash-attn 安装失败 (will use sdpa fallback)"
fi

# ==============================================================================
# Step 3: causal-conv1d (prebuilt wheel from afs, fallback to compile)
# ==============================================================================
info "Step 3: causal-conv1d..."
if uv run --no-sync python3 -c "import causal_conv1d; print(f'causal-conv1d {causal_conv1d.__version__}')" 2>/dev/null; then
    ok "causal-conv1d 已安装"
else
    CC_WHL=$(ls ${AFS_WHEELS}/causal_conv1d-*cp310*linux_x86_64.whl 2>/dev/null | head -1)
    if [ -n "$CC_WHL" ]; then
        info "从 afs 安装: $(basename $CC_WHL)"
        .venv/bin/python -m pip install --force-reinstall --no-deps --no-build-isolation --no-index "$CC_WHL" 2>&1 | tail -3
    else
        info "源码编译 causal-conv1d..."
        uv pip install causal-conv1d==1.6.1 --no-build-isolation --no-deps --reinstall 2>&1 | tail -5
    fi
    uv run --no-sync python3 -c "import causal_conv1d; print(f'causal-conv1d {causal_conv1d.__version__}')" 2>/dev/null \
        && ok "causal-conv1d 完成" || warn "causal-conv1d 安装失败"
fi

# ==============================================================================
# Step 4: mamba-ssm (prebuilt wheel from afs, fallback to compile)
# ==============================================================================
info "Step 4: mamba-ssm..."
if uv run --no-sync python3 -c "import mamba_ssm; print(f'mamba-ssm {mamba_ssm.__version__}')" 2>/dev/null; then
    ok "mamba-ssm 已安装"
else
    MS_WHL=$(ls ${AFS_WHEELS}/mamba_ssm-*cp310*linux_x86_64.whl 2>/dev/null | head -1)
    if [ -n "$MS_WHL" ]; then
        info "从 afs 安装: $(basename $MS_WHL)"
        .venv/bin/python -m pip install --force-reinstall --no-deps --no-build-isolation --no-index "$MS_WHL" 2>&1 | tail -3
    else
        info "源码编译 mamba-ssm..."
        uv pip install mamba-ssm --no-build-isolation --no-deps --reinstall 2>&1 | tail -5
    fi
    uv run --no-sync python3 -c "import mamba_ssm; print(f'mamba-ssm {mamba_ssm.__version__}')" 2>/dev/null \
        && ok "mamba-ssm 完成" || warn "mamba-ssm 安装失败"
fi

# ==============================================================================
# Step 5: flash-linear-attention (no prebuilt wheel, compile from source)
# ==============================================================================
info "Step 5: flash-linear-attention..."
if uv run --no-sync python3 -c "import fla" 2>/dev/null; then
    ok "flash-linear-attention 已安装"
else
    info "源码编译 flash-linear-attention..."
    uv pip install flash-linear-attention --no-build-isolation --no-deps --reinstall 2>&1 | tail -5
    uv run --no-sync python3 -c "import fla" 2>/dev/null \
        && ok "flash-linear-attention 完成" || warn "flash-linear-attention 安装失败"
fi

# ==============================================================================
# Step 5.5: [B卡 ONLY] 还原 torch patch
# ==============================================================================
# if [ "${CUDA_PATCHED:-false}" = true ] && [ -f "${CPP_EXT}.bak" ]; then
#     info "还原 torch CUDA patch..."
#     cp "${CPP_EXT}.bak" "$CPP_EXT"
#     rm "${CPP_EXT}.bak"
# fi

# ==============================================================================
# Step 6: NLTK data
# ==============================================================================
info "Step 6: NLTK 数据..."
uv run --no-sync python3 -c "
import nltk
for res in ['wordnet', 'omw-1.4', 'punkt_tab']:
    nltk.download(res, quiet=True)
print('NLTK data ready.')
"

# ==============================================================================
# Step 7: Optional - RaTEScore + medspacy
# ==============================================================================
if [ "$INSTALL_ALL" = true ]; then
    info "Step 7: 安装 RaTEScore + medspacy..."
    uv sync --inexact --extra medical-report
    uv pip install medspacy
    ok "Optional deps 安装完成"
fi

# ==============================================================================
# Verify
# ==============================================================================
info "验证环境..."
uv run --no-sync python3 -c "
import torch
print(f'torch={torch.__version__}, cuda={torch.version.cuda}')
try:
    print(f'GPU: {torch.cuda.get_device_name(0)}, count={torch.cuda.device_count()}')
except: print('GPU: not available')
import transformers; print(f'transformers={transformers.__version__}')
try: import flash_attn; print(f'flash_attn={flash_attn.__version__}')
except: print('flash_attn: NOT INSTALLED')
try: import causal_conv1d; print(f'causal_conv1d={causal_conv1d.__version__}')
except: print('causal_conv1d: NOT INSTALLED')
try: import mamba_ssm; print(f'mamba_ssm={mamba_ssm.__version__}')
except: print('mamba_ssm: NOT INSTALLED')
try: import fla; print('flash_linear_attention=OK')
except: print('flash_linear_attention: NOT INSTALLED')
import deepspeed; print(f'deepspeed={deepspeed.__version__}')
from lmms_eval.tasks._task_utils.report_metrics import calculate_bleu_4, calculate_rouge_l, calculate_meteor
print('lmms_eval core imports: OK')
"

echo ""
ok "环境就绪！"
