#!/usr/bin/env bash
# ==============================================================================
# 全量重建 lmms-eval 环境（B卡 sm_100 + CUDA 13.1 + PyTorch cu128）
#
# 核心问题：系统 nvcc 是 CUDA 13.1，PyTorch 编译用 CUDA 12.8。
# torch.utils.cpp_extension._check_cuda_version() 会因 major 版本不同而报错。
# 解决方案：编译 CUDA 扩展前临时 patch 掉版本检查，编译后还原。
#
# 用法：bash scripts/rebuild_env.sh
# ==============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

info()  { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()    { echo -e "\033[1;32m[OK]\033[0m    $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
err()   { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; }

# Proxy
export http_proxy="${http_proxy:-http://cmcproxy:WvUBhef4bQ@10.251.112.50:8128}"
export https_proxy="${https_proxy:-http://cmcproxy:WvUBhef4bQ@10.251.112.50:8128}"

# CUDA 编译配置
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="9.0;10.0"  # H800 + B卡
export MAX_JOBS="${MAX_JOBS:-8}"

# ==============================================================================
# Step 1: 清理旧 venv，重新 sync
# ==============================================================================
info "Step 1: 清理旧 venv，重新 uv sync..."
rm -rf .venv
uv sync 2>&1 | tail -5
ok "uv sync 完成"

# ==============================================================================
# Step 2: Patch torch CUDA 版本检查
# ==============================================================================
CPP_EXT=".venv/lib/python3.10/site-packages/torch/utils/cpp_extension.py"
info "Step 2: 临时 patch torch CUDA 版本检查..."
cp "$CPP_EXT" "${CPP_EXT}.bak"
sed -i 's/            _check_cuda_version(compiler_name, compiler_version)/            pass  # _check_cuda_version PATCHED/' "$CPP_EXT"
ok "Patch 完成"

# ==============================================================================
# Step 3: 安装 flash-attn（预编译 wheel）
# ==============================================================================
info "Step 3: 安装 flash-attn..."
FA_VERSION="2.8.3"
WHEEL_NAME="flash_attn-${FA_VERSION}+cu128torch2.8-cp310-cp310-linux_x86_64.whl"
WHEEL_URL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/${WHEEL_NAME}"
WHEEL_PATH="/tmp/${WHEEL_NAME}"

if [ ! -f "$WHEEL_PATH" ]; then
    wget -q --show-progress -O "$WHEEL_PATH" "$WHEEL_URL" || {
        err "flash-attn wheel 下载失败"; WHEEL_PATH=""
    }
fi
if [ -n "$WHEEL_PATH" ] && [ -f "$WHEEL_PATH" ]; then
    uv pip install --no-deps "$WHEEL_PATH" 2>&1 | tail -3
    ok "flash-attn 安装完成"
else
    warn "跳过 flash-attn"
fi

# ==============================================================================
# Step 4: 重编译 causal-conv1d
# ==============================================================================
info "Step 4: 编译 causal-conv1d..."
uv pip install causal-conv1d==1.6.1 --no-build-isolation --no-deps --reinstall 2>&1 | tail -5
ok "causal-conv1d 完成"

# ==============================================================================
# Step 5: 重编译 flash-linear-attention
# ==============================================================================
info "Step 5: 编译 flash-linear-attention..."
uv pip install flash-linear-attention --no-build-isolation --no-deps --reinstall 2>&1 | tail -5
ok "flash-linear-attention 完成"

# ==============================================================================
# Step 6: 还原 torch patch
# ==============================================================================
info "Step 6: 还原 torch patch..."
cp "${CPP_EXT}.bak" "$CPP_EXT"
rm "${CPP_EXT}.bak"
ok "Patch 还原"

# ==============================================================================
# Step 7: NLTK 数据
# ==============================================================================
info "Step 7: NLTK 数据..."
.venv/bin/python -c "
import nltk
for res in ['wordnet', 'omw-1.4', 'punkt_tab']:
    nltk.download(res, quiet=True)
print('NLTK data ready.')
"
ok "NLTK 完成"

# ==============================================================================
# Step 8: 验证
# ==============================================================================
info "Step 8: 验证环境..."
.venv/bin/python -c "
import torch
print(f'torch={torch.__version__}, cuda={torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'arch_list: {torch.cuda.get_arch_list()}')

import transformers; print(f'transformers={transformers.__version__}')

try:
    import flash_attn; print(f'flash_attn={flash_attn.__version__}')
except: print('flash_attn: NOT INSTALLED')

try:
    import causal_conv1d; print(f'causal_conv1d={causal_conv1d.__version__}')
except: print('causal_conv1d: NOT INSTALLED')

try:
    import fla; print(f'flash_linear_attention=OK')
except: print('flash_linear_attention: NOT INSTALLED')

import deepspeed; print(f'deepspeed={deepspeed.__version__}')

from lmms_eval.tasks._task_utils.report_metrics import (
    calculate_bleu_4, calculate_rouge_l, calculate_meteor,
)
print('lmms_eval core imports: OK')
"

echo ""
ok "环境重建完成！"
echo "  运行评估: uv run python scripts/run_eval.py -c scripts/configs/med_eval_mini.yaml --models Qwen3-VL-2B-Instruct --limit 2"
