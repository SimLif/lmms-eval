#!/usr/bin/env bash

# Load credentials (proxy, tokens) from .secrets/env if available
SECRETS_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.secrets/env"
[ -f "$SECRETS_ENV" ] && source "$SECRETS_ENV"
# ============================================================================
# vLLM Judge Server Startup Script
# ============================================================================
#
# Starts a vLLM OpenAI-compatible server for LLM judge scoring on H800 GPUs.
# Manages an isolated Python environment via uv to avoid conflicts with the
# system-installed packages (which carry outdated deep_gemm, among others).
#
# Usage:
#   bash serve/start_judge_server.sh [MODEL] [PORT] [TP] [GPU_IDS]
#
# Examples:
#   # Single instance (GPUs 0-3)
#   bash serve/start_judge_server.sh Qwen/Qwen3-VL-32B-Instruct 8000 4 0,1,2,3
#
#   # Second instance on same node (GPUs 4-7) — must start AFTER instance 1
#   bash serve/start_judge_server.sh Qwen/Qwen3-VL-32B-Instruct 8001 4 4,5,6,7
#
# IMPORTANT: When running two instances on the same node, start them
# sequentially. Wait for instance 1's /health endpoint to return HTTP 200
# before launching instance 2. Simultaneous starts cause IPC resource
# contention that crashes the vLLM engine core.
# ============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- Parameters ---
MODEL="${1:-Qwen/Qwen3-VL-32B-Instruct}"
PORT="${2:-8000}"
TP="${3:-4}"
GPU_IDS="${4:-}"  # e.g. "0,1,2,3" or "4,5,6,7"

EXPECTED_VLLM="0.17.1"

# ============================================================================
# Model source resolution
# ============================================================================
#
# MODEL may be either:
#   (a) a HuggingFace hub id, e.g. "Qwen/Qwen3-VL-32B-Instruct"  — vLLM pulls
#       from ~/.cache/huggingface/hub/ (or downloads if online).
#   (b) an absolute filesystem path to a local snapshot, e.g.
#       "/root/.../models/Qwen3-VL-32B-Instruct"                   — vLLM reads
#       safetensors directly from that directory.
#
# When HF_HUB_OFFLINE=1 (we always set it below) and MODEL looks like a hub id
# that is NOT cached locally, vLLM would silently hang on .list_repo_files().
# Detect this case early and point the user at sync_judge_model.sh.

MODEL_IS_LOCAL=0
if [ -d "$MODEL" ]; then
    MODEL_IS_LOCAL=1
fi

# ============================================================================
# Environment variables
# ============================================================================

# HF_HUB_OFFLINE: Prevents huggingface_hub from making network requests.
# Without this, vLLM startup calls huggingface_hub.list_repo_files() which
# hangs indefinitely on offline nodes (or adds seconds of latency through
# the proxy on connected nodes). The model is fully cached locally.
export HF_HUB_OFFLINE=1

# If MODEL is a hub id (not a local dir), verify it's actually cached. The
# HuggingFace cache layout converts "Qwen/Qwen3-VL-32B-Instruct" → dir name
# "models--Qwen--Qwen3-VL-32B-Instruct". Skipping this check under offline
# mode means hanging on repo listing forever.
if [ "$MODEL_IS_LOCAL" = "0" ]; then
    HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}/hub"
    # transform "Qwen/Qwen3-VL-32B-Instruct" → "models--Qwen--Qwen3-VL-32B-Instruct"
    CACHE_SUBDIR="models--$(echo "$MODEL" | sed 's|/|--|g')"
    if [ ! -d "$HF_CACHE_DIR/$CACHE_SUBDIR" ]; then
        echo "[ERROR] MODEL='$MODEL' not cached at $HF_CACHE_DIR/$CACHE_SUBDIR" >&2
        echo "[ERROR] and HF_HUB_OFFLINE=1 would cause an infinite hang." >&2
        echo "[HINT]  Run scripts/sync_judge_model.sh first, then pass the local path:" >&2
        echo "[HINT]    bash scripts/sync_judge_model.sh" >&2
        echo "[HINT]    bash serve/start_judge_server.sh /root/.../models/Qwen3-VL-32B-Instruct" >&2
        exit 1
    fi
fi

# Network proxy: fallback for any outbound requests that slip through
# (e.g. a tokenizer file missing from local cache). The proxy is not
# relied upon — HF_HUB_OFFLINE=1 handles the common case.

# ============================================================================
# GPU selection
# ============================================================================

# CUDA_VISIBLE_DEVICES: isolates each vLLM instance to its own GPU set.
# Without this, both instances would see all 8 GPUs and collide on device 0.
if [ -n "$GPU_IDS" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
fi

# ============================================================================
# Python environment (always use uv sync)
# ============================================================================
#
# We ALWAYS build from the lockfile. This creates a clean venv with exactly
# the pinned vllm==0.17.1 and its dependencies, isolated from the system
# Python. This is critical because:
#
#   - The system Python (installed by the Ducc agent via `pip install vllm`)
#     carries an outdated deep_gemm that causes RuntimeError on Hopper GPUs.
#   - Using --system-site-packages would leak that broken package in.
#   - A cold `uv sync` takes ~47 seconds even with zero cache, which is
#     negligible for jobs that run for days.
#
# By never using --system-site-packages, we eliminate the DeepGEMM issue
# entirely without needing VLLM_USE_DEEP_GEMM=0 or VLLM_MOE_USE_DEEP_GEMM=0.

if [ ! -d ".venv" ]; then
    echo "[INFO] First run: creating isolated venv via uv sync..."
    uv sync
fi

# Version guard: if the venv has the wrong vllm version (e.g. leftover from
# a previous deployment), rebuild from scratch.
VLLM_VER=$(.venv/bin/python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "not found")
if [ "$VLLM_VER" != "$EXPECTED_VLLM" ]; then
    echo "[WARN] Expected vllm $EXPECTED_VLLM but found $VLLM_VER — rebuilding..."
    rm -rf .venv
    uv sync
    VLLM_VER="$EXPECTED_VLLM"
fi

# ============================================================================
# Build vLLM CLI arguments
# ============================================================================

VLLM_ARGS=(
    # --served-model-name: Makes the model accessible by its HuggingFace ID
    # (or absolute local path — same as the arg we pass to `vllm serve`) in
    # the OpenAI-compatible API. Without this, vLLM uses an internal name
    # and clients get 404 "model not found" errors.
    --served-model-name "$MODEL"

    --port "$PORT"

    # --tensor-parallel-size: Distributes the 32B model across 4 GPUs.
    # Each GPU holds ~71.8 GiB of the model — near the H800's 80 GiB limit.
    --tensor-parallel-size "$TP"

    --trust-remote-code

    # --max-model-len: Caps the KV cache allocation. 4096 is sufficient for
    # judge scoring (short prompts + short responses). Higher values waste
    # GPU memory that's already tight at 85% utilization.
    --max-model-len 4096

    # --disable-custom-all-reduce: Disables vLLM's custom all-reduce kernel
    # in favor of NCCL's default implementation. The custom kernel uses CUDA
    # IPC under the hood, which fails on our H800 nodes with a "CUDA IPC
    # not supported" error. This is a hardware/driver limitation, not a
    # software configuration issue.
    --disable-custom-all-reduce

    # --gpu-memory-utilization: Reserves 85% of GPU memory for the model and
    # KV cache. The remaining 15% provides headroom for CUDA context, NCCL
    # buffers, and transient allocations.
    --gpu-memory-utilization 0.85
)

# --compilation-config (second instance only):
# When two vLLM instances share a node, the FlashInfer compilation pass
# "fuse_allreduce_rms" binds a fixed IPC address for the AllReduce workspace.
# The first instance claims this address; the second cannot bind it. If the
# second instance loads a cached compiled graph containing the fused op, it
# crashes at runtime. Disabling this fusion pass for the second instance
# avoids the conflict. Cost: one-time recompilation (~10 min on first start).
if [ "$PORT" != "8000" ]; then
    VLLM_ARGS+=(--compilation-config '{"pass_config": {"fuse_allreduce_rms": false}}')
fi

# ============================================================================
# Launch
# ============================================================================

echo "[INFO] Starting vLLM server"
echo "[INFO]   model=$MODEL  port=$PORT  tp=$TP  vllm=$VLLM_VER"
[ "$MODEL_IS_LOCAL" = "1" ] && echo "[INFO]   source=LOCAL_PATH"
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] && echo "[INFO]   CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[INFO]   HF_HUB_OFFLINE=$HF_HUB_OFFLINE"

exec uv run vllm serve "$MODEL" "${VLLM_ARGS[@]}"
