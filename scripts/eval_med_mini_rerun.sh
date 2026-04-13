#!/bin/bash
# DEPRECATED: Use run_eval.py instead:
#   uv run python scripts/run_eval.py -c scripts/configs/med_eval_mini.yaml --skip-if-done --batch-size 1
set -euo pipefail

# Re-run only the 8 failed models for med_eval_mini
# Fixes: reduced batch_size for Qwen3-VL (16), batch_size=1 for InternVL

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=/root/paddlejob/workspace/env_run/hqguo/lmms-eval/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN environment variable must be set}"
export LMMS_EVAL_USE_CACHE=True

# Proxy for HuggingFace Hub access (set via environment)
export http_proxy="${http_proxy:-}"
export https_proxy="${https_proxy:-}"

WORKDIR="/root/paddlejob/workspace/env_run/hqguo/lmms-eval"
TASKS="med_eval_mini"
NUM_GPUS=8
PORT=12346
MAX_PIXELS=3211264
ATTN_IMPL=sdpa
OUTPUT_DIR="logs/med_eval_mini"
LOG_FILE="logs/med_eval_mini_rerun.log"

DEFAULT_TRANSFORMERS_VER="4.57.6"
DEFAULT_VLLM_VER="0.11.0"

# Format: "MODEL_TYPE|PRETRAINED|LAUNCH_MODE|TF_VER|EXTRA_ARGS|BATCH_SIZE"
MODELS=(
    # All use batch_size=1 to avoid OOM on large medical images
    "qwen3_vl|Qwen/Qwen3-VL-2B-Instruct|multi|||1"
    "qwen3_vl|Qwen/Qwen3-VL-4B-Instruct|multi|||1"
    "qwen3_vl|Qwen/Qwen3-VL-8B-Instruct|multi|||1"
    "vllm|Qwen/Qwen3-VL-30B-A3B-Instruct|vllm|||32"
    "vllm|Qwen/Qwen3-VL-32B-Instruct|vllm|||32"
    "internvl2|OpenGVLab/InternVL2_5-8B|multi|4.49.0||1"
    "internvl3|OpenGVLab/InternVL3-8B|multi|||1"
    "internvl3_5|OpenGVLab/InternVL3_5-8B|multi|||1"
)

VLLM_TP=${VLLM_TP:-$NUM_GPUS}
VLLM_GPU_UTIL=${VLLM_GPU_UTIL:-0.80}

cd "$WORKDIR"
mkdir -p "$OUTPUT_DIR"
: > "$LOG_FILE"

CURRENT_TF_VER=""
ensure_transformers_ver() {
    local want="$1"
    [ -z "$want" ] && want="$DEFAULT_TRANSFORMERS_VER"
    [ "$CURRENT_TF_VER" = "$want" ] && return
    echo "[env] Switching transformers to $want ..." | tee -a "$LOG_FILE"
    if [ "$want" = "$DEFAULT_TRANSFORMERS_VER" ]; then
        uv add "transformers==$want" "vllm==$DEFAULT_VLLM_VER" 2>&1 | tee -a "$LOG_FILE"
    else
        uv add "transformers==$want" "vllm>=0.1" 2>&1 | tee -a "$LOG_FILE"
    fi
    CURRENT_TF_VER="$want"
}

CURRENT_TF_VER=$(uv run python -c "import transformers; print(transformers.__version__)" 2>/dev/null)
echo "[env] Current transformers: $CURRENT_TF_VER" | tee -a "$LOG_FILE"

TOTAL=${#MODELS[@]}
DONE=0

for entry in "${MODELS[@]}"; do
    IFS='|' read -r MODEL_TYPE PRETRAINED LAUNCH_MODE TF_VER EXTRA_ARGS BATCH_SIZE <<< "$entry"
    LAUNCH_MODE="${LAUNCH_MODE:-multi}"
    TF_VER="${TF_VER:-}"
    EXTRA_ARGS="${EXTRA_ARGS:-}"
    BATCH_SIZE="${BATCH_SIZE:-32}"
    MODEL_NAME=$(basename "$PRETRAINED")
    MODEL_OUTPUT_DIR="${OUTPUT_DIR}/${MODEL_NAME}"

    # Skip if results already exist
    if find "$MODEL_OUTPUT_DIR" -name "*_results.json" 2>/dev/null | grep -q .; then
        echo "[SKIP] $PRETRAINED — results already exist" | tee -a "$LOG_FILE"
        DONE=$((DONE + 1))
        continue
    fi

    {
        echo ""
        echo "========================================="
        echo "[$((DONE+1))/$TOTAL] Model: $PRETRAINED"
        echo "Batch:  $BATCH_SIZE"
        echo "Launch: $LAUNCH_MODE"
        echo "Time:   $(date '+%Y-%m-%d %H:%M:%S')"
        echo "========================================="
    } | tee -a "$LOG_FILE"

    ensure_transformers_ver "${TF_VER:-$DEFAULT_TRANSFORMERS_VER}"

    MODEL_ARGS="pretrained=${PRETRAINED},max_pixels=${MAX_PIXELS},attn_implementation=${ATTN_IMPL}"

    if [ "$LAUNCH_MODE" = "vllm" ]; then
        VLLM_ARGS="model=${PRETRAINED},tensor_parallel_size=${VLLM_TP},gpu_memory_utilization=${VLLM_GPU_UTIL},disable_log_stats=True,mm_processor_kwargs={\"max_pixels\":${MAX_PIXELS}}"
        uv run python -m lmms_eval \
            --model vllm \
            --model_args "$VLLM_ARGS" \
            --tasks "$TASKS" \
            --batch_size "$BATCH_SIZE" \
            --log_samples \
            --output_path "$MODEL_OUTPUT_DIR" \
            2>&1 | tee -a "$LOG_FILE"
    elif [ "$LAUNCH_MODE" = "single" ]; then
        uv run python -m lmms_eval \
            --model "$MODEL_TYPE" \
            --model_args "$MODEL_ARGS" \
            --tasks "$TASKS" \
            --batch_size "$BATCH_SIZE" \
            --log_samples \
            --output_path "$MODEL_OUTPUT_DIR" \
            2>&1 | tee -a "$LOG_FILE"
    else
        uv run accelerate launch \
            --num_processes=$NUM_GPUS \
            --main_process_port=$PORT \
            -m lmms_eval \
            --model "$MODEL_TYPE" \
            --model_args "$MODEL_ARGS" \
            --tasks "$TASKS" \
            --batch_size "$BATCH_SIZE" \
            --log_samples \
            --output_path "$MODEL_OUTPUT_DIR" \
            2>&1 | tee -a "$LOG_FILE"
    fi

    echo "[DONE] $PRETRAINED  $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
    DONE=$((DONE + 1))
done

ensure_transformers_ver "$DEFAULT_TRANSFORMERS_VER"

{
    echo ""
    echo "========================================="
    echo "All $TOTAL re-run evaluations complete!"
    echo "========================================="
} | tee -a "$LOG_FILE"
