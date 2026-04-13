#!/bin/bash
# DEPRECATED: Use run_eval.py instead:
#   uv run python scripts/run_eval.py -c scripts/configs/med_eval_mini.yaml --task med_eval
set -euo pipefail

# Med-eval benchmark for multiple models on 8 GPUs
# Usage: bash scripts/eval_med_multi.sh
# Stops immediately if any model evaluation fails.

export LD_LIBRARY_PATH=/root/paddlejob/workspace/env_run/hqguo/lmms-eval/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN environment variable must be set}"
export LMMS_EVAL_USE_CACHE=True

# Proxy for HuggingFace Hub access (set via environment)
export http_proxy="${http_proxy:-}"
export https_proxy="${https_proxy:-}"

WORKDIR="/root/paddlejob/workspace/env_run/hqguo/lmms-eval"
TASKS="med_eval"
NUM_GPUS=8
PORT=12346
MAX_PIXELS=3211264  # 576 * 28 * 28
ATTN_IMPL=sdpa
OUTPUT_DIR="logs/med_eval"
LOG_FILE="logs/med_eval.log"

# Default package versions (restored after version switches)
DEFAULT_TRANSFORMERS_VER="4.57.6"
DEFAULT_VLLM_VER="0.11.0"

# Launch modes:
#   multi  = accelerate data-parallel (HF, small models fit 1 GPU)
#   single = single process device_map=auto (HF, large models)
#   vllm   = vLLM tensor-parallel across all GPUs
#
# Format: "MODEL_TYPE|PRETRAINED|LAUNCH_MODE|TRANSFORMERS_VER|EXTRA_ARGS"
#   TRANSFORMERS_VER is optional; omit or leave empty to use DEFAULT_TRANSFORMERS_VER.
#   EXTRA_ARGS is optional; comma-separated key=value pairs:
#     - enable_thinking=True → appended to model_args; output dir gets "-thinking" suffix;
#       cache uses separate "-thinking" namespace and is cleaned up after eval
#     - remaining keys (temperature, top_p, max_new_tokens, ...) → passed via --gen_kwargs
#   InternVL2/2.5 remote code requires transformers <4.50 (GenerationMixin removal).
MODELS=(
    # --- HF backends (already evaluated) ---
    # "qwen3_vl|Qwen/Qwen3-VL-2B-Instruct|multi"
    # "qwen3_vl|Qwen/Qwen3-VL-4B-Instruct|multi"
    # "qwen2_5_vl|Qwen/Qwen2.5-VL-7B-Instruct|multi"
    # "qwen3_vl|Qwen/Qwen3-VL-8B-Instruct|multi"
    # "qwen3_vl|Qwen/Qwen3-VL-32B-Instruct|single"
    # "qwen3_vl|Qwen/Qwen3-VL-30B-A3B-Instruct|single"

    # --- vLLM backends ---
    # "vllm|Qwen/Qwen3-VL-32B-Instruct|vllm"
    # "vllm|Qwen/Qwen3-VL-30B-A3B-Instruct|vllm"

    # --- InternVL backends (HF, 8B fits single GPU → multi data parallel) ---
    # "internvl2|OpenGVLab/InternVL2_5-8B|multi|4.49.0"
    # "internvl3|OpenGVLab/InternVL3-8B|multi"
    # "internvl3_5|OpenGVLab/InternVL3_5-8B|multi"

    # --- Qwen3.5 non-thinking (already evaluated) ---
    # "qwen3_5|Qwen/Qwen3.5-2B|multi|5.2.0"
    # "qwen3_5|Qwen/Qwen3.5-4B|multi|5.2.0"
    # "qwen3_5|Qwen/Qwen3.5-9B|multi|5.2.0"

    # --- Qwen3.5 thinking (official recommended: temp=1.0, top_p=0.95) ---
    "qwen3_5|Qwen/Qwen3.5-2B|multi|5.2.0|enable_thinking=True,max_thinking_tokens=512,temperature=1.0,top_p=0.95,max_new_tokens=1024"
    # "qwen3_5|Qwen/Qwen3.5-4B|multi|5.2.0|enable_thinking=True,max_thinking_tokens=512,temperature=1.0,top_p=0.95,max_new_tokens=1024"
    # "qwen3_5|Qwen/Qwen3.5-9B|multi|5.2.0|enable_thinking=True,max_thinking_tokens=512,temperature=1.0,top_p=0.95,max_new_tokens=1024"
)

# vLLM-specific defaults
VLLM_TP=${VLLM_TP:-$NUM_GPUS}
VLLM_GPU_UTIL=${VLLM_GPU_UTIL:-0.80}
VLLM_BATCH_SIZE=${VLLM_BATCH_SIZE:-32}
HF_BATCH_SIZE=${HF_BATCH_SIZE:-32}

# Optional: set LIMIT=N for quick testing (e.g. LIMIT=1 bash scripts/eval_med_multi.sh)
LIMIT_ARGS=""
if [ -n "${LIMIT:-}" ]; then
    LIMIT_ARGS="--limit $LIMIT"
fi

cd "$WORKDIR"
mkdir -p "$OUTPUT_DIR"

# Truncate log on each fresh run
: > "$LOG_FILE"

# ---------------------------------------------------------------------------
# Helper: ensure the installed transformers version matches the requested one.
# vllm 0.11.0 requires transformers >=4.55.2, so when downgrading transformers
# we loosen the vllm pin to let the resolver find a compatible version, then
# restore both to their defaults when switching back.
# ---------------------------------------------------------------------------
CURRENT_TF_VER=""

ensure_transformers_ver() {
    local want="$1"
    if [ -z "$want" ]; then
        want="$DEFAULT_TRANSFORMERS_VER"
    fi
    if [ "$CURRENT_TF_VER" = "$want" ]; then
        return
    fi
    echo "[env] Switching transformers to $want ..." | tee -a "$LOG_FILE"
    if [ "$want" = "$DEFAULT_TRANSFORMERS_VER" ]; then
        # Restore both transformers and vllm to defaults
        uv add "transformers==$want" "vllm==$DEFAULT_VLLM_VER" 2>&1 | tee -a "$LOG_FILE"
    else
        # Loosen vllm pin so resolver can find a compatible version
        uv add "transformers==$want" "vllm>=0.1" 2>&1 | tee -a "$LOG_FILE"
    fi
    CURRENT_TF_VER="$want"
}

# Detect the currently installed version once at startup
CURRENT_TF_VER=$(uv run python -c "import transformers; print(transformers.__version__)" 2>/dev/null)
echo "[env] Current transformers: $CURRENT_TF_VER" | tee -a "$LOG_FILE"

TOTAL=${#MODELS[@]}
DONE=0

for entry in "${MODELS[@]}"; do
    IFS='|' read -r MODEL_TYPE PRETRAINED LAUNCH_MODE TF_VER EXTRA_ARGS <<< "$entry"
    LAUNCH_MODE="${LAUNCH_MODE:-multi}"
    TF_VER="${TF_VER:-}"
    EXTRA_ARGS="${EXTRA_ARGS:-}"
    MODEL_NAME=$(basename "$PRETRAINED")

    # --- Parse EXTRA_ARGS: split model args from gen_kwargs ---
    EXTRA_MODEL_ARGS=""
    GEN_KWARGS_ARGS=""
    THINKING_MODE=false
    if [ -n "$EXTRA_ARGS" ]; then
        IFS=',' read -ra PARTS <<< "$EXTRA_ARGS"
        model_parts=()
        gen_parts=()
        for part in "${PARTS[@]}"; do
            if [ "$part" = "enable_thinking=True" ]; then
                model_parts+=("enable_thinking=True")
                THINKING_MODE=true
            elif [[ "$part" == max_thinking_tokens=* ]]; then
                model_parts+=("$part")
            else
                gen_parts+=("$part")
            fi
        done
        EXTRA_MODEL_ARGS=$(IFS=','; echo "${model_parts[*]+"${model_parts[*]}"}")
        GEN_KWARGS_ARGS=$(IFS=','; echo "${gen_parts[*]+"${gen_parts[*]}"}")
    fi

    # Thinking mode: adjust output dir suffix
    if [ "$THINKING_MODE" = true ]; then
        MODEL_NAME="${MODEL_NAME}-thinking"
    fi
    MODEL_OUTPUT_DIR="${OUTPUT_DIR}/${MODEL_NAME}"

    GEN_KWARGS_CLI=""
    if [ -n "$GEN_KWARGS_ARGS" ]; then
        GEN_KWARGS_CLI="--gen_kwargs $GEN_KWARGS_ARGS"
    fi

    {
        echo ""
        echo "========================================="
        echo "[$((DONE+1))/$TOTAL] Model: $PRETRAINED"
        echo "Type:   $MODEL_TYPE"
        echo "Launch: $LAUNCH_MODE"
        echo "TF ver: ${TF_VER:-$DEFAULT_TRANSFORMERS_VER (default)}"
        echo "Output: $MODEL_OUTPUT_DIR"
        echo "GPUs:   $NUM_GPUS"
        if [ -n "$EXTRA_ARGS" ]; then
            echo "Extra:  $EXTRA_ARGS"
        fi
        echo "Time:   $(date '+%Y-%m-%d %H:%M:%S')"
        echo "========================================="
    } | tee -a "$LOG_FILE"

    # Switch transformers version if needed
    ensure_transformers_ver "${TF_VER:-$DEFAULT_TRANSFORMERS_VER}"

    if [ "$LAUNCH_MODE" = "vllm" ]; then
        # vLLM tensor-parallel: single process, model sharded across GPUs
        MODEL_ARGS="model=${PRETRAINED},tensor_parallel_size=${VLLM_TP},gpu_memory_utilization=${VLLM_GPU_UTIL},disable_log_stats=True,mm_processor_kwargs={\"max_pixels\":${MAX_PIXELS}}"
        uv run python -m lmms_eval \
            --model vllm \
            --model_args "$MODEL_ARGS" \
            --tasks "$TASKS" \
            --batch_size "$VLLM_BATCH_SIZE" \
            --log_samples \
            --output_path "$MODEL_OUTPUT_DIR" \
            $GEN_KWARGS_CLI \
            $LIMIT_ARGS \
            2>&1 | tee -a "$LOG_FILE"
    elif [ "$LAUNCH_MODE" = "single" ]; then
        # Single process: model uses device_map=auto to span all GPUs
        MODEL_ARGS="pretrained=${PRETRAINED},max_pixels=${MAX_PIXELS},attn_implementation=${ATTN_IMPL}"
        [ -n "$EXTRA_MODEL_ARGS" ] && MODEL_ARGS="${MODEL_ARGS},${EXTRA_MODEL_ARGS}"
        uv run python -m lmms_eval \
            --model "$MODEL_TYPE" \
            --model_args "$MODEL_ARGS" \
            --tasks "$TASKS" \
            --batch_size "$HF_BATCH_SIZE" \
            --log_samples \
            --output_path "$MODEL_OUTPUT_DIR" \
            $GEN_KWARGS_CLI \
            $LIMIT_ARGS \
            2>&1 | tee -a "$LOG_FILE"
    else
        # Multi-GPU data parallel via accelerate
        MODEL_ARGS="pretrained=${PRETRAINED},max_pixels=${MAX_PIXELS},attn_implementation=${ATTN_IMPL}"
        [ -n "$EXTRA_MODEL_ARGS" ] && MODEL_ARGS="${MODEL_ARGS},${EXTRA_MODEL_ARGS}"
        uv run accelerate launch \
            --num_processes=$NUM_GPUS \
            --main_process_port=$PORT \
            -m lmms_eval \
            --model "$MODEL_TYPE" \
            --model_args "$MODEL_ARGS" \
            --tasks "$TASKS" \
            --batch_size "$HF_BATCH_SIZE" \
            --log_samples \
            --output_path "$MODEL_OUTPUT_DIR" \
            $GEN_KWARGS_CLI \
            $LIMIT_ARGS \
            2>&1 | tee -a "$LOG_FILE"
    fi

    # pipefail ensures tee does not mask the exit code
    echo "[DONE] $PRETRAINED  $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
    DONE=$((DONE + 1))
done

# Restore default transformers version if we switched away
ensure_transformers_ver "$DEFAULT_TRANSFORMERS_VER"

{
    echo ""
    echo "========================================="
    echo "All $TOTAL evaluations complete!"
    echo "========================================="
} | tee -a "$LOG_FILE"
