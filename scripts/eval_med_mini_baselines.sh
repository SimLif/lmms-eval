#!/bin/bash
set -euo pipefail

# Med-eval-mini benchmark for 5 medical baseline models
# Usage: bash scripts/eval_med_mini_baselines.sh
#
# Models:
#   1. HuatuoGPT-Vision-7B   (qwen2_5_vl, 1 GPU)
#   2. MedGemma-4b-it         (gemma3, 1 GPU)
#   3. LLaVA-Med-v1.5         (llava_hf, 1 GPU)
#   4. Med-MoE-Phi 2.7B       (med_moe_phi, 1 GPU, deepspeed)
#   5. Med-MoE-StableLM 1.6B  (med_moe_stablelm, 1 GPU, deepspeed)
#
# All 5 models run in parallel on separate GPUs (0-4).

export LD_LIBRARY_PATH=/root/paddlejob/workspace/env_run/hqguo/lmms-eval/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN environment variable must be set}"
export LMMS_EVAL_USE_CACHE=True
export http_proxy="${http_proxy:-}"
export https_proxy="${https_proxy:-}"

WORKDIR="/root/paddlejob/workspace/env_run/hqguo/lmms-eval"
TASKS="med_eval_mini"
OUTPUT_DIR="logs/med_eval_mini"
LOG_DIR="logs"

cd "$WORKDIR"
mkdir -p "$OUTPUT_DIR"

# Optional: set LIMIT=N for quick testing
LIMIT_ARGS=""
if [ -n "${LIMIT:-}" ]; then
    LIMIT_ARGS="--limit $LIMIT"
fi

echo "========================================="
echo "Med-eval-mini Baseline Evaluation"
echo "Tasks: $TASKS"
echo "Time:  $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="

# --- 1. HuatuoGPT-Vision-7B (GPU 0) ---
echo "[1/5] Launching HuatuoGPT-Vision-7B on GPU 0..."
CUDA_VISIBLE_DEVICES=0 uv run python -m lmms_eval \
    --model qwen2_5_vl \
    --model_args "pretrained=FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL" \
    --tasks "$TASKS" \
    --batch_size 1 \
    --log_samples \
    --output_path "${OUTPUT_DIR}/HuatuoGPT-Vision-7B" \
    $LIMIT_ARGS \
    > "${LOG_DIR}/baseline_huatuogpt.log" 2>&1 &
PID_HUATUO=$!

# --- 2. MedGemma-4b-it (GPU 1) ---
echo "[2/5] Launching MedGemma-4b-it on GPU 1..."
CUDA_VISIBLE_DEVICES=1 uv run python -m lmms_eval \
    --model gemma3 \
    --model_args "pretrained=google/medgemma-4b-it" \
    --tasks "$TASKS" \
    --batch_size 1 \
    --log_samples \
    --output_path "${OUTPUT_DIR}/MedGemma-4b-it" \
    $LIMIT_ARGS \
    > "${LOG_DIR}/baseline_medgemma.log" 2>&1 &
PID_MEDGEMMA=$!

# --- 3. LLaVA-Med-v1.5 (GPU 2) ---
echo "[3/5] Launching LLaVA-Med-v1.5 on GPU 2..."
CUDA_VISIBLE_DEVICES=2 uv run python -m lmms_eval \
    --model llava_hf \
    --model_args "pretrained=chaoyinshe/llava-med-v1.5-mistral-7b-hf" \
    --tasks "$TASKS" \
    --batch_size 1 \
    --log_samples \
    --output_path "${OUTPUT_DIR}/LLaVA-Med-v1.5" \
    $LIMIT_ARGS \
    > "${LOG_DIR}/baseline_llava_med.log" 2>&1 &
PID_LLAVA=$!

# --- 4. Med-MoE-Phi 2.7B (GPU 3, single-process with deepspeed) ---
echo "[4/5] Launching Med-MoE-Phi on GPU 3..."
CUDA_VISIBLE_DEVICES=3 uv run python -m lmms_eval \
    --model med_moe_phi \
    --model_args "pretrained=/root/paddlejob/workspace/env_run/hqguo/models/Med-MoE/stage3/llavaphi-2.7b-medmoe" \
    --tasks "$TASKS" \
    --batch_size 1 \
    --log_samples \
    --output_path "${OUTPUT_DIR}/Med-MoE-Phi" \
    $LIMIT_ARGS \
    > "${LOG_DIR}/baseline_med_moe_phi.log" 2>&1 &
PID_MOE_PHI=$!

# --- 5. Med-MoE-StableLM 1.6B (GPU 4, single-process with deepspeed) ---
echo "[5/5] Launching Med-MoE-StableLM on GPU 4..."
CUDA_VISIBLE_DEVICES=4 uv run python -m lmms_eval \
    --model med_moe_stablelm \
    --model_args "pretrained=/root/paddlejob/workspace/env_run/hqguo/models/Med-MoE/stage3/llavastablelm-1.6b-medmoe" \
    --tasks "$TASKS" \
    --batch_size 1 \
    --log_samples \
    --output_path "${OUTPUT_DIR}/Med-MoE-StableLM" \
    $LIMIT_ARGS \
    > "${LOG_DIR}/baseline_med_moe_stablelm.log" 2>&1 &
PID_MOE_SLM=$!

echo ""
echo "All 5 models launched in parallel."
echo "PIDs: HuatuoGPT=$PID_HUATUO MedGemma=$PID_MEDGEMMA LLaVA-Med=$PID_LLAVA MoE-Phi=$PID_MOE_PHI MoE-StableLM=$PID_MOE_SLM"
echo "Logs: ${LOG_DIR}/baseline_*.log"
echo ""
echo "Waiting for all models to complete..."

# Wait for all and report status
FAILED=0
for name_pid in "HuatuoGPT:$PID_HUATUO" "MedGemma:$PID_MEDGEMMA" "LLaVA-Med:$PID_LLAVA" "MoE-Phi:$PID_MOE_PHI" "MoE-StableLM:$PID_MOE_SLM"; do
    IFS=':' read -r name pid <<< "$name_pid"
    if wait "$pid"; then
        echo "  ✓ $name (PID $pid) completed successfully"
    else
        echo "  ✗ $name (PID $pid) FAILED (exit code $?)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
if [ $FAILED -eq 0 ]; then
    echo "All 5 baseline models completed successfully!"
else
    echo "$FAILED model(s) failed. Check logs for details."
fi
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
