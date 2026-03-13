#!/usr/bin/env bash
# Start vLLM server for LLM Judge scoring.
# Usage: bash serve/start_judge_server.sh [MODEL] [PORT] [TP]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL="${1:-Qwen/Qwen3-VL-32B-Instruct}"
PORT="${2:-8000}"
TP="${3:-4}"

if [ ! -d ".venv" ]; then
    echo "[INFO] First run: creating serve environment..."
    uv venv .venv --python 3.10 --system-site-packages
    # Symlink the system vllm CLI into the venv
    ln -sf /usr/local/bin/vllm .venv/bin/vllm
fi

# Ensure vllm binary is symlinked (idempotent)
if [ ! -f ".venv/bin/vllm" ]; then
    ln -sf /usr/local/bin/vllm .venv/bin/vllm
fi

echo "[INFO] Starting vLLM server: model=$MODEL port=$PORT tp=$TP"
exec .venv/bin/vllm serve "$MODEL" \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --trust-remote-code \
    --max-model-len 4096 \
    --disable-custom-all-reduce
