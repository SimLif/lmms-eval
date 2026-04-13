#!/usr/bin/env bash
# ==============================================================================
# One-shot: 启动 judge server + 运行 posthoc LLM judge + 关闭 server
#
# Usage:
#   bash scripts/run_posthoc_judge.sh <results_dir> [GPU_IDS] [PORT] [WORKERS]
#
# Example:
#   bash scripts/run_posthoc_judge.sh logs/med_eval_mini/Qwen3-VL-2B-Instruct/Qwen__Qwen3-VL-2B-Instruct
#   bash scripts/run_posthoc_judge.sh logs/med_eval_mini/Qwen3-VL-2B-Instruct/Qwen__Qwen3-VL-2B-Instruct 0,1,2,3 8000 32
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

RESULTS_DIR="${1:?Usage: $0 <results_dir> [GPU_IDS] [PORT] [WORKERS]}"
GPU_IDS="${2:-0,1,2,3}"
PORT="${3:-8000}"
WORKERS="${4:-32}"
JUDGE_MODEL="Qwen/Qwen3-VL-32B-Instruct"
TP=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)

info()  { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()    { echo -e "\033[1;32m[OK]\033[0m    $*"; }
err()   { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; }

# Proxy
export http_proxy="${http_proxy:-http://cmcproxy:WvUBhef4bQ@10.251.112.50:8128}"
export https_proxy="${https_proxy:-http://cmcproxy:WvUBhef4bQ@10.251.112.50:8128}"

# ==============================================================================
# Step 1: Start judge server
# ==============================================================================
info "Starting judge server: model=$JUDGE_MODEL tp=$TP port=$PORT gpus=$GPU_IDS"

# Launch in background, capture PID
bash serve/start_judge_server.sh "$JUDGE_MODEL" "$PORT" "$TP" "$GPU_IDS" > /tmp/judge_server.log 2>&1 &
SERVER_PID=$!

# Cleanup on exit
cleanup() {
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        info "Shutting down judge server (PID=$SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null
        wait "$SERVER_PID" 2>/dev/null || true
        ok "Server stopped."
    fi
}
trap cleanup EXIT

# Wait for server to be ready (health check)
info "Waiting for server to be ready..."
MAX_WAIT=600  # 10 minutes max
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PORT}/health" 2>/dev/null | grep -q "200"; then
        ok "Server is ready (${WAITED}s)"
        break
    fi
    # Check if server process died
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        err "Server process died. Check /tmp/judge_server.log"
        tail -20 /tmp/judge_server.log
        exit 1
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    if [ $((WAITED % 30)) -eq 0 ]; then
        info "  Still waiting... (${WAITED}s)"
    fi
done

if [ $WAITED -ge $MAX_WAIT ]; then
    err "Server failed to start within ${MAX_WAIT}s"
    tail -20 /tmp/judge_server.log
    exit 1
fi

# ==============================================================================
# Step 2: Run posthoc judge
# ==============================================================================
info "Running posthoc LLM judge on: $RESULTS_DIR"
info "  Judge model: $JUDGE_MODEL"
info "  Workers: $WORKERS"

.venv/bin/python scripts/posthoc_llm_judge.py \
    --results_dir "$RESULTS_DIR" \
    --run_judge \
    --judge_api_type openai \
    --judge_api_url "http://localhost:${PORT}/v1" \
    --judge_model "$JUDGE_MODEL" \
    --workers "$WORKERS"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    ok "Posthoc judge complete!"
else
    err "Posthoc judge failed (exit=$EXIT_CODE)"
fi

# Server is stopped by trap cleanup
exit $EXIT_CODE
