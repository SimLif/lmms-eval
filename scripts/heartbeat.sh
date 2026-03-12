#!/bin/bash
# Heartbeat monitor for eval_med_multi.sh
# Usage: bash scripts/heartbeat.sh [interval_seconds]
# Default interval: 120s (2 min)

INTERVAL=${1:-120}
LOG="/root/paddlejob/workspace/env_run/hqguo/lmms-eval/logs/eval_med_multi.log"
SCRIPT_NAME="eval_med_multi"

echo "=== Heartbeat Monitor Started (interval=${INTERVAL}s) ==="
echo ""

while true; do
    TS=$(date '+%Y-%m-%d %H:%M:%S')

    # Check if process alive
    PID=$(pgrep -f "$SCRIPT_NAME" | head -1)
    if [[ -z "$PID" ]]; then
        echo "[$TS] PROCESS EXITED. Evaluation finished or crashed."
        # Show last few meaningful lines
        echo "--- Last output ---"
        grep -E "DONE|Error|accuracy|Traceback|All evaluations" "$LOG" | tail -5
        break
    fi

    # Extract current model from log (last ========= block)
    MODEL=$(grep "^Model:" "$LOG" | tail -1 | sed 's/Model: //')

    # Extract progress bar
    PROGRESS=$(grep -oP 'Model Responding:\s+\K\d+%\|[^|]*\|\s*\d+/\d+\s*\[.*?\]' "$LOG" | tail -1)
    if [[ -z "$PROGRESS" ]]; then
        PROGRESS=$(grep -oP '\d+/\d+\s*\[' "$LOG" | tail -1 | tr -d '[')
    fi

    # Extract speed
    SPEED=$(grep -oP '\d+\.\d+it/s' "$LOG" | tail -1)

    # GPU memory (compact)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | paste -sd'/' -)

    # GPU utilization (average)
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1; n++} END {printf "%.0f", s/n}')

    # Completed models
    DONE_COUNT=$(grep -c "^\[DONE\]" "$LOG" 2>/dev/null || echo 0)

    echo "[$TS] model=${DONE_COUNT}/4 done | current: ${MODEL:-loading...} | progress: ${PROGRESS:-building...} | speed: ${SPEED:-n/a} | gpu_mem: ${GPU_MEM} MiB | gpu_util: ${GPU_UTIL}%"

    sleep "$INTERVAL"
done

echo ""
echo "=== Heartbeat Monitor Stopped ==="
