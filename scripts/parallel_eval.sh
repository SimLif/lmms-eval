#!/bin/bash
# parallel_eval.sh — Run lmms_eval across multiple GPUs via task-level sharding.
#
# Each GPU runs a disjoint subset of tasks. Avoids per-task offset issues.
#
# Usage:
#   bash scripts/parallel_eval.sh <model_name> [num_gpus] [config]
#
# Example:
#   bash scripts/parallel_eval.sh Med-MoE-Phi 8 scripts/configs/med_eval_mini.yaml

set -euo pipefail

MODEL="$1"
NUM_GPUS="${2:-8}"
CONFIG="${3:-scripts/configs/med_eval_mini.yaml}"
BASE_OUTPUT="logs/med_eval_mini/${MODEL}"

# Read model config via env vars (no shell injection)
read -r MODEL_TYPE PRETRAINED <<< "$(
    _PE_MODEL="$MODEL" _PE_CONFIG="$CONFIG" python3 -c "
import yaml, os
cfg = yaml.safe_load(open(os.environ['_PE_CONFIG']))
model = os.environ['_PE_MODEL']
for m in cfg.get('models', []):
    if m['name'] == model:
        print(m.get('model_type', ''), m.get('pretrained', '')); break
")"

if [ -z "$MODEL_TYPE" ] || [ -z "$PRETRAINED" ]; then
    echo "ERROR: model '$MODEL' not found in $CONFIG"
    exit 1
fi

# Bin-pack tasks into GPUs
TASK_ASSIGNMENTS="$(
    _PE_CONFIG="$CONFIG" _PE_NUM_GPUS="$NUM_GPUS" python3 -c "
import yaml, json, os, sys
cfg = yaml.safe_load(open(os.environ['_PE_CONFIG']))
task_name = cfg.get('task', cfg.get('tasks', 'med_eval_mini'))
N = int(os.environ['_PE_NUM_GPUS'])

try:
    from lmms_eval.tasks import get_task_dict, initialize_tasks
    initialize_tasks()
    task_dict = get_task_dict([task_name])
    tasks = {}
    def collect(d):
        for k, v in d.items():
            if hasattr(v, 'test_docs') and v.has_test_docs():
                tasks[k] = len(v.test_docs())
            elif hasattr(v, 'items'):
                collect(v)
    collect(task_dict)
except Exception as e:
    print(f'WARNING: task introspection failed ({e}), using fallback', file=sys.stderr)
    tasks = {'med_eval_mini': 12105}

bins = [[] for _ in range(N)]
sizes = [0] * N
for t, n in sorted(tasks.items(), key=lambda x: -x[1]):
    idx = sizes.index(min(sizes))
    bins[idx].append(t)
    sizes[idx] += n
result = [{'gpu': i, 'tasks': b, 'samples': s} for i, (b, s) in enumerate(zip(bins, sizes)) if b]
print(json.dumps(result))
")"

echo "=== Parallel Eval (task-level sharding) ==="
echo "  Model:      $MODEL ($MODEL_TYPE)"
echo "  Pretrained: $PRETRAINED"
echo "  GPUs:       $NUM_GPUS"
echo "  Config:     $CONFIG"
echo

echo "$TASK_ASSIGNMENTS" | python3 -c "
import json, sys
for a in json.loads(sys.stdin.read()):
    print(f\"  GPU {a['gpu']:d}: {a['samples']:5d} samples — {','.join(a['tasks'])}\")"
echo

PIDS=()
SHARD_DIRS=()

while IFS= read -r assignment || [ -n "$assignment" ]; do
    GPU=$(echo "$assignment" | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['gpu'])")
    TASKS=$(echo "$assignment" | python3 -c "import json,sys; print(','.join(json.loads(sys.stdin.read())['tasks']))")

    SHARD_DIR="${BASE_OUTPUT}/_shard${GPU}"
    SHARD_DIRS+=("$SHARD_DIR")
    SHARD_LOG="${BASE_OUTPUT}/_shard${GPU}.log"
    mkdir -p "$SHARD_DIR"

    echo "[GPU $GPU] tasks=$TASKS"

    CUDA_VISIBLE_DEVICES=$GPU MASTER_PORT=$((29500 + GPU)) .venv/bin/python -m lmms_eval \
        --model "$MODEL_TYPE" \
        --model_args "pretrained=$PRETRAINED" \
        --tasks "$TASKS" \
        --batch_size 1 \
        --output_path "$SHARD_DIR" \
        --log_samples \
        > "$SHARD_LOG" 2>&1 &

    PIDS+=($!)
done < <(echo "$TASK_ASSIGNMENTS" | python3 -c "
import json, sys
for a in json.loads(sys.stdin.read()):
    print(json.dumps(a))")

echo
echo "Launched ${#PIDS[@]} GPU processes, waiting..."

FAILED=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" || true
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "[GPU $i] FAILED (exit=$rc)"
        FAILED=$((FAILED + 1))
    else
        echo "[GPU $i] done"
    fi
done

if [ "$FAILED" -gt 0 ]; then
    echo "ERROR: $FAILED GPUs failed"
    exit 1
fi

echo
echo "All GPUs complete. Merging..."

.venv/bin/python scripts/merge_shards.py \
    --shard-dirs "${SHARD_DIRS[@]}" \
    --output-dir "${BASE_OUTPUT}/_merged"

echo "=== Done: ${BASE_OUTPUT}/_merged ==="
