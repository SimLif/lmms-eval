#!/bin/bash
# clean_gpus.sh — 3-layer cleanup for a set of GPU indices on the current node.
# Usage:
#   clean_gpus.sh                  # all GPUs
#   clean_gpus.sh 0,3,5            # specific GPUs
#
# Layer 1: lsof /dev/nvidiaN → kill user processes (EXCLUDE monitoring tools)
# Layer 2: cuDevicePrimaryCtxReset → release zombie CUDA contexts
# Layer 3: rm /dev/shm/nccl-* → NCCL shared-memory residuals

set -euo pipefail

NGPUS_DEFAULT=8
EXCLUDE="nvitop|gpustat|nvtop"

# --- resolve target GPU list ---
if [[ $# -gt 0 && "$1" != "" ]]; then
    IFS=',' read -ra GPUS <<< "$1"
else
    GPUS=()
    for i in $(seq 0 $((NGPUS_DEFAULT-1))); do
        [ -e "/dev/nvidia$i" ] && GPUS+=("$i")
    done
fi

# --- auto-detect libcuda.so.* ---
LIBCUDA=$(ls /usr/lib/x86_64-linux-gnu/libcuda.so.*.*.* 2>/dev/null | head -1)
[ -z "$LIBCUDA" ] && LIBCUDA=$(ldconfig -p | awk '/libcuda.so.[0-9]/ {print $NF; exit}')
[ -z "$LIBCUDA" ] && LIBCUDA="libcuda.so.1"

echo "  Target GPUs: ${GPUS[*]}"

# --- also signal placeholders to exit gracefully first ---
mkdir -p /tmp/gpu-watchdog
for i in "${GPUS[@]}"; do
    touch "/tmp/gpu-watchdog/release-$i"
done
sleep 1

# --- Layer 1: lsof + kill ---
# NOTE: lsof /dev/nvidiaN returns ALL processes that opened that device,
# but a PyTorch process typically opens all 8 devices regardless of which
# it computes on. To avoid killing unrelated processes, we cross-check
# each candidate's CUDA_VISIBLE_DEVICES env (set at exec time, visible
# in /proc/PID/environ) — if it explicitly targets GPUs NOT in our list,
# skip it. Processes without CUDA_VISIBLE_DEVICES are assumed to use all
# GPUs and get killed.
target_set=" ${GPUS[*]} "
all_pids=()
for i in "${GPUS[@]}"; do
    dev="/dev/nvidia${i}"
    [ -e "$dev" ] || continue
    while read -r pid; do
        [ -z "$pid" ] && continue
        [ ! -d "/proc/$pid" ] && continue
        cmdline=$(ps -o args= -p "$pid" 2>/dev/null || echo "")
        echo "$cmdline" | grep -qE "$EXCLUDE" && continue
        # Check CUDA_VISIBLE_DEVICES at exec time
        env_cvd=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | awk -F= '/^CUDA_VISIBLE_DEVICES=/ {print $2; exit}')
        if [ -n "$env_cvd" ]; then
            # process explicitly targets specific GPUs — only kill if any intersect with our target
            skip=true
            IFS=',' read -ra proc_gpus <<< "$env_cvd"
            for pg in "${proc_gpus[@]}"; do
                [[ "$target_set" == *" $pg "* ]] && { skip=false; break; }
            done
            $skip && continue
        fi
        all_pids+=("$pid")
    done < <(lsof "$dev" 2>/dev/null | awk 'NR>1 {print $2}' | sort -un)
done

unique_pids=($(printf '%s\n' "${all_pids[@]}" | sort -un))

if [ ${#unique_pids[@]} -gt 0 ]; then
    echo "  Layer 1: ${#unique_pids[@]} GPU processes → SIGTERM"
    kill -TERM "${unique_pids[@]}" 2>/dev/null || true
    sleep 2
    alive=0
    for pid in "${unique_pids[@]}"; do
        [ -d "/proc/$pid" ] && kill -9 "$pid" 2>/dev/null && alive=$((alive+1))
    done
    [ $alive -gt 0 ] && echo "  Layer 1: SIGKILL → $alive remaining"
    sleep 1
else
    echo "  Layer 1: no live processes"
fi

# --- Layer 2: cuDevicePrimaryCtxReset for zombies ---
zombies=()
while read -r pid; do
    pid=$(echo "$pid" | tr -d ' ')
    [ -z "$pid" ] && continue
    [ ! -d "/proc/$pid" ] && zombies+=("$pid")
done < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)

if [ ${#zombies[@]} -gt 0 ]; then
    echo "  Layer 2: ${#zombies[@]} zombie CUDA contexts → reset"
    # Best-effort: if cuInit fails (e.g. error 803 — userspace libcuda / host
    # driver version mismatch from stale ldconfig path), print WARN and skip
    # ctx reset. Subsequent vllm/torch runs load their own CUDA libs and
    # work regardless. Layer 3 (NCCL shm) still runs.
    set +e
    python3 -c "
import ctypes, sys
gpus = [$(IFS=,; echo "${GPUS[*]}")]
try:
    lib = ctypes.CDLL('$LIBCUDA')
    rc = lib.cuInit(0)
    if rc != 0:
        # 803 = CUDA_ERROR_SYSTEM_DRIVER_MISMATCH — userspace lib path issue,
        # not a real device failure. See https://docs.nvidia.com/cuda/cuda-driver-api/
        print(f'    WARN: cuInit returned {rc} (userspace libcuda mismatch via ldconfig; skipping ctx reset — vllm/torch use bundled CUDA libs and are unaffected)', file=sys.stderr)
    else:
        for i in gpus:
            ret = lib.cuDevicePrimaryCtxReset(i)
            print(f'    GPU {i}: {\"OK\" if ret == 0 else f\"FAIL({ret})\"}')
except Exception as e:
    print(f'    WARN: Layer 2 exception: {e}', file=sys.stderr)
" 2>&1
    set -e
else
    echo "  Layer 2: no zombies"
fi

# --- Layer 3: NCCL SHM ---
nccl_count=$(shopt -s nullglob; files=(/dev/shm/nccl-*); echo "${#files[@]}")
if [ "$nccl_count" -gt 0 ]; then
    rm -f /dev/shm/nccl-*
    echo "  Layer 3: removed $nccl_count NCCL SHM files"
else
    echo "  Layer 3: no NCCL residuals"
fi

# --- also clean placeholder PID files for cleared GPUs ---
for i in "${GPUS[@]}"; do
    rm -f "/tmp/gpu-watchdog/placeholder-$i.pid" \
          "/tmp/gpu-watchdog/placeholder-$i.log" \
          "/tmp/gpu-watchdog/release-$i" 2>/dev/null
done

# --- report ---
echo "  Result:"
for i in "${GPUS[@]}"; do
    line=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null | awk -v g="$i" -F, '$1 == g {print $2}')
    echo "    GPU $i: $line"
done
