#!/usr/bin/env bash
# sync_judge_model.sh — ensure Qwen3-VL-32B-Instruct (the judge model) is
# present locally before start_judge_server.sh runs.
#
# Usage:
#   bash scripts/sync_judge_model.sh                  # download from HF hub
#   bash scripts/sync_judge_model.sh --from-node <job>/<rank>   # rsync from a peer node
#
# Target path: /root/paddlejob/workspace/env_run/hqguo/models/Qwen3-VL-32B-Instruct
#
# Idempotent: if target already has >=14 model-*.safetensors shards, exits 0
# without touching the network.
#
# Requires .secrets/env (for HF_TOKEN + proxy) and, for --from-node,
# .secrets/hosts.yaml + ~/.ssh/id_rsa.hqguo (read via cluster-cli conventions).

set -euo pipefail

TARGET="/root/paddlejob/workspace/env_run/hqguo/models/Qwen3-VL-32B-Instruct"
HF_REPO="Qwen/Qwen3-VL-32B-Instruct"
MIN_SHARDS=14

FROM_NODE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --from-node) FROM_NODE="${2:-}"; shift 2 ;;
        -h|--help)
            sed -n '1,20p' "$0"
            exit 0
            ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------
# 0. Fast-path: already cached
# ---------------------------------------------------------------
count_shards() {
    local d="$1"
    [ -d "$d" ] || { echo 0; return; }
    # shellcheck disable=SC2012
    ls "$d"/model-*.safetensors 2>/dev/null | wc -l
}

SHARDS=$(count_shards "$TARGET")
if [ "$SHARDS" -ge "$MIN_SHARDS" ]; then
    SIZE=$(du -sh "$TARGET" 2>/dev/null | awk '{print $1}')
    echo "[INFO] judge model ready at $TARGET (size: ${SIZE}, shards: ${SHARDS})"
    exit 0
fi

echo "[INFO] $TARGET not complete yet (found ${SHARDS} shards, need >=${MIN_SHARDS})"

# ---------------------------------------------------------------
# 1. Load .secrets/env for HF_TOKEN + proxy
# ---------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
SECRETS_ENV="${REPO_DIR}/.secrets/env"
if [ ! -f "$SECRETS_ENV" ]; then
    echo "ERROR: .secrets/env missing at $SECRETS_ENV. Mirror from AFS first." >&2
    exit 1
fi
# shellcheck source=/dev/null
source "$SECRETS_ENV"

mkdir -p "$(dirname "$TARGET")"

# ---------------------------------------------------------------
# 2. Acquire weights
# ---------------------------------------------------------------
if [ -n "$FROM_NODE" ]; then
    # ------- Rsync from peer node via .secrets/hosts.yaml -------
    HOSTS_YAML="${REPO_DIR}/.secrets/hosts.yaml"
    if [ ! -f "$HOSTS_YAML" ]; then
        echo "ERROR: .secrets/hosts.yaml missing; cannot look up --from-node" >&2
        exit 1
    fi
    eval "$(python3 -c "
import os, sys, shlex, yaml
with open('$HOSTS_YAML') as f:
    cfg = yaml.safe_load(f) or {}
key = '$FROM_NODE'
if '/' not in key:
    print('echo \"ERROR: --from-node must be <job>/<rank>\" >&2; exit 1'); sys.exit(0)
job_id, rank_s = key.split('/', 1)
job = (cfg.get('jobs') or {}).get(job_id)
nodes = (job or {}).get('nodes') or []
if not job or not rank_s.isdigit() or int(rank_s) >= len(nodes):
    print(f'echo \"ERROR: node $FROM_NODE not found in hosts.yaml\" >&2; exit 1'); sys.exit(0)
meta = cfg.get('meta', {})
ssh_key = meta.get('ssh_key') or ''
if ssh_key:
    ssh_key = os.path.expanduser(ssh_key)
print(f'SRC_HOST={shlex.quote(nodes[int(rank_s)])}')
print(f'SRC_PORT={job.get(\"port\", 22)}')
print(f'SRC_USER={shlex.quote(meta.get(\"user\", \"root\"))}')
print(f'SRC_KEY={shlex.quote(ssh_key)}')
")"

    if [ -n "${SRC_KEY:-}" ]; then
        SSH_OPTS="ssh -i ${SRC_KEY} -p ${SRC_PORT} -o StrictHostKeyChecking=no"
    else
        SSH_OPTS="ssh -p ${SRC_PORT} -o StrictHostKeyChecking=no"
    fi
    echo "[INFO] rsync from ${SRC_USER}@${SRC_HOST}:${TARGET}/ → ${TARGET}/"
    rsync -az --info=progress2 -e "$SSH_OPTS" \
        "${SRC_USER}@${SRC_HOST}:${TARGET}/" "${TARGET}/"
else
    # ------- Download from HF hub -------
    if [ -z "${HF_TOKEN:-}" ]; then
        echo "WARN: HF_TOKEN not set; download may fail for gated repos" >&2
    fi
    echo "[INFO] downloading ${HF_REPO} from HuggingFace Hub → ${TARGET}"
    # Try `hf` first (new CLI), then fall back to `huggingface-cli`, then python.
    if command -v hf >/dev/null 2>&1; then
        hf download "$HF_REPO" --local-dir "$TARGET"
    elif command -v huggingface-cli >/dev/null 2>&1; then
        huggingface-cli download "$HF_REPO" --local-dir "$TARGET"
    else
        python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="$HF_REPO", local_dir="$TARGET")
PY
    fi
fi

# ---------------------------------------------------------------
# 3. Verify completeness against safetensors.index.json
# ---------------------------------------------------------------
SHARDS=$(count_shards "$TARGET")
EXPECTED=""
INDEX="${TARGET}/model.safetensors.index.json"
if [ -f "$INDEX" ]; then
    EXPECTED=$(python3 -c "
import json
with open('$INDEX') as f:
    idx = json.load(f)
# Unique shard filenames referenced in the weight map
print(len(set(idx['weight_map'].values())))
" 2>/dev/null || echo "")
fi

SIZE=$(du -sh "$TARGET" 2>/dev/null | awk '{print $1}')
if [ -n "$EXPECTED" ]; then
    if [ "$SHARDS" -lt "$EXPECTED" ]; then
        echo "ERROR: only ${SHARDS}/${EXPECTED} shards present at $TARGET" >&2
        exit 1
    fi
    echo "[INFO] judge model ready at $TARGET (size: ${SIZE}, shards: ${SHARDS}/${EXPECTED})"
else
    if [ "$SHARDS" -lt "$MIN_SHARDS" ]; then
        echo "ERROR: only ${SHARDS} shards at $TARGET (need >=${MIN_SHARDS}); no index.json to verify" >&2
        exit 1
    fi
    echo "[INFO] judge model ready at $TARGET (size: ${SIZE}, shards: ${SHARDS}, no index.json)"
fi
