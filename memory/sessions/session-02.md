# Session 02: 2026-04-14 — Auto Batch Size + Missing Model Research

## Goal
Implement auto batch size with OOM retry and chunk halving. Then begin adding models from Sheet that are missing from med_eval_mini.yaml.

## Completed Tasks

1. **Auto batch size implemented**: `lmms_eval/models/model_utils/batch_utils.py` (NEW) with `parse_batch_size()` and `run_with_oom_retry()`. Supports `--batch_size auto` / `auto:N`. On CUDA OOM: halves chunk via FIFO queue, clears cache, updates `batch_size_per_gpu`.

2. **4 model files updated**: `simple/qwen2_5_vl.py`, `simple/qwen3_vl.py`, `chat/qwen2_5_vl.py`, `chat/qwen3_vl.py` — extracted `_process_chunk()` method, main loop calls `run_with_oom_retry()`. Chat models measure timing at outer loop level.

3. **run_eval.py updated**: `batch_size` fields changed from `int` to `int | str`, CLI `--batch-size` accepts string for "auto:N" syntax.

4. **s2 node environment set up**: code rsynced, model caches synced, proxy configured (`http_proxy`/`https_proxy` with cmcproxy). Both curl and Python verified working through proxy.

5. **OOM recovery tested on s2**: Qwen3-VL-32B (32B bf16 ~64GB weights) with bs=128 on single 183GB GPU. Successfully OOM'd and auto-halved: bs=128 → 64 → 32, completed MMMU eval.

6. **Committed and pushed**: `72938bd feat(models): auto batch size with OOM retry and chunk halving` to origin/main.

7. **Missing model research (partial)**: Identified 10 models in Sheet but not in YAML config. Architecture mapping:
   - Lingshu-32B/7B → `qwen2_5_vl` (standard Qwen2.5-VL arch)
   - Lingshu-I-8B → InternVL arch (internvl2 or internvl3 model type)
   - Hulu-Med-30A3 → `Qwen3VLMoeForConditionalGeneration` (qwen3_vl MoE)
   - Hulu-Med-4B/14B → custom `HulumedQwen3ForCausalLM` (needs investigation)
   - Hulu-Med-7B/32B → custom `HulumedQwen2ForCausalLM` (needs investigation)
   - p0_blend_B1-dense, FGMoE-G32-Full → FGMoE checkpoints (need local paths)

## Key Findings

- **SSH script pattern**: Scripts run via SSH must be written to `/tmp/lmms-eval/` locally first with `cd "$PROJECT"` at start, because SSH starts in `/root` not the project dir.
- **Proxy for HF on s2**: Must explicitly `export http_proxy=...` in scripts; env vars not inherited via SSH. `HF_HUB_OFFLINE=1` blocks dataset loading — use proxy instead.
- **Small models don't OOM easily**: Qwen3-VL-2B/3B with bs=128/256 on 183GB card never OOM'd. Need 32B+ to trigger OOM for testing.
- **GPU 0-1 on s3**: Used by FGMoE training, must use s2 for eval testing.
- **InternVL models assert bs==1**: `internvl2.py`, `internvl3.py` all assert batch_size==1.

## Next Steps

1. **Research Hulu-Med architectures**: Determine if `HulumedQwen2ForCausalLM`/`HulumedQwen3ForCausalLM` can use existing model types with `trust_remote_code=True` or need new implementations
2. **Find FGMoE checkpoint paths**: Locate local paths for p0_blend_B1-dense (wandb: w08niumg) and FGMoE-G32-Full (wandb: reye83pn)
3. **Add missing models to med_eval_mini.yaml**: Configure all 10 models with correct model_type, pretrained path, launch mode, pixels, etc.
4. **Evaluate one by one on s2**: Run each missing model through med_eval_mini pipeline (--no-judge first for speed, judge later)
5. **Exclude Qwen3.5-2B-Think**: User explicitly said not to include this model
