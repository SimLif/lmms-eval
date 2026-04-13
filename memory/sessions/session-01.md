# Session 01: 2026-04-14 — Pipeline Build + Qwen3-VL-2B Eval

## Goal
Recover lost evaluation infrastructure after machine crash. Build a decoupled eval pipeline and reproduce Qwen3-VL-2B results on med_eval_mini.

## Completed Tasks

1. **Eval runner decoupled**: `scripts/run_eval.py` + `scripts/configs/med_eval_mini.yaml` — 19 models with explicit pixels/attn/params. Three launch modes (multi/single/vllm). Old shell scripts marked DEPRECATED.

2. **Environment rebuilt**: `scripts/rebuild_env.sh` — B-card (sm_100, CUDA 13.1 vs torch cu128). Patched torch CUDA version check, compiled flash-attn/causal-conv1d/flash-linear-attention. Pinned torch==2.8.0 in pyproject.toml.

3. **Qwen3-VL-2B evaluated**: batch_size=32, 8 GPU accelerate, 931s. Results match target within ~1%: VQA-RAD 52.1 (target 51.68), SLAKE 44.6 (44.84), PathVQA 33.5 (33.27).

4. **LLM judge enhanced**: `judge_utils.py` now returns judge_model + judge_raw_response per sample. 5 task utils updated. Posthoc judge completed all 3569 open-ended samples with Qwen3-VL-32B.

5. **VLLMLauncher**: Context manager for auto judge server lifecycle. Fixed stdout PIPE buffer deadlock (→ log file). Fixed orphan worker cleanup. `posthoc_llm_judge.py --auto_serve` works end-to-end.

6. **Google Sheet upload**: `scripts/upload_eval_results.py` — clear→write→format strategy. Gradient conditional formatting per metric column. Model background colors by group. Fixed: data must be numeric (not string) for gradients to render.

7. **Pipeline orchestrator**: `scripts/pipeline.py` — eval → judge → parse → upload. Judge ON by default, upload OFF. YAML update by name match. "-" for judge-dependent metrics when judge not run.

8. **Handbook**: `docs/med-eval-handbook.md` — usage guide for FGMoE evaluation.

9. **Google Sheets MCP**: Configured service account + mcp-gsheets with proxy.

10. **Cluster SSH**: 4 nodes verified accessible via port 56683 with id_rsa key. s2 (10.51.202.85) confirmed idle (8x B-card, no active GPU processes).

11. **Git push**: 2 commits pushed to origin/main via SSH (remote set to git@github.com:SimLif/lmms-eval.git with proxy).

12. **Memory bank**: Initialized `memory/` with work-in-progress.md + session-01.md. Added to git (removed `memory/` from .gitignore, kept `memory/plan.md` ignored).

## Key Findings

- **CUDA version mismatch**: B-card has CUDA 13.1, torch compiled with 12.8. Must patch `torch.utils.cpp_extension._check_cuda_version` for source builds. `uv pip install --no-deps` essential to prevent torch/vllm upgrades.
- **Gradient formatting requires numeric values**: Google Sheets gradient rules only apply to NUMBER type cells, not strings. `USER_ENTERED` + prior `numberFormat: TEXT` reset causes values to stay as strings.
- **vLLM stdout PIPE deadlock**: `subprocess.PIPE` for vLLM server output causes buffer fill → server blocks. Must use file output.
- **transformers version switching**: `uv add` triggers sync that deletes flash-attn. Use `uv pip install --no-deps` with companion package swaps (tokenizers for 4.49.0).

## Next Steps

1. Implement auto batch size detection (OOM retry with chunk halving)
2. Run full med_eval_mini on remaining 18 models
3. Explore multi-node evaluation across 4 cluster nodes (s0-s3, port 56683)
4. Fine-tune upload format (red bold cell count, column-max vs group-max)
5. Consider using s2 (idle) for parallel model evaluation or judge server
