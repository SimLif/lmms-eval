# Session 03: 2026-04-14~15 — Add Missing Models + Cross-Task Fix + Pipeline Redesign

## Goal
Continue from Session 02: add 8 missing models (Lingshu×3, Hulu-Med×5) to med_eval_mini, evaluate on s2, compare with Google Sheet. Then add FGMoE support and fix eval bugs.

## Completed Tasks

1. **Researched Hulu-Med architectures**: Hulu-Med-30A3 uses native `qwen3_vl` (MoE). Hulu-Med-4B/14B/7B/32B are LLaVA-style custom architectures needing `trust_remote_code=True`.

2. **Fixed qwen3_vl.py MoE detection**: Replaced regex `A\d+B` with config-based `"moe" in config.model_type`. Handles Hulu-Med-30A3 correctly.

3. **Created `hulumed` model handler**: `lmms_eval/models/simple/hulumed.py` — loads via `AutoModelForCausalLM` + `AutoProcessor` with `trust_remote_code=True`. Key: model.generate() returns only generated tokens (no input prefix), cast pixel_values to bfloat16.

4. **Added 8 models to med_eval_mini.yaml**: Lingshu-7B/32B/I-8B, Hulu-Med-30A3/4B/7B/14B/32B. All evaluated + judged on s2, results match Google Sheet Summary_Backup within ±3%.

5. **Fixed Lingshu-I-8B**: Changed from `internvl2` (old .chat() API) to `internvl_hf` (native InternVL in transformers 4.57.6). Fixed `internvl_hf.py` for text-only tasks (`images=None`). Fixed `internvl2.py` flash_attn TypeError fallback.

6. **Fixed Hulu-Med-32B**: `launch: single` (device_map=auto) caused 0% on some tasks. Changed to `launch: multi` (data parallel). Results restored to correct values.

7. **Created FGMoE handler**: `fgmoe_qwen3_vl.py` — loads via importlib (bypasses auto_map format issues), chat_template fallback from tokenizer, inherits Qwen3_VL eval logic.

8. **Found and fixed cross-task batching bug**: Collator with `grouping=True` grouped by `gen_kwargs` (args[1]), not task_name. Mixed tasks in same batch caused `task_dict[task_A][split][doc_id_from_task_B]` → IndexError. Fix: `group_fn=lambda x: {"task": x[4]}` groups by task_name.

9. **Redesigned pipeline.py**: Added `--no-parse` (remote eval+judge only), `--parse-only` (local parse), `--preview` (formatted table with "-" for missing judge). Supports distributed workflow: remote eval → sync logs → local parse → preview → upload.

10. **Security: moved credentials to .secrets/**: Removed hardcoded proxy password from 3 shell scripts. Created `.secrets/` (gitignored) for proxy, HF token, WandB key, Google credentials, hosts. Created `.secrets.example/` templates.

11. **Removed LMMS_EVAL_USE_CACHE**: Was causing stale cache confusion. Default is False anyway.

12. **Lingshu MMMU data contamination analysis**: Lingshu-7B/I-8B score 82-90% on MMMU mini (150 questions) but paper reports 49-54%. 149/150 responses are single-letter (no reasoning) — training data contamination of MMMU validation set.

13. **Uploaded Lingshu paper benchmark results**: Created `upload_lingshu_report.py`, uploaded 26-model comparison table to Google Sheet "Lingshu Report" tab.

## Key Findings

- **Cross-task batching root cause**: Collator groups by gen_kwargs (args[1]) not task_name (args[4]). When tasks share same gen_kwargs, requests mix across tasks. _process_chunk uses task[0] for all items → wrong dataset lookup. Only manifests when doc_ids exceed smallest dataset size (slake 2094 > omni_med 2000).
- **device_map=auto unreliable for custom MoE models**: Hulu-Med-32B got 0% on pmc_vqa/omnimedvqa with device_map=auto but correct results with multi (data parallel).
- **LMMS_EVAL_USE_CACHE creates stale cache**: Failed runs write partial cache, subsequent runs load stale entries. Disabled.
- **InternVL native vs trust_remote_code**: Models without auto_map + .py files need native transformers support (4.54+). `internvl_hf` handler uses `InternVLForConditionalGeneration` + `AutoProcessor`.
- **Lingshu paper comparison**: Our eval matches paper within ±3% on judge-independent metrics. Systematic -7~20 on SLAKE/VQA-RAD/PathVQA because we include open-ended judge scores, paper likely uses closed-only.

## Next Steps

1. **Run full baseline re-eval**: Use new Collator fix to re-evaluate existing baselines (Qwen2.5-VL, InternVL, etc.) and verify no cross-task contamination in previous results
2. **FGMoE iterative evaluation**: As new FGMoE checkpoints are produced, evaluate using `pipeline.py --no-parse` on training machine workflow
3. **Consider MMMU full set**: Replace 150-question mini with full MMMU Medical test set to avoid contamination artifacts
4. **Fix run_eval.py success detection**: Currently reports PASS based on subprocess exit code even when accelerate child processes have errors
