# Med-Eval Handbook

Medical VLM evaluation pipeline: eval → LLM judge → results YAML → Google Sheet.

## Quick Start

```bash
# 1. Setup environment (first time or after GPU change)
bash scripts/rebuild_env.sh

# 2. Evaluate a single model (judge enabled by default)
.venv/bin/python scripts/pipeline.py -c scripts/configs/med_eval_mini.yaml \
    --models Qwen3-VL-2B-Instruct

# 3. Evaluate without judge (faster, judge-dependent metrics show "-")
.venv/bin/python scripts/pipeline.py -c scripts/configs/med_eval_mini.yaml \
    --models Qwen3-VL-2B-Instruct --no-judge

# 4. Evaluate + upload to Google Sheet
.venv/bin/python scripts/pipeline.py -c scripts/configs/med_eval_mini.yaml \
    --models Qwen3-VL-2B-Instruct --upload

# 5. Re-run judge on existing results (skip eval)
.venv/bin/python scripts/pipeline.py -c scripts/configs/med_eval_mini.yaml \
    --models Qwen3-VL-2B-Instruct --skip-eval

# 6. Quick test (2 samples per task)
.venv/bin/python scripts/pipeline.py -c scripts/configs/med_eval_mini.yaml \
    --models Qwen3-VL-2B-Instruct --limit 2 --no-judge
```

## Adding a New FGMoE Checkpoint

1. Add model to `scripts/configs/med_eval_mini.yaml`:
```yaml
  - name: FGMoE-v15-B1
    model_type: qwen3_vl          # or your custom model type
    pretrained: /path/to/checkpoint
    launch: multi
    params: "~2B"
    max_pixels: 3211264
    min_pixels: 200704
    attn_implementation: sdpa
    tags: [fgmoe]
```

2. Run pipeline:
```bash
.venv/bin/python scripts/pipeline.py -c scripts/configs/med_eval_mini.yaml \
    --models FGMoE-v15-B1 --yaml-group "FGMoE Checkpoints" --upload
```

## Key Files

| File | Purpose |
|------|---------|
| `scripts/pipeline.py` | Pipeline orchestrator |
| `scripts/configs/med_eval_mini.yaml` | Model configs (19 models) |
| `data/eval_results/med_eval_results.yaml` | Results + Sheet format definition |
| `scripts/run_eval.py` | Evaluation runner |
| `scripts/posthoc_llm_judge.py` | LLM judge (Qwen3-VL-32B) |
| `scripts/upload_eval_results.py` | Google Sheet uploader |

## Benchmark: med_eval_mini (9 tasks, ~12K samples)

| Task | Samples | Judge needed |
|------|---------|:------------:|
| MMMU-Med | 150 | No |
| VQA-RAD (open+closed) | 451 | Yes (open) |
| SLAKE (open+closed) | 2,094 | Yes (open) |
| PathVQA-Mini (open+closed) | 1,500 | Yes (open) |
| PMC-VQA | 2,000 | No |
| VQA-Med | 1,382 | Yes |
| OmniMedVQA | 2,028 | No |
| MedXpertQA | 2,000 | No |
| PubMedQA | 500 | No |
