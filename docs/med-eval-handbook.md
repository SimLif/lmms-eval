# Med-Eval Handbook

Medical VLM evaluation pipeline: eval → LLM judge → parse → upload.

## Workflow

```
Training Machine (remote)           Local Machine
┌─────────────────────────┐         ┌─────────────────────────┐
│ 1. eval + judge         │  rsync  │ 3. parse                │
│    --no-parse           │ ──────► │    --parse-only          │
│                         │  logs/  │ 4. preview               │
│ 2. repeat for each      │         │    --parse-only --preview│
│    model / machine      │         │ 5. upload                │
│                         │         │    --parse-only --upload  │
└─────────────────────────┘         └─────────────────────────┘
```

## Quick Start

```bash
# Remote: eval + judge (no parse, no upload)
pipeline.py -c scripts/configs/med_eval_mini.yaml \
    --models Qwen3-VL-2B-Instruct --no-parse

# Remote: eval only (faster, judge-dependent metrics will show "-")
pipeline.py -c scripts/configs/med_eval_mini.yaml \
    --models Qwen3-VL-2B-Instruct --no-judge --no-parse

# Sync logs to local
rsync -az remote:project/logs/med_eval_mini/ logs/med_eval_mini/

# Local: parse + preview
pipeline.py -c scripts/configs/med_eval_mini.yaml \
    --models Qwen3-VL-2B-Instruct --parse-only --preview

# Local: parse + upload to Google Sheet
pipeline.py -c scripts/configs/med_eval_mini.yaml \
    --models Qwen3-VL-2B-Instruct --parse-only --upload
```

## Adding a New FGMoE Checkpoint

1. Add model to `scripts/configs/med_eval_mini.yaml`:
```yaml
  - name: FGMoE-v15-B1
    model_type: fgmoe_qwen3_vl
    pretrained: /path/to/checkpoint
    launch: multi
    params: "~2B"
    max_pixels: 1003520
    min_pixels: 200704
    attn_implementation: sdpa
    tags: [fgmoe]
```

2. Run on training machine:
```bash
pipeline.py -c scripts/configs/med_eval_mini.yaml \
    --models FGMoE-v15-B1 --no-parse
```

3. Sync logs, parse, and upload locally:
```bash
rsync -az remote:project/logs/med_eval_mini/FGMoE-v15-B1/ logs/med_eval_mini/FGMoE-v15-B1/
pipeline.py -c scripts/configs/med_eval_mini.yaml \
    --models FGMoE-v15-B1 --parse-only --preview --yaml-group "FGMoE Checkpoints"
pipeline.py -c scripts/configs/med_eval_mini.yaml \
    --models FGMoE-v15-B1 --parse-only --upload --yaml-group "FGMoE Checkpoints"
```

## Pipeline Flags

| Flag | Phase | Description |
|------|-------|-------------|
| `--no-parse` | eval+judge | Stop after eval/judge, skip parse and upload |
| `--no-judge` | eval | Skip LLM judge (judge metrics show "-") |
| `--parse-only` | parse | Skip eval+judge, parse existing logs only |
| `--preview` | parse | Print formatted results table after parse |
| `--upload` | upload | Push results YAML to Google Sheet |
| `--skip-eval` | judge | Re-run judge on existing eval results |
| `--limit N` | eval | Limit samples per task (testing only) |
| `--dry-run` | all | Show commands without executing |

## Key Files

| File | Purpose |
|------|---------|
| `scripts/pipeline.py` | Pipeline orchestrator |
| `scripts/configs/med_eval_mini.yaml` | Model configs |
| `data/eval_results/med_eval_results.yaml` | Results + Sheet format |
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
