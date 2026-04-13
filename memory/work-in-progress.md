# Work In Progress

> Created: 2026-04-14
> Project: lmms-eval Medical Evaluation Pipeline
> Structure: Main file (status + index) · [sessions/](sessions/) (session logs) · [references/](references/) (knowledge)

---

## Current Status

*Last Updated: 2026-04-14 (Session 01)*

### Progress Summary

- **Eval Runner**: decoupled from shell scripts → `run_eval.py` + YAML config | done
- **LLM Judge**: posthoc with auto-serve VLLMLauncher | done
- **Google Sheet**: upload with gradient formatting | done
- **Pipeline**: eval → judge → parse → upload orchestrator | done
- **Qwen3-VL-2B Reproduction**: eval + judge completed, results match target | done

### Pending Tasks

| Priority | Task | Description | Status |
|----------|------|-------------|--------|
| P1 | Auto batch size | OOM retry with chunk halving in generate_until | designed, not implemented |
| P1 | Evaluate all 19 models | Run full med_eval_mini on remaining models | pending |
| P2 | Multi-node eval | Distribute models across 4 cluster nodes | pending |
| P3 | Upload format polish | Red bold outlier detection may mark too many cells | minor |

### Architecture Decisions

1. **pixels**: `max_pixels`/`min_pixels` per-model in YAML, not defaults (InternVL/llava don't support)
2. **transformers switching**: `uv pip install --no-deps` to preserve flash-attn/causal-conv1d
3. **judge server**: VLLMLauncher with stdout→file (not PIPE) to avoid vLLM buffer deadlock
4. **YAML names**: unified with eval config names (e.g., `Qwen3-VL-2B-Instruct`)
5. **no-judge metrics**: show "-" for judge-dependent metrics when judge not run

### Cluster

- 4 nodes × 8 GPU B-card (CF-NG-BZZ2-O, 183GB), SSH port 56683
- Hostfile: `/root/paddlejob/workspace/hostfile`
- Current node: s3 (RANK=3, 10.51.201.204)

---

## References

| Document | Content | Last Updated |
|----------|---------|--------------|
| (none yet) | | |

---

## Session Log

> Detailed records in [sessions/](sessions/)

| # | Date | Title | Key Output |
|---|------|-------|------------|
| 01 | 04-14 | Pipeline build + Qwen3-VL-2B eval | run_eval.py, pipeline.py, upload script, VLLMLauncher, handbook |
