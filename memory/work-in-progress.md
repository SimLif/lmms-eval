# Work In Progress

> Created: 2026-04-14
> Project: lmms-eval Medical Evaluation Pipeline
> Structure: Main file (status + index) · [sessions/](sessions/) (session logs) · [references/](references/) (knowledge)

---

## Current Status

*Last Updated: 2026-04-15 (Session 03)*

### Progress Summary

- **Eval Runner**: decoupled from shell scripts → `run_eval.py` + YAML config | done
- **LLM Judge**: posthoc with auto-serve VLLMLauncher | done
- **Google Sheet**: upload with gradient formatting | done
- **Pipeline**: eval → judge → parse → upload orchestrator | done (redesigned with --no-parse/--parse-only/--preview)
- **Auto Batch Size**: OOM retry with chunk halving | done (72938bd)
- **8 new medical models**: Lingshu×3, Hulu-Med×5 evaluated + judged | done
- **FGMoE handler**: fgmoe_qwen3_vl.py with importlib loading | done
- **Cross-task batching fix**: Collator group_fn by task_name | done (b094f2a)
- **Security**: credentials moved to .secrets/ | done (8824e4e)

### Pending Tasks

| Priority | Task | Description | Status |
|----------|------|-------------|--------|
| P1 | FGMoE iterative eval | Evaluate new FGMoE checkpoints as they come | ongoing |
| P2 | Baseline re-eval | Re-eval existing baselines with Collator fix to verify no contamination | pending |
| P2 | MMMU full set | Replace 150-question mini to avoid Lingshu contamination | pending |
| P3 | run_eval.py error detection | Fix false PASS when accelerate children fail | pending |

### Architecture Decisions

1. **pixels**: `max_pixels`/`min_pixels` per-model in YAML, not defaults
2. **transformers switching**: `uv pip install --no-deps` + .pyc cache clearing
3. **judge server**: VLLMLauncher context manager with auto cleanup
4. **cross-task fix**: `group_fn=lambda x: {"task": x[4]}` in Collator — groups by task_name not gen_kwargs
5. **pipeline phases**: remote (eval+judge --no-parse) → sync → local (--parse-only --preview --upload)
6. **FGMoE loading**: importlib bypasses auto_map; chat_template fallback from tokenizer
7. **InternVL native**: Lingshu-I-8B uses internvl_hf (transformers 4.57.6 native) not internvl2 (trust_remote_code .chat() API)
8. **credentials**: .secrets/ (gitignored) with .secrets.example/ templates

### Cluster

- 4 nodes × 8 GPU B-card (183GB), SSH port 56683
- Hostfile: `/root/paddlejob/workspace/hostfile`
- Current node: s3 (RANK=3, 10.51.201.204)
- s2 (10.51.202.85): main eval machine, env set up
- Machine config: `.secrets/hosts.yaml`

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
| 01 | 04-14 | Pipeline build + Qwen3-VL-2B eval | run_eval.py, pipeline.py, upload script, VLLMLauncher |
| 02 | 04-14 | Auto batch size + missing model research | batch_utils.py, OOM retry, s2 env setup |
| 03 | 04-14~15 | Add models + cross-task fix + pipeline redesign | hulumed.py, fgmoe_qwen3_vl.py, Collator fix, pipeline --no-parse/--parse-only/--preview, .secrets/ |
