"""Pipeline: eval → judge → parse → upload.

Orchestrates medical VLM evaluation in separable phases:

  Remote (training machine):
    eval + judge → produces logs with results.json + judged samples

  Local (aggregation machine):
    parse  → reads logs, updates med_eval_results.yaml
    preview → shows formatted table (judge-missing metrics shown as "-")
    upload → pushes YAML to Google Sheet

Usage:
    # Remote: eval + judge (no parse, no upload)
    pipeline.py -c configs/med_eval_mini.yaml --models X --no-parse

    # Remote: eval only (no judge, no parse)
    pipeline.py -c configs/med_eval_mini.yaml --models X --no-judge --no-parse

    # Local: parse from existing logs
    pipeline.py -c configs/med_eval_mini.yaml --models X --parse-only

    # Local: parse + preview table
    pipeline.py -c configs/med_eval_mini.yaml --models X --parse-only --preview

    # Local: parse + upload to Google Sheet
    pipeline.py -c configs/med_eval_mini.yaml --models X --parse-only --upload
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metric mapping: _results.json key → YAML field name
# ---------------------------------------------------------------------------

RESULT_TO_YAML = [
    # (results_json_task_key, metric_key, yaml_field)
    ("mmmu_medical_val", "mmmu_acc,none", "mmmu_med"),
    ("vqa_rad", "accuracy,none", "vqa_rad"),
    ("slake", "accuracy,none", "slake"),
    ("path_vqa_mini", "accuracy,none", "pathvqa"),
    ("pmc_vqa", "accuracy,none", "pmc_vqa"),
    ("vqa_med", "llm_judge,none", "vqa_med"),
    ("omni_med_vqa_mini_v2", "accuracy,none", "omnimedvqa"),
    ("medxpertqa_mm", "accuracy,none", "medxpertqa"),
    ("pubmedqa", "accuracy,none", "pubmedqa"),
]

# Groups whose accuracy requires LLM judge on the open subtask
GROUPS_REQUIRING_JUDGE = {
    "vqa_rad": "vqa_rad_open",
    "slake": "slake_open",
    "path_vqa_mini": "path_vqa_mini_open",
}

DEFAULT_YAML = "data/eval_results/med_eval_results.yaml"
DEFAULT_GROUP = "New Results"

METRIC_FIELDS = [r[2] for r in RESULT_TO_YAML]  # ordered yaml field names


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_eval_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def find_model_in_config(config: dict, name: str) -> dict | None:
    for m in config.get("models", []):
        if m["name"] == name:
            return m
    return None


# ---------------------------------------------------------------------------
# Phase 1: Eval
# ---------------------------------------------------------------------------


def run_eval(
    config_path: str,
    model_name: str,
    batch_size: int | None,
    limit: int | None,
    dry_run: bool,
) -> bool:
    """Run evaluation via run_eval.py subprocess."""
    cmd = [
        sys.executable, "scripts/run_eval.py",
        "-c", config_path,
        "--models", model_name,
    ]
    if batch_size is not None:
        cmd.extend(["--batch-size", str(batch_size)])
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    if dry_run:
        cmd.append("--dry-run")

    log.info("Phase 1: EVAL — %s", model_name)
    log.info("  cmd: %s", " ".join(cmd))

    if dry_run:
        subprocess.run(cmd)
        return True

    result = subprocess.run(cmd)
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Phase 2: Judge
# ---------------------------------------------------------------------------


def find_results_dir(output_dir: str, model_name: str) -> str:
    """Find the inner directory containing *_results.json."""
    base = Path(output_dir) / model_name
    if not base.exists():
        raise FileNotFoundError(f"Model output dir not found: {base}")

    # Search one level deep for *_results.json
    for d in sorted(base.iterdir()):
        if d.is_dir() and list(d.glob("*_results.json")):
            return str(d)

    raise FileNotFoundError(f"No *_results.json found under {base}")


def run_judge(
    results_dir: str,
    judge_model: str,
    serve_gpu_ids: str,
    serve_port: int,
    workers: int,
    dry_run: bool,
    judge_api_url: str | None = None,
    judge_api_type: str = "openai",
) -> bool:
    """Run posthoc LLM judge.

    - If ``judge_api_url`` is set, use remote OpenAI-compatible API (skip
      local vLLM --auto_serve). Expects ``OPENAI_API_KEY`` in env.
    - Otherwise: legacy auto-serve (spin up local vLLM on ``serve_gpu_ids``).
    """
    cmd = [
        sys.executable, "scripts/posthoc_llm_judge.py",
        "--results_dir", results_dir,
        "--run_judge",
        "--judge_model", judge_model,
        "--workers", str(workers),
    ]
    if judge_api_url:
        cmd += [
            "--judge_api_type", judge_api_type,
            "--judge_api_url", judge_api_url,
        ]
    else:
        cmd += [
            "--auto_serve",
            "--serve_gpu_ids", serve_gpu_ids,
            "--serve_port", str(serve_port),
        ]
    if dry_run:
        cmd.append("--dry_run")

    log.info("Phase 2: JUDGE — %s", results_dir)
    log.info("  judge_model: %s", judge_model)
    if judge_api_url:
        log.info("  judge_api_url: %s (remote API, skipping vLLM auto-serve)", judge_api_url)
    log.info("  cmd: %s", " ".join(cmd))

    # setsid to prevent SIGTSTP issues
    result = subprocess.run(
        cmd,
        stdin=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Phase 3: Parse results → update YAML
# ---------------------------------------------------------------------------


def get_metric_value(results: dict, task_key: str, metric_key: str) -> float | str:
    """Extract metric from _results.json, returning '-' if unavailable."""
    task_data = results.get("results", {}).get(task_key, {})

    # For groups requiring judge, verify judge was actually run
    if task_key in GROUPS_REQUIRING_JUDGE:
        open_task = GROUPS_REQUIRING_JUDGE[task_key]
        open_data = results.get("results", {}).get(open_task, {})
        if "llm_judge,none" not in open_data:
            return "-"

    # For vqa_med, check llm_judge directly
    if metric_key == "llm_judge,none" and metric_key not in task_data:
        return "-"

    value = task_data.get(metric_key)
    if value is None:
        return "-"
    return round(float(value), 2)


def parse_results(results_dir: str) -> dict[str, float | str]:
    """Parse _results.json into a dict of YAML field → value."""
    results_files = sorted(Path(results_dir).glob("*_results.json"))
    if not results_files:
        raise FileNotFoundError(f"No _results.json in {results_dir}")

    with open(results_files[-1]) as f:
        results = json.load(f)

    metrics = {}
    for task_key, metric_key, yaml_field in RESULT_TO_YAML:
        metrics[yaml_field] = get_metric_value(results, task_key, metric_key)

    # Compute average (skip "-" values)
    numeric = [v for v in metrics.values() if isinstance(v, (int, float))]
    metrics["avg"] = round(sum(numeric) / len(numeric), 2) if numeric else "-"

    return metrics


def update_yaml(
    yaml_path: str,
    model_name: str,
    params: str,
    metrics: dict,
    yaml_group: str | None,
) -> None:
    """Update or append model entry in the results YAML.

    Auto-creates the file (and parent dirs) with a minimal scaffold when
    missing, so first-run evals don't crash on a bare repo checkout.
    """
    if not os.path.isfile(yaml_path):
        os.makedirs(os.path.dirname(yaml_path) or ".", exist_ok=True)
        with open(yaml_path, "w") as f:
            yaml.dump({"summary": {"groups": []}, "models": []}, f,
                      default_flow_style=False, sort_keys=False)
        log.info("  Created new results YAML: %s", yaml_path)

    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    data.setdefault("summary", {})
    data["summary"].setdefault("groups", [])
    data.setdefault("models", [])

    # Search for existing model by name
    found = False
    for group in data["summary"]["groups"]:
        for model in group["models"]:
            if model["name"] == model_name:
                model.update(metrics)
                model["params"] = params
                found = True
                log.info("  Updated existing entry: %s in '%s'", model_name, group["name"])
                break
        if found:
            break

    if not found:
        target_group_name = yaml_group or DEFAULT_GROUP
        target_group = None
        for group in data["summary"]["groups"]:
            if group["name"] == target_group_name:
                target_group = group
                break

        if target_group is None:
            target_group = {"name": target_group_name, "models": []}
            data["summary"]["groups"].append(target_group)
            log.info("  Created new group: '%s'", target_group_name)

        new_entry = {"name": model_name, "params": params}
        new_entry.update(metrics)
        target_group["models"].append(new_entry)
        log.info("  Appended new entry: %s to '%s'", model_name, target_group_name)

    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    log.info("  YAML saved: %s", yaml_path)


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------


def preview_results(yaml_path: str, model_names: list[str]) -> None:
    """Print a formatted table of results for the given models."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    models = {}
    for group in data["summary"]["groups"]:
        for m in group["models"]:
            if m["name"] in model_names:
                models[m["name"]] = m

    if not models:
        log.warning("No matching models found in %s", yaml_path)
        return

    headers = ["Model", "Params"] + [r[2] for r in RESULT_TO_YAML] + ["Avg"]
    col_widths = [max(len(h), 12) for h in headers]
    col_widths[0] = max(col_widths[0], max(len(n) for n in models) + 2)

    def fmt_val(v):
        if isinstance(v, (int, float)):
            return f"{v:.2f}"
        return str(v)

    # Header
    header_line = "  ".join(h.center(w) for h, w in zip(headers, col_widths))
    print()
    print(header_line)
    print("─" * len(header_line))

    # Data rows
    has_missing = False
    for name in model_names:
        m = models.get(name)
        if m is None:
            print(f"  {name}: NOT FOUND")
            continue
        vals = [name, str(m.get("params", "?"))]
        for field in METRIC_FIELDS:
            v = m.get(field, "-")
            if v == "-":
                has_missing = True
            vals.append(fmt_val(v))
        vals.append(fmt_val(m.get("avg", "-")))
        row = "  ".join(v.center(w) for v, w in zip(vals, col_widths))
        print(row)

    print()
    if has_missing:
        log.warning("Some metrics show '-' — LLM judge may not have been run.")
        log.warning("Run eval+judge on the remote machine, sync logs, then re-parse.")


# ---------------------------------------------------------------------------
# Phase 4: Upload
# ---------------------------------------------------------------------------


def run_upload(yaml_path: str, dry_run: bool) -> bool:
    cmd = [sys.executable, "scripts/upload_eval_results.py", "--data", yaml_path]
    if dry_run:
        cmd.append("--dry-run")

    log.info("Phase 4: UPLOAD")
    result = subprocess.run(cmd)
    return result.returncode == 0


# ---------------------------------------------------------------------------
# CLI + Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Eval pipeline: eval → judge → parse → upload",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow:
  Remote:  pipeline.py -c CFG --models X --no-parse      # eval + judge
  Local:   pipeline.py -c CFG --models X --parse-only     # parse logs
  Local:   pipeline.py -c CFG --models X --parse-only --preview  # parse + table
  Local:   pipeline.py -c CFG --models X --parse-only --upload   # parse + sheet
""",
    )
    p.add_argument("-c", "--config", required=True, help="Eval config YAML")
    p.add_argument("--models", required=True, help="Comma-separated model names")

    # Phase control
    phase = p.add_argument_group("phase control")
    phase.add_argument("--no-judge", action="store_true", help="Skip LLM judge")
    phase.add_argument("--no-parse", action="store_true",
                       help="Stop after eval+judge (no parse/upload)")
    phase.add_argument("--parse-only", action="store_true",
                       help="Skip eval+judge, only parse existing logs")
    phase.add_argument("--skip-eval", action="store_true",
                       help="Skip eval, re-run judge on existing results")
    phase.add_argument("--preview", action="store_true",
                       help="Print formatted results table after parse")
    phase.add_argument("--upload", action="store_true",
                       help="Upload to Google Sheet after parse")

    # Eval options
    ev = p.add_argument_group("eval options")
    ev.add_argument("--limit", type=int, default=None, help="Limit samples per task")
    ev.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    ev.add_argument("--dry-run", action="store_true", help="Dry run all phases")

    # Judge options
    jg = p.add_argument_group("judge options")
    # Prefer local Qwen3-VL-32B when present (HF_HUB_OFFLINE=1 in the judge
    # launcher can't resolve HF repo IDs). Fall back to HF ID for pre-cached
    # shared caches only.
    _LOCAL_JUDGE = "/root/paddlejob/workspace/env_run/hqguo/models/Qwen3-VL-32B-Instruct"
    jg.add_argument("--judge-model",
                    default=(_LOCAL_JUDGE if os.path.isdir(_LOCAL_JUDGE) else "Qwen/Qwen3-VL-32B-Instruct"))
    jg.add_argument("--serve-gpu-ids", default="0,1,2,3")
    jg.add_argument("--serve-port", type=int, default=8000)
    jg.add_argument("--judge-workers", type=int, default=16)
    # Remote-API judge (skips local vLLM start; uses OpenAI-compatible endpoint)
    jg.add_argument("--judge-api-url", default=os.getenv("JUDGE_API_URL"),
                    help="OpenAI-compatible base URL. If set, skip --auto_serve "
                         "and call the remote API directly. "
                         "Env fallback: $JUDGE_API_URL (set in .secrets/env; do not paste here).")
    jg.add_argument("--judge-api-type", default="openai",
                    help="API provider type (openai, anthropic, ...). "
                         "Only relevant with --judge-api-url.")

    # YAML / output options
    yo = p.add_argument_group("output options")
    yo.add_argument("--yaml-path", default=DEFAULT_YAML,
                    help="Results YAML file (default: %(default)s)")
    yo.add_argument("--yaml-group", default=None,
                    help="Group name for new models (default: 'New Results')")

    args = p.parse_args()

    # Validation
    if args.parse_only and args.no_parse:
        p.error("--parse-only and --no-parse are mutually exclusive")
    if args.upload and args.no_parse:
        p.error("--upload requires parse (incompatible with --no-parse)")
    if args.preview and args.no_parse:
        p.error("--preview requires parse (incompatible with --no-parse)")

    return args


def main():
    args = parse_args()
    config = load_eval_config(args.config)
    model_names = [n.strip() for n in args.models.split(",")]
    output_dir = config.get("defaults", {}).get("output_dir", "logs/med_eval_mini")

    for model_name in model_names:
        model_config = find_model_in_config(config, model_name)
        if model_config is None:
            log.error("Model '%s' not found in config %s", model_name, args.config)
            sys.exit(1)

        params = model_config.get("params", "?")

        log.info("=" * 60)
        log.info("Pipeline: %s (params=%s)", model_name, params)
        log.info("  judge=%s parse=%s upload=%s",
                 not args.no_judge and not args.parse_only,
                 not args.no_parse,
                 args.upload)
        log.info("=" * 60)

        # Phase 1: Eval
        if not args.skip_eval and not args.parse_only:
            ok = run_eval(args.config, model_name, args.batch_size, args.limit, args.dry_run)
            if not ok:
                log.error("Eval failed for %s", model_name)
                sys.exit(1)
        else:
            log.info("Phase 1: EVAL — skipped")

        # Find results dir (needed for judge and/or parse)
        if not args.dry_run:
            try:
                results_dir = find_results_dir(output_dir, model_name)
                log.info("Results dir: %s", results_dir)
            except FileNotFoundError as e:
                log.error(str(e))
                sys.exit(1)
        else:
            results_dir = f"{output_dir}/{model_name}/????"

        # Assert *_results.json was emitted by lmms-eval, even under --no-parse.
        # This guarantees the raw metrics file is always on disk after Phase 1,
        # which matters for: (a) parse on a different machine later, (b)
        # uploading to dashboards, (c) re-running judge without re-running eval.
        if not args.dry_run and not args.parse_only:
            rj = sorted(Path(results_dir).glob("*_results.json"))
            if not rj:
                log.error("Phase 1 finished but no *_results.json found in %s", results_dir)
                sys.exit(1)
            log.info("  results.json: %s", rj[-1])

        # Phase 2: Judge
        if not args.no_judge and not args.parse_only:
            # Pre-judge gpu_clean: Phase 1 (eval subprocess) may have crashed or
            # left stale CUDA contexts. Clean them before spawning the 32B judge.
            # Safe here because pipeline.py itself has NOT imported torch (no
            # /dev/nvidia FDs in this process). Must NOT be done inside the judge
            # python process — see memory/feedback_gpu_clean_self_kill.md.
            # Skip when using remote-API judge (no local GPU use).
            if not args.judge_api_url:
                _pipeline_dir = Path(__file__).resolve().parent
                _gpu_clean = _pipeline_dir.parent / "scripts" / "gpu_clean.sh"
                if _gpu_clean.is_file():
                    log.info("Pre-judge GPU clean: %s", _gpu_clean)
                    try:
                        subprocess.run(["bash", str(_gpu_clean)],
                                       timeout=30, capture_output=True, check=False)
                    except Exception as e:
                        log.warning("Pre-judge GPU clean skipped: %s", e)

            try:
                judge_ok = run_judge(
                    results_dir,
                    args.judge_model,
                    args.serve_gpu_ids,
                    str(args.serve_port),
                    args.judge_workers,
                    args.dry_run,
                    judge_api_url=args.judge_api_url,
                    judge_api_type=args.judge_api_type,
                )
                if not judge_ok:
                    log.warning("Judge failed — continuing with '-' for judge metrics")
            except Exception as e:
                log.warning("Judge error: %s — continuing with '-'", e)
        else:
            log.info("Phase 2: JUDGE — skipped")

        # Phase 3: Parse
        if not args.no_parse:
            log.info("Phase 3: PARSE — %s", model_name)
            if not args.dry_run:
                metrics = parse_results(results_dir)
                log.info("  Metrics: %s", metrics)
                update_yaml(args.yaml_path, model_name, params, metrics, args.yaml_group)
            else:
                log.info("  [DRY-RUN] Would parse and update %s", args.yaml_path)
        else:
            log.info("Phase 3: PARSE — skipped (--no-parse)")

    # Preview (after all models parsed)
    if args.preview and not args.no_parse:
        preview_results(args.yaml_path, model_names)

    # Phase 4: Upload (once, after all models)
    if args.upload:
        run_upload(args.yaml_path, args.dry_run)
    else:
        log.info("Phase 4: UPLOAD — skipped")

    log.info("=" * 60)
    log.info("Pipeline complete: %s", ", ".join(model_names))
    log.info("=" * 60)


if __name__ == "__main__":
    main()
