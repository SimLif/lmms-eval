"""Pipeline: eval → judge → parse → upload.

Orchestrates the full evaluation pipeline for a single model:
1. Run evaluation (run_eval.py)
2. Run LLM judge on open-ended tasks (posthoc_llm_judge.py)
3. Parse results and update local YAML
4. Upload to Google Sheet (optional)

Usage:
    # Full pipeline (eval + judge + parse, no upload)
    .venv/bin/python scripts/pipeline.py -c scripts/configs/med_eval_mini.yaml \
        --models Qwen3-VL-2B-Instruct

    # Skip eval, only judge + parse existing results
    .venv/bin/python scripts/pipeline.py -c scripts/configs/med_eval_mini.yaml \
        --models Qwen3-VL-2B-Instruct --skip-eval

    # No judge, parse with "-" for judge metrics
    .venv/bin/python scripts/pipeline.py -c scripts/configs/med_eval_mini.yaml \
        --models Qwen3-VL-2B-Instruct --no-judge

    # Full pipeline + upload
    .venv/bin/python scripts/pipeline.py -c scripts/configs/med_eval_mini.yaml \
        --models Qwen3-VL-2B-Instruct --upload
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
) -> bool:
    """Run posthoc LLM judge with auto-serve."""
    cmd = [
        sys.executable, "scripts/posthoc_llm_judge.py",
        "--results_dir", results_dir,
        "--run_judge",
        "--auto_serve",
        "--judge_model", judge_model,
        "--serve_gpu_ids", serve_gpu_ids,
        "--serve_port", str(serve_port),
        "--workers", str(workers),
    ]
    if dry_run:
        cmd.append("--dry_run")

    log.info("Phase 2: JUDGE — %s", results_dir)
    log.info("  judge_model: %s", judge_model)
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
    """Update or append model entry in the results YAML."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    # Search for existing model by name
    found = False
    for group in data["summary"]["groups"]:
        for model in group["models"]:
            if model["name"] == model_name:
                # Update metrics in place
                model.update(metrics)
                model["params"] = params
                found = True
                log.info("  Updated existing entry: %s in '%s'", model_name, group["name"])
                break
        if found:
            break

    if not found:
        # Append to specified group or default
        target_group_name = yaml_group or DEFAULT_GROUP
        target_group = None
        for group in data["summary"]["groups"]:
            if group["name"] == target_group_name:
                target_group = group
                break

        if target_group is None:
            # Create the group
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
    p = argparse.ArgumentParser(description="Eval pipeline: eval → judge → parse → upload")
    p.add_argument("-c", "--config", required=True, help="Eval config YAML")
    p.add_argument("--models", required=True, help="Model name (from eval config)")
    p.add_argument("--no-judge", action="store_true", help="Skip LLM judge (default: judge ON)")
    p.add_argument("--upload", action="store_true", help="Upload to Google Sheet (default: OFF)")
    p.add_argument("--skip-eval", action="store_true", help="Skip eval, use existing results")
    p.add_argument("--limit", type=int, default=None, help="Pass --limit to eval")
    p.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    p.add_argument("--dry-run", action="store_true", help="Dry run all phases")
    # Judge options
    p.add_argument("--judge-model", default="Qwen/Qwen3-VL-32B-Instruct")
    p.add_argument("--serve-gpu-ids", default="0,1,2,3")
    p.add_argument("--serve-port", type=int, default=8000)
    p.add_argument("--judge-workers", type=int, default=16)
    # YAML options
    p.add_argument("--yaml-path", default=DEFAULT_YAML, help="Results YAML file")
    p.add_argument("--yaml-group", default=None, help="Group for new models (default: 'New Results')")
    return p.parse_args()


def main():
    args = parse_args()
    config = load_eval_config(args.config)
    model_name = args.models
    output_dir = config.get("defaults", {}).get("output_dir", "logs/med_eval_mini")

    # Find model in config for params
    model_config = find_model_in_config(config, model_name)
    if model_config is None:
        log.error("Model '%s' not found in config %s", model_name, args.config)
        sys.exit(1)

    params = model_config.get("params", "?")

    log.info("=" * 60)
    log.info("Pipeline: %s (params=%s)", model_name, params)
    log.info("  judge=%s upload=%s", not args.no_judge, args.upload)
    log.info("=" * 60)

    # Phase 1: Eval
    if not args.skip_eval:
        ok = run_eval(args.config, model_name, args.batch_size, args.limit, args.dry_run)
        if not ok:
            log.error("Eval failed for %s", model_name)
            sys.exit(1)
    else:
        log.info("Phase 1: EVAL — skipped (--skip-eval)")

    # Find results dir
    if not args.dry_run:
        try:
            results_dir = find_results_dir(output_dir, model_name)
            log.info("Results dir: %s", results_dir)
        except FileNotFoundError as e:
            log.error(str(e))
            sys.exit(1)
    else:
        results_dir = f"{output_dir}/{model_name}/????"

    # Phase 2: Judge
    judge_ok = True
    if not args.no_judge:
        try:
            judge_ok = run_judge(
                results_dir,
                args.judge_model,
                args.serve_gpu_ids,
                str(args.serve_port),
                args.judge_workers,
                args.dry_run,
            )
            if not judge_ok:
                log.warning("Judge failed — continuing with '-' for judge metrics")
        except Exception as e:
            log.warning("Judge error: %s — continuing with '-' for judge metrics", e)
            judge_ok = False
    else:
        log.info("Phase 2: JUDGE — skipped (--no-judge)")

    # Phase 3: Parse
    log.info("Phase 3: PARSE — %s", model_name)
    if not args.dry_run:
        metrics = parse_results(results_dir)
        log.info("  Metrics: %s", {k: v for k, v in metrics.items() if k != "avg"})
        log.info("  Avg: %s", metrics["avg"])

        update_yaml(args.yaml_path, model_name, params, metrics, args.yaml_group)
    else:
        log.info("  [DRY-RUN] Would parse %s and update %s", results_dir, args.yaml_path)

    # Phase 4: Upload
    if args.upload:
        run_upload(args.yaml_path, args.dry_run)
    else:
        log.info("Phase 4: UPLOAD — skipped (use --upload to enable)")

    log.info("=" * 60)
    log.info("Pipeline complete: %s", model_name)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
