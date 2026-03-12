"""Post-hoc RaTEScore computation for existing evaluation samples.

Reads saved JSONL sample files from a completed evaluation run,
recomputes RaTEScore for report generation tasks, and updates results.

Usage:
    python scripts/posthoc_ratescore.py \
        --results_dir logs/med_eval/Qwen3-VL-30B-A3B-Instruct/Qwen__Qwen3-VL-30B-A3B-Instruct \
        --batch_size 64
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lmms_eval.tasks._task_utils.report_metrics import calculate_ratescore

# Tasks that use RaTEScore
REPORT_TASKS = ["mimic_cxr", "iu_xray"]


def find_sample_files(results_dir: str) -> dict[str, Path]:
    """Find all JSONL sample files in results directory."""
    results_path = Path(results_dir)
    files = {}
    for f in sorted(results_path.glob("*_samples_*.jsonl")):
        name = f.stem
        parts = name.split("_samples_", 1)
        if len(parts) == 2:
            task_name = parts[1]
            files[task_name] = f
    return files


def run_ratescore_on_task(
    filepath: Path,
    task_name: str,
    batch_size: int = 64,
    force: bool = False,
) -> tuple[int, float, float]:
    """Compute RaTEScore for a task's samples.

    Returns:
        (total, mean_score, stderr)
    """
    samples = []
    with open(filepath) as f:
        for line in f:
            samples.append(json.loads(line))

    total = len(samples)

    # Check if already computed (non-zero ratescore)
    has_ratescore = [s.get("ratescore", 0.0) > 0 for s in samples]
    if all(has_ratescore) and not force:
        scores = [s["ratescore"] for s in samples]
        mean_score = sum(scores) / len(scores)
        stderr = float(np.std(scores, ddof=1) / np.sqrt(len(scores)))
        print(f"  [{task_name}] All {total} samples already have RaTEScore, skipping.")
        print(f"  Mean: {mean_score:.2f} ± {stderr:.2f}")
        return total, mean_score, stderr

    # Extract candidates and references
    candidates = []
    references = []
    for sample in samples:
        pred = sample.get("filtered_resps", "")
        if isinstance(pred, list):
            pred = pred[0] if pred else ""
        target = sample.get("target", "")

        candidates.append(pred)
        references.append(target)

    print(f"  [{task_name}] Computing RaTEScore for {total} samples...")

    # Compute in batches
    all_scores = []
    for i in range(0, total, batch_size):
        batch_cands = candidates[i : i + batch_size]
        batch_refs = references[i : i + batch_size]

        batch_scores = calculate_ratescore(batch_cands, batch_refs)
        if not batch_scores:
            print(f"  ERROR: RaTEScore computation failed. Is medspacy installed?")
            return total, 0.0, 0.0

        all_scores.extend(batch_scores)
        print(f"  Progress: {len(all_scores)}/{total}")

    # Update samples
    for idx, score in enumerate(all_scores):
        samples[idx]["ratescore"] = score

    # Write updated samples
    with open(filepath, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    mean_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    stderr = (
        float(np.std(all_scores, ddof=1) / np.sqrt(len(all_scores)))
        if len(all_scores) > 1
        else 0.0
    )

    print(f"  [{task_name}] Done. Mean: {mean_score:.2f} ± {stderr:.2f}")
    return total, mean_score, stderr


def update_results_json(
    results_dir: str,
    ratescore_results: dict[str, tuple[float, float, int]],
) -> None:
    """Update the results JSON file with RaTEScore values.

    Args:
        ratescore_results: {task_name: (mean_score, stderr, n_samples)}
    """
    results_path = Path(results_dir)
    json_files = sorted(results_path.glob("*_results.json"))
    if not json_files:
        print("Warning: No results JSON found.")
        return

    results_file = json_files[-1]
    with open(results_file) as f:
        results = json.load(f)

    # Update RaTEScore values
    for task_name, (score, stderr, n) in ratescore_results.items():
        if task_name in results.get("results", {}):
            results["results"][task_name]["ratescore,none"] = score
            results["results"][task_name]["ratescore_stderr,none"] = stderr
            results["results"][task_name]["ratescore_stderr_clt,none"] = stderr
            results["results"][task_name]["ratescore_stderr_clustered,none"] = "N/A"

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Updated {results_file}")


def main():
    parser = argparse.ArgumentParser(description="Post-hoc RaTEScore computation")
    parser.add_argument(
        "--results_dir",
        required=True,
        help="Path to model results directory containing JSONL samples",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for RaTEScore computation",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation even if RaTEScore already exists",
    )
    args = parser.parse_args()

    sample_files = find_sample_files(args.results_dir)
    print(f"Found {len(sample_files)} sample files in {args.results_dir}")

    ratescore_results: dict[str, tuple[float, float, int]] = {}

    print(f"\n=== RaTEScore Computation (batch_size={args.batch_size}) ===")
    for task_name in REPORT_TASKS:
        if task_name not in sample_files:
            print(f"  Skipping {task_name}: no sample file found")
            continue

        print(f"\nProcessing {task_name}...")
        total, mean_score, stderr = run_ratescore_on_task(
            sample_files[task_name],
            task_name,
            batch_size=args.batch_size,
            force=args.force,
        )
        ratescore_results[task_name] = (mean_score, stderr, total)

    if ratescore_results:
        update_results_json(args.results_dir, ratescore_results)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if ratescore_results:
        print("\nRaTEScore Results:")
        for task, (score, stderr, n) in sorted(ratescore_results.items()):
            print(f"  {task:15s}: {score:.2f} ± {stderr:.2f}  (n={n})")


if __name__ == "__main__":
    main()
