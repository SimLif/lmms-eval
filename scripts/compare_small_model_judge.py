"""Compare small model judge vs Sonnet llm_judge for correlation analysis.

This script:
1. Reads samples that already have Sonnet llm_judge scores
2. Runs a small model judge on the same samples
3. Calculates correlation between small model and Sonnet scores

Usage:
    # Start vLLM server first:
    # python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-VL-3B-Instruct --port 8000

    python scripts/compare_small_model_judge.py \
        --results_dir logs/med_eval \
        --small_model_url http://localhost:8000/v1 \
        --small_model_name qwen2.5-vl-3b-judge \
        --sample_size 500 \
        --workers 8
"""

import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lmms_eval.tasks._task_utils.answer_utils import parse_reasoning_answer

# Tasks that support LLM judge (open-ended VQA)
JUDGE_TASKS = [
    "vqa_rad_open",
    "slake_open",
    "path_vqa_open",
    "vqa_med",
]


def build_judge_prompt(question: str, answer: str, prediction: str) -> str:
    """Build the judge prompt (same as used in llm_judge)."""
    return f"""You are an expert medical evaluator. Determine if the prediction correctly answers the question based on the ground truth answer.

Question: {question}

Ground Truth Answer: {answer}

Model Prediction: {prediction}

Does the prediction correctly answer the question? Consider semantic equivalence, not just exact match.

Respond with ONLY a single number:
- 1 if the prediction is correct
- 0 if the prediction is incorrect

Your response (0 or 1):"""


def call_small_model(
    url: str,
    model_name: str,
    prompt: str,
    api_key: str = "EMPTY",
    timeout: int = 30,
) -> Optional[int]:
    """Call small model API and return 0 or 1."""
    import requests

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 10,
        "temperature": 0,
    }

    try:
        response = requests.post(
            f"{url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()

        # Parse response
        if "1" in content:
            return 1
        elif "0" in content:
            return 0
        else:
            return None
    except Exception as e:
        print(f"  API error: {e}")
        return None


def judge_single(
    idx: int,
    question: str,
    gt_ans: str,
    pred_ans: str,
    url: str,
    model_name: str,
) -> tuple[int, Optional[int]]:
    """Judge a single sample. Returns (idx, score)."""
    prompt = build_judge_prompt(question, gt_ans, pred_ans)
    score = call_small_model(url, model_name, prompt)
    return idx, score


def collect_samples_with_sonnet_judge(
    results_dir: str,
    task_name: str,
    sample_size: Optional[int] = None,
    sample_per_model: bool = False,
) -> list[dict]:
    """Collect samples that have Sonnet llm_judge scores.

    Args:
        results_dir: Path to results directory
        task_name: Name of the task
        sample_size: Total samples or samples per model (if sample_per_model=True)
        sample_per_model: If True, sample_size is per model; if False, total
    """
    results_path = Path(results_dir)
    samples_by_model = {}

    # Find all model directories
    for model_dir in sorted(results_path.iterdir()):
        if not model_dir.is_dir():
            continue

        # Find sample file for this task
        for sample_file in model_dir.rglob(f"*_samples_{task_name}.jsonl"):
            model_name = model_dir.name
            model_samples = []
            with open(sample_file) as f:
                for line in f:
                    sample = json.loads(line)
                    # Only include samples with llm_judge score
                    if "llm_judge" in sample:
                        sample["_model"] = model_name
                        sample["_file"] = str(sample_file)
                        model_samples.append(sample)
            if model_samples:
                samples_by_model[model_name] = model_samples
            break  # Only one file per model

    # Sample if needed
    if sample_size:
        if sample_per_model:
            # Sample per model
            for model in samples_by_model:
                if len(samples_by_model[model]) > sample_size:
                    samples_by_model[model] = random.sample(
                        samples_by_model[model], sample_size
                    )
        else:
            # Total sample across all models
            all_samples = []
            for model_samples in samples_by_model.values():
                all_samples.extend(model_samples)
            if len(all_samples) > sample_size:
                all_samples = random.sample(all_samples, sample_size)
            # Rebuild samples_by_model
            samples_by_model = {}
            for sample in all_samples:
                model = sample["_model"]
                if model not in samples_by_model:
                    samples_by_model[model] = []
                samples_by_model[model].append(sample)

    return samples_by_model


def run_comparison(
    results_dir: str,
    task_name: str,
    small_model_url: str,
    small_model_name: str,
    sample_size: Optional[int] = None,
    workers: int = 8,
) -> dict:
    """Run small model judge and compare with Sonnet scores."""
    print(f"\n=== {task_name} ===")

    # Collect samples by model
    samples_by_model = collect_samples_with_sonnet_judge(
        results_dir, task_name, sample_size, sample_per_model=True
    )
    total_samples = sum(len(s) for s in samples_by_model.values())
    print(f"  Models: {len(samples_by_model)}, Total samples: {total_samples}")

    if not samples_by_model:
        return {}

    # Flatten samples for processing
    all_samples = []
    sample_to_model = {}
    for model, model_samples in samples_by_model.items():
        for sample in model_samples:
            sample_to_model[len(all_samples)] = model
            all_samples.append(sample)

    # Prepare work items
    work_items = []
    for i, sample in enumerate(all_samples):
        question = sample.get("input", "")
        target = sample.get("target", "")
        pred = sample.get("filtered_resps", "")
        if isinstance(pred, list):
            pred = pred[0] if pred else ""

        pred_clean = parse_reasoning_answer(pred, strict=False)
        pred_ans = pred_clean.lower().strip().replace(".", "")
        gt_ans = target.lower().strip().replace(".", "")

        work_items.append((i, question, gt_ans, pred_ans))

    # Run small model judge with parallel workers
    small_scores = [None] * len(all_samples)
    completed = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                judge_single, idx, q, gt, pred, small_model_url, small_model_name
            ): idx
            for idx, q, gt, pred in work_items
        }

        for future in as_completed(futures):
            idx, score = future.result()
            small_scores[idx] = score
            completed += 1

            if completed % 100 == 0 or completed == len(work_items):
                valid = sum(1 for s in small_scores if s is not None)
                print(f"  Progress: {completed}/{len(work_items)} ({valid} valid)")

    # Aggregate scores by model
    model_sonnet_scores = {}
    model_small_scores = {}
    for i, sample in enumerate(all_samples):
        model = sample_to_model[i]
        if model not in model_sonnet_scores:
            model_sonnet_scores[model] = []
            model_small_scores[model] = []
        model_sonnet_scores[model].append(sample["llm_judge"])
        if small_scores[i] is not None:
            model_small_scores[model].append(small_scores[i] * 100)
        else:
            model_small_scores[model].append(None)

    # Calculate model-level averages
    model_sonnet_avg = {}
    model_small_avg = {}
    for model in model_sonnet_scores:
        model_sonnet_avg[model] = np.mean(model_sonnet_scores[model])
        valid_small = [s for s in model_small_scores[model] if s is not None]
        if valid_small:
            model_small_avg[model] = np.mean(valid_small)

    # Calculate model ranking correlation
    common_models = [m for m in model_sonnet_avg if m in model_small_avg]
    sonnet_ranks = [model_sonnet_avg[m] for m in common_models]
    small_ranks = [model_small_avg[m] for m in common_models]

    if len(common_models) >= 3:
        model_corr, model_p = stats.spearmanr(sonnet_ranks, small_ranks)
    else:
        model_corr, model_p = None, None

    # Sample-level stats
    sonnet_flat = []
    small_flat = []
    for i, sample in enumerate(all_samples):
        if small_scores[i] is not None:
            sonnet_flat.append(sample["llm_judge"])
            small_flat.append(small_scores[i] * 100)

    if len(sonnet_flat) >= 10:
        sample_corr, sample_p = stats.spearmanr(sonnet_flat, small_flat)
        agreement = sum(1 for s, sm in zip(sonnet_flat, small_flat) if s == sm) / len(sonnet_flat)
        sonnet_acc = np.mean(sonnet_flat)
        small_acc = np.mean(small_flat)
    else:
        sample_corr, sample_p, agreement, sonnet_acc, small_acc = None, None, None, None, None

    # Print results
    print(f"  Valid samples: {len(sonnet_flat)}")
    print(f"  Sonnet acc: {sonnet_acc:.1f}%")
    print(f"  Small model acc: {small_acc:.1f}%")
    print(f"  Agreement rate: {agreement:.1%}")
    print(f"  Sample correlation (rho): {sample_corr:.3f}")
    print(f"  Model ranking correlation (rho): {model_corr:.3f}" if model_corr else "  Model correlation: N/A")

    # Print per-model comparison
    print(f"\n  Per-model scores:")
    print(f"  {'Model':<35} {'Sonnet':>8} {'Small':>8} {'Diff':>8}")
    for model in sorted(common_models):
        diff = model_small_avg[model] - model_sonnet_avg[model]
        print(f"  {model:<35} {model_sonnet_avg[model]:>7.1f}% {model_small_avg[model]:>7.1f}% {diff:>+7.1f}%")

    return {
        "task": task_name,
        "n_samples": len(sonnet_flat),
        "n_models": len(common_models),
        "sonnet_acc": sonnet_acc,
        "small_acc": small_acc,
        "agreement": agreement,
        "sample_corr": sample_corr,
        "model_corr": model_corr,
        "model_scores": {
            "sonnet": model_sonnet_avg,
            "small": model_small_avg,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare small model judge vs Sonnet llm_judge"
    )
    parser.add_argument(
        "--results_dir",
        default="logs/med_eval",
        help="Path to results directory",
    )
    parser.add_argument(
        "--small_model_url",
        default="http://localhost:8000/v1",
        help="Small model API URL",
    )
    parser.add_argument(
        "--small_model_name",
        default="qwen2.5-vl-3b-judge",
        help="Small model name",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of samples per task (None for all)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Small model: {args.small_model_name}")
    print(f"API URL: {args.small_model_url}")
    print(f"Sample size: {args.sample_size or 'all'}")

    results = []
    for task_name in JUDGE_TASKS:
        result = run_comparison(
            results_dir=args.results_dir,
            task_name=task_name,
            small_model_url=args.small_model_url,
            small_model_name=args.small_model_name,
            sample_size=args.sample_size,
            workers=args.workers,
        )
        if result:
            results.append(result)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if results:
        print(f"\n{'Task':<20} {'N':>8} {'Models':>8} {'Agree':>8} {'Sample ρ':>10} {'Model ρ':>10}")
        print("-" * 80)
        total_n = 0
        weighted_sample_corr = 0
        model_corrs = []
        for r in results:
            model_corr_str = f"{r['model_corr']:.3f}" if r.get('model_corr') else "N/A"
            sample_corr_str = f"{r['sample_corr']:.3f}" if r.get('sample_corr') else "N/A"
            print(
                f"{r['task']:<20} {r['n_samples']:>8} {r.get('n_models', 0):>8} "
                f"{r['agreement']:>7.1%} {sample_corr_str:>10} {model_corr_str:>10}"
            )
            if r.get('sample_corr'):
                total_n += r["n_samples"]
                weighted_sample_corr += r["sample_corr"] * r["n_samples"]
            if r.get('model_corr'):
                model_corrs.append(r["model_corr"])

        print("-" * 80)
        avg_sample_corr = weighted_sample_corr / total_n if total_n > 0 else 0
        avg_model_corr = np.mean(model_corrs) if model_corrs else 0
        print(f"{'AVERAGE':<20} {total_n:>8} {'-':>8} {'-':>8} {avg_sample_corr:>10.3f} {avg_model_corr:>10.3f}")

        # Print overall model ranking across all tasks
        print("\n" + "=" * 80)
        print("OVERALL MODEL SCORES (averaged across tasks)")
        print("=" * 80)

        # Aggregate model scores
        all_models = set()
        for r in results:
            if "model_scores" in r:
                all_models.update(r["model_scores"]["sonnet"].keys())

        model_overall = {}
        for model in all_models:
            sonnet_scores = []
            small_scores = []
            for r in results:
                if "model_scores" not in r:
                    continue
                if model in r["model_scores"]["sonnet"]:
                    sonnet_scores.append(r["model_scores"]["sonnet"][model])
                if model in r["model_scores"]["small"]:
                    small_scores.append(r["model_scores"]["small"][model])
            if sonnet_scores and small_scores:
                model_overall[model] = {
                    "sonnet": np.mean(sonnet_scores),
                    "small": np.mean(small_scores),
                }

        # Calculate overall model ranking correlation
        models = list(model_overall.keys())
        sonnet_overall = [model_overall[m]["sonnet"] for m in models]
        small_overall = [model_overall[m]["small"] for m in models]
        if len(models) >= 3:
            overall_corr, _ = stats.spearmanr(sonnet_overall, small_overall)
            print(f"\nOverall model ranking correlation: ρ = {overall_corr:.3f}")
            print(f"(Compare with exact_match baseline: ρ = 0.86)")

        print(f"\n{'Model':<35} {'Sonnet':>10} {'Small':>10} {'Diff':>10}")
        print("-" * 70)
        for model in sorted(models, key=lambda m: model_overall[m]["sonnet"], reverse=True):
            diff = model_overall[model]["small"] - model_overall[model]["sonnet"]
            print(
                f"{model:<35} {model_overall[model]['sonnet']:>9.1f}% "
                f"{model_overall[model]['small']:>9.1f}% {diff:>+9.1f}%"
            )


if __name__ == "__main__":
    main()
