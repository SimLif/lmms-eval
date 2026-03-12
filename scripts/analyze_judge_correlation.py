"""Analyze correlation between traditional metrics and LLM judge scores.

Goal: Find low-cost metrics that correlate well with Sonnet llm_judge,
so we can use them during exploration phase instead of expensive API calls.

Usage:
    python scripts/analyze_judge_correlation.py
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

# Optional: BERTScore (requires GPU)
try:
    from bert_score import score as bert_score
    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False
    print("Note: bert_score not available, skipping BERTScore analysis")


def tokenize(text: str) -> list[str]:
    """Simple word tokenization (returns list for order-sensitive metrics)."""
    return re.findall(r'\w+', text.lower())


def tokenize_set(text: str) -> set[str]:
    """Simple word tokenization (returns set for overlap metrics)."""
    return set(re.findall(r'\w+', text.lower()))


def calc_precision(pred: str, target: str) -> float:
    """Calculate token-level precision."""
    pred_tokens = tokenize_set(pred)
    target_tokens = tokenize_set(target)
    if not pred_tokens:
        return 0.0
    common = pred_tokens & target_tokens
    return len(common) / len(pred_tokens)


def calc_recall(pred: str, target: str) -> float:
    """Calculate token-level recall."""
    pred_tokens = tokenize_set(pred)
    target_tokens = tokenize_set(target)
    if not target_tokens:
        return 0.0
    common = pred_tokens & target_tokens
    return len(common) / len(target_tokens)


def calc_f1(pred: str, target: str) -> float:
    """Calculate token-level F1 score."""
    precision = calc_precision(pred, target)
    recall = calc_recall(pred, target)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def calc_exact_match(pred: str, target: str) -> float:
    """Check if prediction exactly matches or contains the target."""
    pred_lower = pred.lower().strip()
    target_lower = target.lower().strip()
    if pred_lower == target_lower:
        return 1.0
    if target_lower in pred_lower:
        return 1.0
    return 0.0


def calc_bleu1(pred: str, target: str) -> float:
    """Calculate BLEU-1 (unigram precision with brevity penalty)."""
    pred_tokens = tokenize(pred)
    target_tokens = tokenize(target)
    if not pred_tokens or not target_tokens:
        return 0.0

    # Unigram precision
    pred_counter = {}
    for t in pred_tokens:
        pred_counter[t] = pred_counter.get(t, 0) + 1
    target_counter = {}
    for t in target_tokens:
        target_counter[t] = target_counter.get(t, 0) + 1

    clipped = sum(min(pred_counter.get(t, 0), c) for t, c in target_counter.items())
    precision = clipped / len(pred_tokens) if pred_tokens else 0

    # Brevity penalty
    bp = min(1.0, np.exp(1 - len(target_tokens) / len(pred_tokens))) if pred_tokens else 0
    return bp * precision


def calc_rouge_l(pred: str, target: str) -> float:
    """Calculate ROUGE-L (longest common subsequence)."""
    pred_tokens = tokenize(pred)
    target_tokens = tokenize(target)
    if not pred_tokens or not target_tokens:
        return 0.0

    # LCS length using DP
    m, n = len(pred_tokens), len(target_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i-1] == target_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]

    # F1-style combination
    prec = lcs / m if m > 0 else 0
    rec = lcs / n if n > 0 else 0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def calc_jaccard(pred: str, target: str) -> float:
    """Calculate Jaccard similarity."""
    pred_tokens = tokenize_set(pred)
    target_tokens = tokenize_set(target)
    if not pred_tokens and not target_tokens:
        return 0.0
    union = pred_tokens | target_tokens
    if not union:
        return 0.0
    intersection = pred_tokens & target_tokens
    return len(intersection) / len(union)


def load_samples(full_dir: Path) -> dict[str, list[dict]]:
    """Load all samples with llm_judge from full evaluation results."""
    open_tasks = ["vqa_rad_open", "slake_open", "vqa_med", "path_vqa_open"]
    all_samples = defaultdict(list)

    for model_dir in sorted(full_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        for task in open_tasks:
            for f in model_dir.rglob(f"*_samples_{task}.jsonl"):
                with open(f) as fp:
                    for line in fp:
                        sample = json.loads(line)
                        if "llm_judge" in sample:
                            sample["_model"] = model_dir.name
                            all_samples[task].append(sample)

    return dict(all_samples)


def analyze_task(task_name: str, samples: list[dict]) -> dict:
    """Analyze correlation between metrics and llm_judge for a task."""

    # All metrics to evaluate
    METRICS = {
        "precision": calc_precision,
        "recall": calc_recall,
        "f1": calc_f1,
        "exact_match": calc_exact_match,
        "bleu1": calc_bleu1,
        "rouge_l": calc_rouge_l,
        "jaccard": calc_jaccard,
    }

    # Extract data
    llm_judges = []
    metric_scores = {name: [] for name in METRICS}

    for s in samples:
        llm_judge = s.get("llm_judge", 0)
        target = s.get("target", "")

        # Get prediction (handle different formats)
        pred = s.get("filtered_resps", "")
        if isinstance(pred, list):
            pred = pred[0] if pred else ""

        llm_judges.append(llm_judge)
        for name, func in METRICS.items():
            metric_scores[name].append(func(pred, target) * 100)

    # Calculate correlations
    results = {
        "n_samples": len(samples),
        "llm_judge_mean": np.mean(llm_judges),
    }

    for name, scores in metric_scores.items():
        rho, p_val = stats.spearmanr(llm_judges, scores)
        results[f"{name}_spearman_rho"] = rho
        results[f"{name}_p_value"] = p_val
        results[f"{name}_mean"] = np.mean(scores)

    return results


def analyze_model_ranking(all_samples: dict[str, list[dict]]) -> dict:
    """Analyze if metrics preserve model ranking (like llm_judge does)."""

    METRICS = {
        "precision": calc_precision,
        "recall": calc_recall,
        "f1": calc_f1,
        "exact_match": calc_exact_match,
        "bleu1": calc_bleu1,
        "rouge_l": calc_rouge_l,
        "jaccard": calc_jaccard,
    }

    # Aggregate scores per model per task
    model_scores = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for task_name, samples in all_samples.items():
        for s in samples:
            model = s["_model"]
            target = s.get("target", "")
            pred = s.get("filtered_resps", "")
            if isinstance(pred, list):
                pred = pred[0] if pred else ""

            model_scores[task_name][model]["llm_judge"].append(s.get("llm_judge", 0))
            for name, func in METRICS.items():
                model_scores[task_name][model][name].append(func(pred, target) * 100)

    # Calculate mean per model and correlate rankings
    results = {}
    for task_name, task_data in model_scores.items():
        models = sorted(task_data.keys())
        n_models = len(models)

        llm_judge_means = [np.mean(task_data[m]["llm_judge"]) for m in models]

        task_results = {"n_models": n_models}
        for metric_name in METRICS:
            metric_means = [np.mean(task_data[m][metric_name]) for m in models]
            rho, p_val = stats.spearmanr(llm_judge_means, metric_means)
            task_results[f"{metric_name}_rho"] = rho

        results[task_name] = task_results

    return results


def main():
    full_dir = Path("logs/med_eval")

    print("Loading samples with llm_judge...")
    all_samples = load_samples(full_dir)

    total = sum(len(s) for s in all_samples.values())
    print(f"Loaded {total} samples across {len(all_samples)} tasks\n")

    METRICS = ["precision", "recall", "f1", "exact_match", "bleu1", "rouge_l", "jaccard"]

    # Per-task sample-level correlation
    print("=" * 90)
    print("SAMPLE-LEVEL CORRELATION (Spearman ρ: Metric vs LLM Judge)")
    print("=" * 90)
    header = f"{'Task':<16} {'N':>7}"
    for m in METRICS:
        header += f" {m[:8]:>9}"
    print(header)
    print("-" * 90)

    for task_name, samples in sorted(all_samples.items()):
        results = analyze_task(task_name, samples)
        row = f"{task_name:<16} {results['n_samples']:>7}"
        for m in METRICS:
            rho = results.get(f"{m}_spearman_rho", 0)
            row += f" {rho:>9.4f}"
        print(row)

    # Model ranking correlation (most important!)
    print("\n" + "=" * 90)
    print("MODEL RANKING CORRELATION (Spearman ρ: Does metric preserve model order?)")
    print("=" * 90)
    header = f"{'Task':<16} {'N':>4}"
    for m in METRICS:
        header += f" {m[:8]:>9}"
    print(header)
    print("-" * 90)

    ranking_results = analyze_model_ranking(all_samples)
    avg_rho = {m: [] for m in METRICS}

    for task_name, r in sorted(ranking_results.items()):
        row = f"{task_name:<16} {r['n_models']:>4}"
        for m in METRICS:
            rho = r.get(f"{m}_rho", 0)
            avg_rho[m].append(rho)
            row += f" {rho:>9.4f}"
        print(row)

    # Average across tasks
    print("-" * 90)
    row = f"{'AVERAGE':<16} {'':>4}"
    for m in METRICS:
        row += f" {np.mean(avg_rho[m]):>9.4f}"
    print(row)

    print("\n" + "=" * 90)
    print("INTERPRETATION (Spearman ρ)")
    print("=" * 90)
    print("""
| ρ 范围    | 相关强度 |
|-----------|----------|
| 0.9+      | 非常强   |
| 0.7-0.9   | 强       |
| 0.5-0.7   | 中等     |
| 0.3-0.5   | 弱       |
| <0.3      | 很弱     |

★ MODEL RANKING CORRELATION 是最重要的指标
  - 它表示用该指标对模型排序是否与 LLM Judge 一致
  - 探索阶段应选择 AVERAGE 最高的指标
""")


if __name__ == "__main__":
    main()
