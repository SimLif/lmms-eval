"""Compare med_eval_mini vs med_eval (full) results.

Computes per-task and aggregate Spearman rank correlation across models
to validate that med_eval_mini preserves model rankings.

Usage:
    python scripts/compare_mini_full.py [full_dir] [mini_dir]

    Defaults: full_dir=logs/med_eval  mini_dir=logs/med_eval_mini
"""

import argparse
import json
import sys
from pathlib import Path

from scipy import stats


# Task mapping: (display_name, full_key, full_metric, mini_key, mini_metric)
TASK_MAP = [
    ("MMMU-Med", "mmmu_medical_val", "mmmu_acc,none", "mmmu_medical_val", "mmmu_acc,none"),
    ("VQA-RAD", "vqa_rad", "accuracy,none", "vqa_rad", "accuracy,none"),
    ("SLAKE", "slake", "accuracy,none", "slake", "accuracy,none"),
    ("PathVQA", "path_vqa", "accuracy,none", "path_vqa_mini", "accuracy,none"),
    ("PMC-VQA", "pmc_vqa", "accuracy,none", "pmc_vqa", "accuracy,none"),
    ("VQA-Med", "vqa_med", "llm_judge,none", "vqa_med", "llm_judge,none"),
    ("OmniMedVQA", "omni_med_vqa", "accuracy,none", "omni_med_vqa_mini_v2", "accuracy,none"),
    ("MedXpertQA-MM", "medxpertqa_mm", "accuracy,none", "medxpertqa_mm", "accuracy,none"),
    ("PubMedQA", "pubmedqa", "accuracy,none", "pubmedqa", "accuracy,none"),
]

# Groups requiring LLM judge for valid group accuracy
GROUPS_REQUIRING_LLM_JUDGE = {
    "vqa_rad": "vqa_rad_open",
    "slake": "slake_open",
    "path_vqa": "path_vqa_open",
    "path_vqa_mini": "path_vqa_mini_open",
}


def load_latest_results(log_dir: Path) -> dict[str, dict]:
    """Load the latest results JSON for each model subdirectory."""
    models = {}
    for model_dir in sorted(log_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        json_files = []
        for pattern in ["*_results.json", "*/*_results.json"]:
            json_files.extend(model_dir.glob(pattern))
        if not json_files:
            continue
        latest = sorted(json_files)[-1]
        with open(latest) as f:
            data = json.load(f)
        models[model_dir.name] = data
    return models


def get_value(
    data: dict,
    key: str,
    metric: str,
    judge_model: str | None = None,
) -> float | None:
    """Extract metric value with LLM judge validation."""
    if key in GROUPS_REQUIRING_LLM_JUDGE:
        open_task = GROUPS_REQUIRING_LLM_JUDGE[key]
        open_result = data.get("results", {}).get(open_task, {})
        if "llm_judge,none" not in open_result:
            return None

    # When judge_model is specified and metric is llm_judge, read from
    # llm_judges[judge_model] instead of the default llm_judge,none
    if judge_model is not None and metric == "llm_judge,none":
        val = data.get("results", {}).get(key, {}).get(
            "llm_judges", {}
        ).get(judge_model)
        if isinstance(val, (int, float)):
            return float(val)
        return None

    val = data.get("results", {}).get(key, {}).get(metric)
    if isinstance(val, (int, float)):
        return float(val)
    return None


def compute_spearman(
    full_scores: list[float], mini_scores: list[float],
) -> tuple[float, float, int]:
    """Compute Spearman rank correlation. Returns (rho, p_value, n)."""
    n = len(full_scores)
    if n < 3:
        return float("nan"), float("nan"), n
    result = stats.spearmanr(full_scores, mini_scores)
    return result.statistic, result.pvalue, n


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare med_eval_mini vs med_eval (full) results via "
            "Spearman rank correlation."
        )
    )
    parser.add_argument(
        "full_dir",
        nargs="?",
        default="logs/med_eval",
        help="Path to full eval logs directory (default: logs/med_eval)",
    )
    parser.add_argument(
        "mini_dir",
        nargs="?",
        default="logs/med_eval_mini",
        help=(
            "Path to mini eval logs directory "
            "(default: logs/med_eval_mini)"
        ),
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default=None,
        help=(
            "Judge model name to read llm_judge scores from "
            "(reads from llm_judges[judge_model] instead of llm_judge,none)"
        ),
    )
    args = parser.parse_args()

    full_dir = Path(args.full_dir)
    mini_dir = Path(args.mini_dir)

    for d in [full_dir, mini_dir]:
        if not d.exists():
            print(f"Error: {d} does not exist")
            sys.exit(1)

    full_models = load_latest_results(full_dir)
    mini_models = load_latest_results(mini_dir)

    # Find common models
    common = sorted(set(full_models) & set(mini_models))
    if not common:
        print("No common models found between full and mini results.")
        sys.exit(1)

    print(
        f"Full models: {len(full_models)}, Mini models: {len(mini_models)}, "
        f"Common: {len(common)}"
    )
    print()

    # === Per-task correlation ===
    print("=" * 70)
    print("Per-Task Spearman Rank Correlation (Mini vs Full)")
    print("=" * 70)
    print(f"{'Task':<16} {'ρ':>8} {'p-value':>10} {'N':>4}  {'Note'}")
    print("-" * 70)

    task_rhos = []
    per_task_full_scores = {}
    per_task_mini_scores = {}

    for name, fkey, fmetric, mkey, mmetric in TASK_MAP:
        full_vals = []
        mini_vals = []
        valid_models = []

        for model in common:
            fv = get_value(
                full_models[model], fkey, fmetric, args.judge_model
            )
            mv = get_value(
                mini_models[model], mkey, mmetric, args.judge_model
            )
            if fv is not None and mv is not None:
                full_vals.append(fv)
                mini_vals.append(mv)
                valid_models.append(model)

        per_task_full_scores[name] = (valid_models, full_vals)
        per_task_mini_scores[name] = (valid_models, mini_vals)

        note = ""
        if fkey == mkey:
            note = "(identical task)"
        else:
            note = f"({mkey} vs {fkey})"

        if len(full_vals) < 3:
            print(
                f"{name:<16} {'N/A':>8} {'N/A':>10} "
                f"{len(full_vals):>4}  {note}"
            )
            continue

        rho, pval, n = compute_spearman(full_vals, mini_vals)
        task_rhos.append(rho)
        print(f"{name:<16} {rho:>8.4f} {pval:>10.4f} {n:>4}  {note}")

    print()

    # === Aggregate correlation ===
    print("=" * 70)
    print("Aggregate Correlation (Average score across tasks)")
    print("=" * 70)

    # For each model, compute average full score and average mini score
    # Only use tasks where both are available
    full_avgs = []
    mini_avgs = []
    agg_models = []

    for model in common:
        f_scores = []
        m_scores = []
        for name, fkey, fmetric, mkey, mmetric in TASK_MAP:
            fv = get_value(
                full_models[model], fkey, fmetric, args.judge_model
            )
            mv = get_value(
                mini_models[model], mkey, mmetric, args.judge_model
            )
            if fv is not None and mv is not None:
                f_scores.append(fv)
                m_scores.append(mv)
        if len(f_scores) >= 5:  # Require at least 5 tasks for meaningful avg
            full_avgs.append(sum(f_scores) / len(f_scores))
            mini_avgs.append(sum(m_scores) / len(m_scores))
            agg_models.append(model)

    if len(full_avgs) >= 3:
        rho, pval, n = compute_spearman(full_avgs, mini_avgs)
        print(f"Aggregate ρ = {rho:.4f}  (p = {pval:.4f}, N = {n} models)")
    else:
        print(
            f"Not enough models with sufficient data (have {len(full_avgs)})"
        )

    print()

    # === Model-by-model comparison ===
    print("=" * 70)
    print("Model Scores (Full avg → Mini avg)")
    print("=" * 70)
    print(f"{'Model':<35} {'Full':>8} {'Mini':>8} {'Δ':>8}")
    print("-" * 70)

    for model, favg, mavg in sorted(
        zip(agg_models, full_avgs, mini_avgs), key=lambda x: -x[1],
    ):
        delta = mavg - favg
        print(f"{model:<35} {favg:>8.2f} {mavg:>8.2f} {delta:>+8.2f}")

    print()

    # === Summary ===
    if task_rhos:
        avg_rho = sum(task_rhos) / len(task_rhos)
        print(f"Mean per-task ρ: {avg_rho:.4f}")
        sub = [r for r, (n, fk, fm, mk, mm) in zip(task_rhos, TASK_MAP)
               if fk != mk]
        if sub:
            print(
                f"Mean ρ (subsampled tasks only): {sum(sub)/len(sub):.4f}"
            )


if __name__ == "__main__":
    main()
