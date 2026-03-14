"""Print consolidated TSV results for all evaluated models.

Outputs three TSV tables (Multimodal VQA, Text QA, Report Generation)
that can be directly copied to Excel.

Usage:
    python scripts/print_all_results.py [logs_dir]

    Default logs_dir: logs/med_eval
"""

import argparse
import json
import sys
from pathlib import Path


# Task definitions matching print_results.py
MM_TASKS = [
    ("MMMU-Med", "mmmu_medical_val", "mmmu_acc,none"),
    ("VQA-RAD", "vqa_rad", "accuracy,none"),
    ("SLAKE", "slake", "accuracy,none"),
    ("PathVQA", "path_vqa", "accuracy,none"),
    ("PMC-VQA", "pmc_vqa", "accuracy,none"),
    ("VQA-Med", "vqa_med", "llm_judge,none"),
    ("OmniMedVQA", "omni_med_vqa", "accuracy,none"),
    ("MedXpertQA", "medxpertqa_mm", "accuracy,none"),
]

TEXT_TASKS = [
    ("MMLU-Med", "mmlu_medical", "accuracy,none"),
    ("PubMedQA", "pubmedqa", "accuracy,none"),
    ("MedMCQA", "medmcqa", "accuracy,none"),
    ("MedQA", "medqa_usmle", "accuracy,none"),
    ("MedBullets-4", "medbullets_op4", "accuracy,none"),
    ("MedBullets-5", "medbullets_op5", "accuracy,none"),
    ("MedXpertQA", "medxpertqa_text", "accuracy,none"),
    ("SuperGPQA", "supergpqa_medicine", "accuracy,none"),
]

REPORT_TASKS = [("IU-XRay", "iu_xray"), ("MIMIC-CXR", "mimic_cxr")]
REPORT_METRICS = ["bleu_4", "rouge_l", "meteor", "ratescore"]
REPORT_NAMES = {
    "bleu_4": "BLEU4",
    "rouge_l": "ROUGE-L",
    "meteor": "METEOR",
    "ratescore": "RaTEScore",
}

# Groups requiring LLM judge for valid group accuracy
GROUPS_REQUIRING_LLM_JUDGE = {
    "vqa_rad": "vqa_rad_open",
    "slake": "slake_open",
    "path_vqa": "path_vqa_open",
}


def load_results(log_dir: Path) -> tuple[dict, str] | None:
    """Load results JSON and return (data, model_name)."""
    for pattern in ["*_results.json", "*/*_results.json"]:
        json_files = sorted(log_dir.glob(pattern))
        if json_files:
            break
    else:
        return None

    results_file = json_files[-1]
    model_name = results_file.parent.name.replace("__", "/")

    # Simplify model name
    if model_name == log_dir.name:
        model_name = log_dir.name

    with open(results_file) as f:
        data = json.load(f)
    return data, model_name


def get_value(
    data: dict, key: str, metric: str, judge_model: str | None = None
) -> float | None:
    """Get metric value, checking LLM judge requirement for groups."""
    # Check if group requires LLM judge
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
            return val
        return None

    val = data.get("results", {}).get(key, {}).get(metric)
    if isinstance(val, (int, float)):
        return val
    return None


def fmt(val: float | None) -> str:
    """Format value for TSV output."""
    return f"{val:.2f}" if val is not None else "-"


def compute_avg(values: list[float | None]) -> float | None:
    """Compute average of non-None values."""
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if valid else None


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Print consolidated TSV results for all evaluated models."
        )
    )
    parser.add_argument(
        "logs_dir",
        nargs="?",
        default="logs/med_eval",
        help="Path to logs directory (default: logs/med_eval)",
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

    logs_dir = Path(args.logs_dir)

    if not logs_dir.exists():
        print(f"Error: {logs_dir} does not exist")
        sys.exit(1)

    # Collect all model results
    all_results = []
    for model_dir in sorted(logs_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        result = load_results(model_dir)
        if result:
            all_results.append(result)

    if not all_results:
        print(f"No results found in {logs_dir}")
        sys.exit(1)

    print(f"Found {len(all_results)} models\n")

    # === Multimodal VQA ===
    print("=" * 60)
    print("Multimodal VQA")
    print("=" * 60)

    headers = ["Model"] + [t[0] for t in MM_TASKS] + ["Avg"]
    print("\t".join(headers))

    for data, model_name in all_results:
        values = [
            get_value(data, key, metric, args.judge_model)
            for _, key, metric in MM_TASKS
        ]
        avg = compute_avg(values)
        row = [model_name] + [fmt(v) for v in values] + [fmt(avg)]
        print("\t".join(row))

    print()

    # === Text QA ===
    print("=" * 60)
    print("Text QA")
    print("=" * 60)

    headers = ["Model"] + [t[0] for t in TEXT_TASKS] + ["Avg"]
    print("\t".join(headers))

    for data, model_name in all_results:
        values = [
            get_value(data, key, metric, args.judge_model)
            for _, key, metric in TEXT_TASKS
        ]
        avg = compute_avg(values)
        row = [model_name] + [fmt(v) for v in values] + [fmt(avg)]
        print("\t".join(row))

    print()

    # === Report Generation ===
    print("=" * 60)
    print("Report Generation")
    print("=" * 60)

    # Build headers
    headers = ["Model"]
    for task_name, _ in REPORT_TASKS:
        for metric in REPORT_METRICS:
            headers.append(f"{task_name}-{REPORT_NAMES[metric]}")
    for metric in REPORT_METRICS:
        headers.append(f"Avg-{REPORT_NAMES[metric]}")
    print("\t".join(headers))

    for data, model_name in all_results:
        row = [model_name]
        metric_values = {m: [] for m in REPORT_METRICS}

        for _, task_key in REPORT_TASKS:
            task_data = data.get("results", {}).get(task_key, {})
            for metric in REPORT_METRICS:
                val = task_data.get(f"{metric},none")
                if not isinstance(val, (int, float)):
                    val = None
                row.append(fmt(val))
                metric_values[metric].append(val)

        # Add averages
        for metric in REPORT_METRICS:
            avg = compute_avg(metric_values[metric])
            row.append(fmt(avg))

        print("\t".join(row))


if __name__ == "__main__":
    main()
