"""Print consolidated TSV results for med_eval_mini.

Outputs a single TSV table for the 9 mini tasks.

Usage:
    python scripts/print_mini_results.py [logs_dir]

    Default logs_dir: logs/med_eval_mini
"""

import json
import sys
from pathlib import Path


MINI_TASKS = [
    ("MMMU-Med", "mmmu_medical_val", "mmmu_acc,none"),
    ("VQA-RAD", "vqa_rad", "accuracy,none"),
    ("SLAKE", "slake", "accuracy,none"),
    ("PathVQA-Mini", "path_vqa_mini", "accuracy,none"),
    ("PMC-VQA", "pmc_vqa", "accuracy,none"),
    ("VQA-Med", "vqa_med", "llm_judge,none"),
    ("OmniMedVQA-Mini", "omni_med_vqa_mini_v2", "accuracy,none"),
    ("MedXpertQA-MM", "medxpertqa_mm", "accuracy,none"),
    ("PubMedQA", "pubmedqa", "accuracy,none"),
]

GROUPS_REQUIRING_LLM_JUDGE = {
    "vqa_rad": "vqa_rad_open",
    "slake": "slake_open",
    "path_vqa_mini": "path_vqa_mini_open",
}


def load_results(log_dir: Path) -> tuple[dict, str] | None:
    json_files = []
    for pattern in ["*_results.json", "*/*_results.json"]:
        json_files.extend(log_dir.glob(pattern))
    if not json_files:
        return None
    results_file = sorted(json_files)[-1]
    model_name = results_file.parent.name.replace("__", "/")
    if model_name == log_dir.name:
        model_name = log_dir.name
    with open(results_file) as f:
        data = json.load(f)
    return data, model_name


def get_value(data: dict, key: str, metric: str) -> float | None:
    if key in GROUPS_REQUIRING_LLM_JUDGE:
        open_task = GROUPS_REQUIRING_LLM_JUDGE[key]
        open_result = data.get("results", {}).get(open_task, {})
        if "llm_judge,none" not in open_result:
            return None
    val = data.get("results", {}).get(key, {}).get(metric)
    if isinstance(val, (int, float)):
        return val
    return None


def fmt(val: float | None) -> str:
    return f"{val:.2f}" if val is not None else "-"


def compute_avg(values: list[float | None]) -> float | None:
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if valid else None


def main() -> None:
    logs_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("logs/med_eval_mini")

    if not logs_dir.exists():
        print(f"Error: {logs_dir} does not exist")
        sys.exit(1)

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

    print("=" * 60)
    print("Med Eval Mini Results")
    print("=" * 60)

    headers = ["Model"] + [t[0] for t in MINI_TASKS] + ["Avg"]
    print("\t".join(headers))

    for data, model_name in all_results:
        values = [get_value(data, key, metric) for _, key, metric in MINI_TASKS]
        avg = compute_avg(values)
        row = [model_name] + [fmt(v) for v in values] + [fmt(avg)]
        print("\t".join(row))


if __name__ == "__main__":
    main()
