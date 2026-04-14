"""Compare eval results with Google Sheet data."""
import json
import sys
from pathlib import Path

import yaml

METRIC_MAP = {
    "mmmu_medical_val": ("mmmu_acc,none", "mmmu_med"),
    "vqa_rad": ("accuracy,none", "vqa_rad"),
    "slake": ("accuracy,none", "slake"),
    "path_vqa_mini": ("accuracy,none", "pathvqa"),
    "pmc_vqa": ("accuracy,none", "pmc_vqa"),
    "vqa_med": ("llm_judge,none", "vqa_med"),
    "omni_med_vqa_mini_v2": ("accuracy,none", "omnimedvqa"),
    "medxpertqa_mm": ("accuracy,none", "medxpertqa"),
    "pubmedqa": ("accuracy,none", "pubmedqa"),
}

# Metrics that don't need LLM judge
NO_JUDGE = {"mmmu_med", "pmc_vqa", "omnimedvqa", "medxpertqa", "pubmedqa"}


def load_sheet_data(yaml_path: str) -> dict[str, dict]:
    """Load model data from Sheet YAML. Returns {model_name: {metric: value}}."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    models = {}
    for group in data["summary"]["groups"]:
        for m in group["models"]:
            models[m["name"]] = {
                k: m.get(k) for k in [
                    "mmmu_med", "vqa_rad", "slake", "pathvqa", "pmc_vqa",
                    "vqa_med", "omnimedvqa", "medxpertqa", "pubmedqa", "avg",
                ]
            }
    return models


def load_eval_results(results_dir: str, model_name: str) -> dict | None:
    """Find and load results.json for a model."""
    base = Path(results_dir) / model_name
    if not base.exists():
        return None
    for f in base.rglob("*_results.json"):
        with open(f) as fp:
            data = json.load(fp)
        return data.get("results", {})
    return None


def compare_model(
    model_name: str, sheet: dict, eval_results: dict
) -> list[tuple]:
    """Compare one model. Returns list of (metric, sheet_val, eval_val, delta, note)."""
    rows = []
    for task, (metric_key, sheet_name) in METRIC_MAP.items():
        sv = sheet.get(sheet_name)
        ev = eval_results.get(task, {}).get(metric_key)
        if ev is not None and sv is not None:
            delta = ev - sv
            note = ""
            if sheet_name not in NO_JUDGE and abs(delta) > 5:
                note = "no judge yet"
            elif abs(delta) > 2:
                note = "DIFF"
            rows.append((sheet_name, sv, ev, delta, note))
        elif ev is None:
            rows.append((sheet_name, sv, None, None, "needs judge"))
        else:
            rows.append((sheet_name, None, ev, None, "not in sheet"))
    return rows


def main():
    yaml_path = "data/eval_results/med_eval_results.yaml"
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "logs/med_eval_mini"

    sheet_data = load_sheet_data(yaml_path)
    target_models = [
        "Lingshu-7B", "Lingshu-32B", "Lingshu-I-8B",
        "Hulu-Med-30A3", "Hulu-Med-4B", "Hulu-Med-7B",
        "Hulu-Med-14B", "Hulu-Med-32B",
    ]

    for model_name in target_models:
        sheet = sheet_data.get(model_name)
        if not sheet:
            print(f"\n{model_name}: NOT IN SHEET")
            continue
        eval_results = load_eval_results(results_dir, model_name)
        if not eval_results:
            print(f"\n{model_name}: NO EVAL RESULTS")
            continue

        rows = compare_model(model_name, sheet, eval_results)
        print(f"\n{'='*65}")
        print(f" {model_name}")
        print(f"{'='*65}")
        print(f"{'Metric':12s} {'Sheet':>8s} {'Eval':>8s} {'Delta':>8s} {'Notes'}")
        print(f"{'-'*65}")
        for metric, sv, ev, delta, note in rows:
            sv_s = f"{sv:.2f}" if sv is not None else "N/A"
            ev_s = f"{ev:.2f}" if ev is not None else "N/A"
            d_s = f"{delta:+.2f}" if delta is not None else ""
            print(f"{metric:12s} {sv_s:>8s} {ev_s:>8s} {d_s:>8s} {note}")


if __name__ == "__main__":
    main()
