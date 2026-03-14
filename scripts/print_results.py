"""Print a human-readable summary of med-eval results.

Reads the results JSON from a log directory and prints three tables
plus a single-row TSV for each table (header + values) for Excel.

Usage:
    python scripts/print_results.py <log_dir>
"""

import argparse
import json
import sys
from pathlib import Path


# ── table definitions ────────────────────────────────────────────────
# (display_name, result_key, metric_key, stderr_key)

MM_TASKS = [
    ("MMMU-Med", "mmmu_medical_val", "mmmu_acc,none", "mmmu_acc_stderr,none"),
    ("VQA-RAD", "vqa_rad", "accuracy,none", "accuracy_stderr,none"),
    ("SLAKE", "slake", "accuracy,none", "accuracy_stderr,none"),
    ("PathVQA", "path_vqa", "accuracy,none", "accuracy_stderr,none"),
    ("PMC-VQA", "pmc_vqa", "accuracy,none", "accuracy_stderr,none"),
    ("VQA-Med", "vqa_med", "llm_judge,none", "llm_judge_stderr,none"),
    ("OmniMedVQA", "omni_med_vqa", "accuracy,none", "accuracy_stderr,none"),
    ("MedXpertQA", "medxpertqa_mm", "accuracy,none", "accuracy_stderr,none"),
]

# Groups that require LLM judge for open tasks to compute valid group accuracy
# Maps group_key -> open_task_key (to check if llm_judge exists)
GROUPS_REQUIRING_LLM_JUDGE = {
    "vqa_rad": "vqa_rad_open",
    "slake": "slake_open",
    "path_vqa": "path_vqa_open",
}

TEXT_TASKS = [
    ("MMLU-Med", "mmlu_medical", "accuracy,none", "accuracy_stderr,none"),
    ("PubMedQA", "pubmedqa", "accuracy,none", "accuracy_stderr,none"),
    ("MedMCQA", "medmcqa", "accuracy,none", "accuracy_stderr,none"),
    ("MedQA", "medqa_usmle", "accuracy,none", "accuracy_stderr,none"),
    ("MedBullets-4", "medbullets_op4", "accuracy,none", "accuracy_stderr,none"),
    ("MedBullets-5", "medbullets_op5", "accuracy,none", "accuracy_stderr,none"),
    ("MedXpertQA", "medxpertqa_text", "accuracy,none", "accuracy_stderr,none"),
    ("SuperGPQA", "supergpqa_medicine", "accuracy,none", "accuracy_stderr,none"),
]

REPORT_TASKS = [
    ("IU-XRay", "iu_xray"),
    ("MIMIC-CXR", "mimic_cxr"),
]
REPORT_METRICS = ["bleu_4", "rouge_l", "meteor", "ratescore"]
REPORT_NAMES = {
    "bleu_4": "BLEU4",
    "rouge_l": "ROUGE-L",
    "meteor": "METEOR",
    "ratescore": "RaTEScore",
}


# ── helpers ──────────────────────────────────────────────────────────

def _load_results(log_dir: str) -> tuple[dict, str]:
    """Find and load the most recent *_results.json. Returns (data, model_name)."""
    p = Path(log_dir)
    for pattern in [
        "*_results.json",
        "*/*_results.json",
        "*/*/*_results.json",
    ]:
        json_files = sorted(p.glob(pattern))
        if json_files:
            break
    else:
        print(f"Error: no *_results.json found under {log_dir}")
        sys.exit(1)

    results_file = json_files[-1]
    # Derive model name from parent dir name (e.g. "Qwen__Qwen2.5-VL-3B-Instruct")
    model_name = results_file.parent.name.replace("__", "/")

    with open(results_file) as f:
        data = json.load(f)
    return data, model_name


def _get(
    results: dict, key: str, metric: str, stderr: str
) -> tuple[float | None, float | None]:
    """Extract (value, stderr) from results dict."""
    task = results.get("results", {}).get(key, {})
    val = task.get(metric)
    se = task.get(stderr)
    if isinstance(val, (list, str)) or val is None:
        return None, None
    if isinstance(se, (list, str)):
        se = None
    return val, se


def _v(val: float | None) -> str:
    return f"{val:.2f}" if val is not None else "-"


def _fmt(val: float | None, se: float | None) -> str:
    if val is None:
        return "-"
    if se is not None:
        return f"{val:.2f} ± {se:.2f}"
    return f"{val:.2f}"


# ── collect all values ───────────────────────────────────────────────

def _collect_acc(
    tasks: list[tuple[str, str, str, str]],
    data: dict,
    judge_model: str | None = None,
) -> list[tuple[str, float | None, float | None]]:
    rows = []
    for display, key, metric, stderr in tasks:
        # For groups requiring LLM judge, check if open task has llm_judge
        # If not, the group accuracy is incomplete (only closed) - show "-"
        if key in GROUPS_REQUIRING_LLM_JUDGE:
            open_task = GROUPS_REQUIRING_LLM_JUDGE[key]
            open_result = data.get("results", {}).get(open_task, {})
            if "llm_judge,none" not in open_result:
                # LLM judge missing for open task, group accuracy is incomplete
                rows.append((display, None, None))
                continue

        # When judge_model is specified and metric is llm_judge, read from
        # llm_judges[judge_model] instead of the default llm_judge,none
        if judge_model is not None and metric == "llm_judge,none":
            task_dict = data.get("results", {}).get(key, {})
            val = task_dict.get("llm_judges", {}).get(judge_model)
            if isinstance(val, (int, float)):
                rows.append((display, val, None))
            else:
                rows.append((display, None, None))
            continue

        val, se = _get(data, key, metric, stderr)
        rows.append((display, val, se))
    return rows


def _avg(rows: list[tuple[str, float | None, float | None]]) -> float | None:
    valid = [v for _, v, _ in rows if v is not None]
    return sum(valid) / len(valid) if valid else None


def _collect_report(data: dict) -> dict[str, list[tuple[str, float | None]]]:
    """Returns {metric: [(task_display, value), ...]}."""
    out: dict[str, list[tuple[str, float | None]]] = {
        m: [] for m in REPORT_METRICS
    }
    for display, key in REPORT_TASKS:
        task = data.get("results", {}).get(key, {})
        for m in REPORT_METRICS:
            val = task.get(f"{m},none")
            if not isinstance(val, (int, float)):
                val = None
            out[m].append((display, val))
    return out


def _report_avg(vals: list[tuple[str, float | None]]) -> float | None:
    valid = [v for _, v in vals if v is not None]
    return sum(valid) / len(valid) if valid else None


# ── pretty print ─────────────────────────────────────────────────────

def _print_acc_table(
    title: str,
    rows: list[tuple[str, float | None, float | None]],
    avg: float | None,
) -> None:
    print(f"{'=' * 52}")
    print(f"  {title}")
    print(f"{'=' * 52}")
    print(f"  {'Task':<16s} {'Acc ± Stderr':>18s}")
    print(f"  {'-' * 16} {'-' * 18}")
    for display, val, se in rows:
        print(f"  {display:<16s} {_fmt(val, se):>18s}")
    print(f"  {'-' * 16} {'-' * 18}")
    print(f"  {'Avg':<16s} {_fmt(avg, None):>18s}")
    print()


def _print_report_table(
    report: dict[str, list[tuple[str, float | None]]],
) -> None:
    print(f"{'=' * 68}")
    print(f"  Report Generation")
    print(f"{'=' * 68}")
    header = f"  {'Task':<12s}"
    for m in REPORT_METRICS:
        header += f" {REPORT_NAMES[m]:>12s}"
    print(header)
    print(f"  {'-' * 12}" + f" {'-' * 12}" * len(REPORT_METRICS))

    n_tasks = len(REPORT_TASKS)
    for i in range(n_tasks):
        display = REPORT_TASKS[i][0]
        line = f"  {display:<12s}"
        for m in REPORT_METRICS:
            _, val = report[m][i]
            line += f" {_v(val):>12s}"
        print(line)

    print(f"  {'-' * 12}" + f" {'-' * 12}" * len(REPORT_METRICS))
    line = f"  {'Avg':<12s}"
    for m in REPORT_METRICS:
        a = _report_avg(report[m])
        line += f" {_v(a):>12s}"
    print(line)
    print()


# ── single-row TSV ───────────────────────────────────────────────────

def _print_tsv_acc(
    title: str,
    model: str,
    rows: list[tuple[str, float | None, float | None]],
    avg: float | None,
) -> None:
    """Print header + value row for an accuracy table."""
    headers = ["Model"] + [d for d, _, _ in rows] + ["Avg"]
    values = [model] + [_v(v) for _, v, _ in rows] + [_v(avg)]
    print(f"[TSV: {title}]")
    print("\t".join(headers))
    print("\t".join(values))
    print()


def _print_tsv_report(
    model: str,
    report: dict[str, list[tuple[str, float | None]]],
) -> None:
    """Print header + value row for report generation."""
    headers = ["Model"]
    values = [model]
    for i, (display, _) in enumerate(REPORT_TASKS):
        for m in REPORT_METRICS:
            headers.append(f"{display}-{REPORT_NAMES[m]}")
            _, val = report[m][i]
            values.append(_v(val))
    # Per-metric avg
    for m in REPORT_METRICS:
        headers.append(f"Avg-{REPORT_NAMES[m]}")
        values.append(_v(_report_avg(report[m])))
    print("[TSV: Report Generation]")
    print("\t".join(headers))
    print("\t".join(values))
    print()


# ── main ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print a human-readable summary of med-eval results."
    )
    parser.add_argument(
        "log_dir",
        help="Path to log directory containing *_results.json",
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

    data, model = _load_results(args.log_dir)
    print(f"Model: {model}\n")

    mm_rows = _collect_acc(MM_TASKS, data, judge_model=args.judge_model)
    mm_avg = _avg(mm_rows)
    text_rows = _collect_acc(TEXT_TASKS, data, judge_model=args.judge_model)
    text_avg = _avg(text_rows)
    report = _collect_report(data)

    _print_acc_table("Multimodal VQA", mm_rows, mm_avg)
    _print_acc_table("Text QA", text_rows, text_avg)
    _print_report_table(report)

    _print_tsv_acc("Multimodal VQA", model, mm_rows, mm_avg)
    _print_tsv_acc("Text QA", model, text_rows, text_avg)
    _print_tsv_report(model, report)


if __name__ == "__main__":
    main()
