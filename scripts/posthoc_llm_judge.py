"""Post-hoc LLM judge scoring for existing evaluation samples.

Reads saved JSONL sample files from a completed evaluation run,
applies LLM judge to open-ended VQA tasks, and writes updated results.

Usage (remote API):
    export API_TYPE=anthropic
    export JUDGE_MODEL=claude-sonnet-4-5-20250929
    export ANTHROPIC_API_KEY=sk-xxx
    export ANTHROPIC_BASE_URL=http://...

    python scripts/posthoc_llm_judge.py \
        --results_dir logs/med_eval_mini/model_name \
        --run_judge \
        --workers 16

Usage (local vLLM):
    python scripts/posthoc_llm_judge.py \
        --results_dir logs/med_eval_mini/model_name \
        --run_judge \
        --judge_api_type openai \
        --judge_api_url http://localhost:8000/v1 \
        --judge_model Qwen/Qwen3-VL-32B \
        --workers 16
"""

import argparse
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lmms_eval.llm_judge import ServerConfig, get_server
from lmms_eval.llm_judge.base import ServerInterface
from lmms_eval.tasks._task_utils.answer_utils import parse_reasoning_answer

# Tasks that support LLM judge (open-ended VQA)
JUDGE_TASKS = [
    "vqa_rad_open",
    "slake_open",
    "path_vqa_open",
    "path_vqa_mini_open",
    "vqa_med",
]


def create_judge_server(args: argparse.Namespace) -> tuple[ServerInterface, str]:
    """Create judge server from CLI args, falling back to env vars.

    Returns:
        (server, judge_model_name)
    """
    api_type = args.judge_api_type or os.getenv("API_TYPE", "openai")
    model_name = args.judge_model or os.getenv("JUDGE_MODEL", "claude-sonnet-4-6")

    # For local models via OpenAI-compatible API (e.g. vLLM)
    if args.judge_api_url:
        os.environ["OPENAI_API_URL"] = args.judge_api_url
        if not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = "EMPTY"

    config = ServerConfig(model_name=model_name, temperature=0, max_tokens=256)
    server = get_server(server_name=api_type, config=config)
    return server, model_name


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


def _judge_single(
    idx: int,
    question: str,
    gt_ans: str,
    pred_ans: str,
    server: ServerInterface,
) -> tuple[int, float, str]:
    """Judge a single sample. Returns (idx, score, status)."""
    try:
        result = server.evaluate_binary(
            question=question,
            answer=gt_ans,
            prediction=pred_ans,
            output_format="0/1",
        )
    except Exception as e:
        print(f"  Judge call failed (all retries exhausted): {e!s}")
        return idx, -1.0, "timeout"

    if result.get("success") and result.get("result") is not None:
        return idx, 100.0 if result["result"] == 1 else 0.0, "ok"
    return idx, 0.0, "ok"


def run_judge_on_task(
    filepath: Path,
    task_name: str,
    server: ServerInterface,
    judge_model_name: str,
    workers: int = 16,
    dry_run: bool = False,
    force: bool = False,
) -> tuple[int, int, float, float]:
    """Apply LLM judge to a task's samples using parallel workers.

    Returns:
        (total, judged_correct, mean_score, stderr)
    """
    samples = []
    with open(filepath) as f:
        for line in f:
            samples.append(json.loads(line))

    total = len(samples)

    # Auto-migrate old format: llm_judge -> llm_judges dict
    for sample in samples:
        if "llm_judge" in sample and "llm_judges" not in sample:
            old_model = sample.get("llm_judge_model", "unknown")
            sample["llm_judges"] = {old_model: sample["llm_judge"]}

    # Skip check: count samples already judged by this specific model
    already_judged = sum(
        1 for s in samples
        if s.get("llm_judges", {}).get(judge_model_name) is not None
    )

    if already_judged == total and not force:
        scores = [
            s.get("llm_judges", {}).get(judge_model_name, 0.0)
            for s in samples
        ]
        correct = sum(1 for sc in scores if sc > 0)
        mean_score = sum(scores) / len(scores)
        stderr = float(np.std(scores, ddof=1) / np.sqrt(len(scores)))
        print(
            f"  [{task_name}] All {total} samples already judged "
            f"by {judge_model_name}, skipping."
        )
        return total, correct, mean_score, stderr

    # Force: only clear this judge's entry, not all judges
    if force and already_judged > 0:
        print(
            f"  [{task_name}] WARNING: --force overwriting {already_judged} "
            f"existing scores for {judge_model_name}"
        )
        for sample in samples:
            judges = sample.setdefault("llm_judges", {})
            if judge_model_name in judges:
                del judges[judge_model_name]
            # Clear backward-compat fields only if they point to current judge
            if sample.get("llm_judge_model") == judge_model_name:
                if "llm_judge" in sample:
                    del sample["llm_judge"]
                if "llm_judge_model" in sample:
                    del sample["llm_judge_model"]
        already_judged = 0

    # Prepare work items: skip samples already scored by this judge
    work_items = []
    for i, sample in enumerate(samples):
        if sample.get("llm_judges", {}).get(judge_model_name) is not None:
            continue  # Already judged by this model

        question = sample.get("input", "")
        target = sample.get("target", "")
        pred = sample.get("filtered_resps", "")
        # Flatten nested lists (e.g. chat models store [['answer']] not ['answer'])
        while isinstance(pred, list):
            pred = pred[0] if pred else ""

        pred_clean = parse_reasoning_answer(pred, strict=False)
        pred_ans = pred_clean.lower().strip().replace(".", "")
        gt_ans = target.lower().strip().replace(".", "")

        work_items.append((i, question, gt_ans, pred_ans))

    print(
        f"  [{task_name}] {len(work_items)} samples to judge"
        f" ({already_judged} already done)"
    )

    if dry_run:
        scores = [
            s.get("llm_judges", {}).get(judge_model_name, 0.0)
            for s in samples
        ]
        correct = sum(1 for sc in scores if sc > 0)
        mean_score = sum(scores) / len(scores) if scores else 0.0
        stderr = (
            float(np.std(scores, ddof=1) / np.sqrt(len(scores)))
            if len(scores) > 1
            else 0.0
        )
        return total, correct, mean_score, stderr
    else:
        completed = 0
        correct_so_far = sum(
            1 for s in samples
            if s.get("llm_judges", {}).get(judge_model_name, 0) > 0
        )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _judge_single, idx, q, gt, pred, server
                ): idx
                for idx, q, gt, pred in work_items
            }

            for future in as_completed(futures):
                idx, score, status = future.result()
                if status == "timeout":
                    samples[idx]["llm_judge_status"] = "timeout"
                    score = -1.0
                samples[idx].setdefault("llm_judges", {})[
                    judge_model_name
                ] = score
                samples[idx]["llm_judge"] = score
                samples[idx]["llm_judge_model"] = judge_model_name
                completed += 1
                if score > 0:
                    correct_so_far += 1

                if completed % 100 == 0 or completed == len(work_items):
                    total_done = already_judged + completed
                    timeouts = sum(1 for s in samples if s.get("llm_judge_status") == "timeout")
                    print(
                        f"  [{task_name}] {total_done}/{total} "
                        f"(acc: {correct_so_far}/{total_done} = "
                        f"{correct_so_far / total_done * 100:.1f}%"
                        f"{f', timeouts={timeouts}' if timeouts else ''})"
                    )

    # Ensure every sample's llm_judge/llm_judge_model points to current judge
    for sample in samples:
        judge_score = sample.get("llm_judges", {}).get(judge_model_name)
        if judge_score is not None:
            sample["llm_judge"] = judge_score
            sample["llm_judge_model"] = judge_model_name
            sample["llm_judge_prompt_version"] = "medevalkit_0/1"

    # Write updated samples
    with open(filepath, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    scores = [
        s.get("llm_judges", {}).get(judge_model_name, 0.0)
        for s in samples
    ]
    correct = sum(1 for sc in scores if sc > 0)
    mean_score = sum(scores) / len(scores) if scores else 0.0
    stderr = (
        float(np.std(scores, ddof=1) / np.sqrt(len(scores)))
        if len(scores) > 1
        else 0.0
    )

    return total, correct, mean_score, stderr


def update_results_json(
    results_dir: str,
    judge_results: dict[str, tuple[float, float, int]],
    judge_model_name: str,
) -> None:
    """Update the results JSON file with judge scores and group accuracy.

    Args:
        judge_results: {task_name: (mean_score, stderr, n_samples)}
        judge_model_name: Name of the judge model used.
    """
    results_path = Path(results_dir)
    json_files = sorted(results_path.glob("*_results.json"))
    if not json_files:
        print("Warning: No results JSON found.")
        return

    results_file = json_files[-1]
    with open(results_file) as f:
        results = json.load(f)

    # Update judge scores with stderr
    for task_name, (score, stderr, n) in judge_results.items():
        if task_name in results.get("results", {}):
            task_dict = results["results"][task_name]

            # Auto-migrate old llm_judges if missing
            if "llm_judge,none" in task_dict and "llm_judges" not in task_dict:
                old_model = task_dict.get("llm_judge_model", "unknown")
                old_score = task_dict["llm_judge,none"]
                task_dict["llm_judges"] = {old_model: old_score}

            # Set backward-compat fields
            task_dict["llm_judge,none"] = score
            task_dict["llm_judge_stderr,none"] = stderr
            task_dict["llm_judge_stderr_clt,none"] = stderr
            task_dict["llm_judge_stderr_clustered,none"] = "N/A"
            task_dict["llm_judge_model"] = judge_model_name

            # Set per-judge entry (flat, no nested stderr)
            task_dict.setdefault("llm_judges", {})[judge_model_name] = score

    # Compute group-level accuracy (open llm_judge + closed accuracy)
    groups = {
        "vqa_rad": ("vqa_rad_open", "vqa_rad_closed"),
        "slake": ("slake_open", "slake_closed"),
        "path_vqa": ("path_vqa_open", "path_vqa_closed"),
        "path_vqa_mini": ("path_vqa_mini_open", "path_vqa_mini_closed"),
    }
    n_samples = results.get("n-samples", {})

    for group, (open_task, closed_task) in groups.items():
        if open_task not in judge_results:
            continue
        if closed_task not in results.get("results", {}):
            continue

        n_open = n_samples.get(open_task, {}).get("effective", 0)
        n_closed = n_samples.get(closed_task, {}).get("effective", 0)
        if n_open == 0 or n_closed == 0:
            continue

        n_total = n_open + n_closed
        acc_open = judge_results[open_task][0]  # llm_judge mean
        se_open = judge_results[open_task][1]  # llm_judge stderr
        acc_closed = results["results"][closed_task].get(
            "accuracy,none", 0
        )
        se_closed = results["results"][closed_task].get(
            "accuracy_stderr,none", 0
        )

        # Weighted average
        overall_acc = (acc_open * n_open + acc_closed * n_closed) / n_total

        # Pooled stderr
        if isinstance(se_closed, (int, float)) and se_closed > 0:
            pooled_var = (
                (n_open - 1) * se_open**2 * n_open
                + (n_closed - 1) * se_closed**2 * n_closed
            ) / (n_total - 2)
            overall_se = math.sqrt(pooled_var / n_total)
        else:
            overall_se = "N/A"

        results["results"][group] = {
            "alias": f" - {group}",
            "accuracy,none": overall_acc,
            "accuracy_stderr,none": overall_se,
            "llm_judge_model": judge_model_name,
        }
        print(
            f"  Group {group}: "
            f"open({n_open})={acc_open:.1f} + "
            f"closed({n_closed})={acc_closed:.1f} "
            f"-> overall({n_total})={overall_acc:.1f}"
            f"±{overall_se:.2f}"
            if isinstance(overall_se, float)
            else f"  Group {group}: overall={overall_acc:.1f}"
        )

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Updated {results_file}")


def main():
    parser = argparse.ArgumentParser(description="Post-hoc LLM judge scoring")
    parser.add_argument(
        "--results_dir",
        required=True,
        help="Path to model results directory containing JSONL samples",
    )
    parser.add_argument(
        "--run_judge",
        action="store_true",
        help="Run LLM judge on open-ended VQA tasks",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel workers for LLM judge API calls",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Dry run: don't actually call LLM, just show what would be "
        "processed",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-evaluation even if samples already have llm_judge "
        "scores. Use with caution: this overwrites previous scores.",
    )
    # Judge model configuration (override env vars)
    parser.add_argument(
        "--judge_api_type",
        type=str,
        default=None,
        help="Judge API type: openai, anthropic, etc. "
        "(default: $API_TYPE or 'openai')",
    )
    parser.add_argument(
        "--judge_api_url",
        type=str,
        default=None,
        help="Judge API base URL, e.g. http://localhost:8000/v1 for vLLM "
        "(default: $OPENAI_API_URL)",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default=None,
        help="Judge model name, e.g. Qwen/Qwen3-VL-32B "
        "(default: $JUDGE_MODEL)",
    )
    # Auto-serve: automatically start/stop a local vLLM judge server
    parser.add_argument(
        "--auto_serve",
        action="store_true",
        help="Auto start a local vLLM judge server, run judge, then stop. "
        "No need to manually manage the server.",
    )
    parser.add_argument(
        "--serve_gpu_ids",
        type=str,
        default="0,1,2,3",
        help="GPU IDs for auto-serve (comma-separated, default: 0,1,2,3)",
    )
    parser.add_argument(
        "--serve_tp",
        type=int,
        default=None,
        help="Tensor parallel size for auto-serve (default: number of GPUs)",
    )
    parser.add_argument(
        "--serve_port",
        type=int,
        default=8000,
        help="Port for auto-serve (default: 8000)",
    )
    parser.add_argument(
        "--serve_timeout",
        type=int,
        default=600,
        help="Max seconds to wait for server startup (default: 600)",
    )
    args = parser.parse_args()

    sample_files = find_sample_files(args.results_dir)
    print(f"Found {len(sample_files)} sample files in {args.results_dir}")
    for name, path in sample_files.items():
        print(f"  {name}: {path.name}")

    judge_results: dict[str, tuple[float, float, int]] = {}
    judge_model_name = ""

    if args.run_judge:
        # Determine if we need auto-serve
        serve_ctx = None
        if args.auto_serve:
            from lmms_eval.llm_judge.launcher import get_launcher

            judge_model_name = args.judge_model or os.getenv("JUDGE_MODEL", "Qwen/Qwen3-VL-32B-Instruct")
            tp = args.serve_tp or len(args.serve_gpu_ids.split(","))

            VLLMLauncher = get_launcher("vllm")
            serve_ctx = VLLMLauncher(
                model=judge_model_name,
                port=args.serve_port,
                tp=tp,
                gpu_ids=args.serve_gpu_ids,
                timeout=args.serve_timeout,
            )
            # Override API settings to point to local server
            args.judge_api_type = "openai"
            args.judge_api_url = f"http://localhost:{args.serve_port}/v1"
            args.judge_model = judge_model_name

        def _run_judge():
            nonlocal judge_model_name, judge_results
            server, judge_model_name = create_judge_server(args)
            print(
                f"\n=== LLM Judge Scoring ==="
                f"\n  Model:   {judge_model_name}"
                f"\n  Workers: {args.workers}"
                f"\n  Force:   {args.force}"
            )

            for task_name in JUDGE_TASKS:
                if task_name not in sample_files:
                    print(f"  Skipping {task_name}: no sample file found")
                    continue

                print(f"\nProcessing {task_name}...")
                total, correct, mean_score, stderr = run_judge_on_task(
                    sample_files[task_name],
                    task_name,
                    server=server,
                    judge_model_name=judge_model_name,
                    workers=args.workers,
                    dry_run=args.dry_run,
                    force=args.force,
                )
                judge_results[task_name] = (mean_score, stderr, total)
                print(
                    f"  Result: {correct}/{total} correct, "
                    f"mean llm_judge = {mean_score:.1f} ± {stderr:.2f}"
                )

        if serve_ctx is not None:
            with serve_ctx:
                _run_judge()
        else:
            _run_judge()

    if judge_results:
        update_results_json(
            args.results_dir, judge_results, judge_model_name
        )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if judge_results:
        print(f"\nJudge model: {judge_model_name}")
        print("LLM Judge Scores:")
        for task, (score, stderr, n) in sorted(judge_results.items()):
            print(f"  {task:25s}: {score:.1f} ± {stderr:.2f}  (n={n})")


if __name__ == "__main__":
    main()
