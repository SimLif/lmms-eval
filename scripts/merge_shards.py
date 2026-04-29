"""Merge sharded lmms_eval outputs into a single result directory.

Task-level sharding: each shard runs disjoint tasks. This script:
1. Concatenates per-task sample JSONLs across shards (each task in exactly one shard)
2. Collects metrics from each shard's results.json into a unified results.json
3. Output is compatible with pipeline.py judge + parse

Usage:
    python scripts/merge_shards.py --shard-dirs dir0 dir1 ... --output-dir merged/
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def find_sample_files(shard_dir: str) -> dict[str, Path]:
    """Find *_samples_*.jsonl files in a shard directory."""
    files = {}
    for p in Path(shard_dir).rglob("*_samples_*.jsonl"):
        m = re.search(r"_samples_(.+)\.jsonl$", p.name)
        if m:
            files[m.group(1)] = p
    return files


def find_results_json(shard_dir: str) -> Path | None:
    """Find *_results.json in a shard directory."""
    for p in Path(shard_dir).rglob("*_results.json"):
        return p
    return None


def copy_jsonl(src: Path, dst: Path) -> int:
    """Copy JSONL file, return sample count."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            if line.strip():
                fout.write(line)
                n += 1
    return n


def merge_results(paths: list[Path]) -> dict:
    """Merge results.json from disjoint-task shards (union, not average)."""
    merged_results = {}
    merged_versions = {}
    for p in paths:
        data = json.loads(p.read_text())
        for task, metrics in data.get("results", {}).items():
            merged_results[task] = metrics
        for task, ver in data.get("versions", {}).items():
            merged_versions[task] = ver
    out = {"results": merged_results}
    if merged_versions:
        out["versions"] = merged_versions
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_files: dict[str, list[Path]] = defaultdict(list)
    results_files: list[Path] = []

    for shard_dir in args.shard_dirs:
        for task, path in find_sample_files(shard_dir).items():
            task_files[task].append(path)
        rj = find_results_json(shard_dir)
        if rj:
            results_files.append(rj)

    # Get timestamp prefix from first shard
    ts_prefix = ""
    if results_files:
        m = re.match(r"(\d{8}_\d{6})_", results_files[0].name)
        if m:
            ts_prefix = m.group(1) + "_"

    print(f"Merging {len(task_files)} tasks from {len(args.shard_dirs)} shards:")
    for task, paths in sorted(task_files.items()):
        out_path = output_dir / f"{ts_prefix}samples_{task}.jsonl"
        n = copy_jsonl(paths[0], out_path)
        if len(paths) > 1:
            print(f"  WARNING: {task} found in {len(paths)} shards, using first only")
        print(f"  {task}: {n} samples")

    if results_files:
        merged = merge_results(results_files)
        results_out = output_dir / f"{ts_prefix}results.json"
        results_out.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
        print(f"  results.json: {len(merged['results'])} tasks from {len(results_files)} shards")

    print(f"\nOutput: {output_dir}")


if __name__ == "__main__":
    main()
