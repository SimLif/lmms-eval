"""Build mini evaluation datasets for medical VQA benchmarks.

Constructs two HuggingFace datasets via stratified / random sampling:

1. simwit/omni-med-vqa-mini-v2
   - Stratified sampling from simwit/omni-med-vqa (test, ~89K rows)
   - ~2000 samples with min 5 per sub-dataset

2. simwit/medmoe-path-vqa-mini
   - Random 750-sample subsets of test_open and test_closed
   - From simwit/medmoe-path-vqa

Usage:
    python scripts/build_med_eval_mini_datasets.py
    python scripts/build_med_eval_mini_datasets.py --dry-run
    python scripts/build_med_eval_mini_datasets.py --dataset omni
    python scripts/build_med_eval_mini_datasets.py --dataset pathvqa
"""

from __future__ import annotations

import argparse
import math
import os
from collections import Counter
from typing import Literal

from datasets import Dataset, load_dataset
from huggingface_hub import login

# ── Constants ────────────────────────────────────────────────────────

HF_TOKEN = os.getenv("HF_TOKEN", "")
SEED = 42

OMNI_SRC_REPO = "simwit/omni-med-vqa"
OMNI_DST_REPO = "simwit/omni-med-vqa-mini-v2"
OMNI_TARGET_TOTAL = 2000
OMNI_MIN_PER_GROUP = 5
OMNI_MAX_TOTAL = 2200

PATHVQA_SRC_REPO = "simwit/medmoe-path-vqa"
PATHVQA_DST_REPO = "simwit/medmoe-path-vqa-mini"
PATHVQA_SAMPLE_SIZE = 750
PATHVQA_SPLITS = ("test_open", "test_closed")


# ── Helpers ──────────────────────────────────────────────────────────


def _compute_quota(
    group_size: int,
    total_size: int,
    target: int = OMNI_TARGET_TOTAL,
    minimum: int = OMNI_MIN_PER_GROUP,
) -> int:
    """Return the number of samples to draw from a group.

    Formula: max(minimum, round(group_size / total_size * target))
    Capped at the actual group size.
    """
    proportional = round(group_size / total_size * target)
    return min(group_size, max(minimum, proportional))


def _print_distribution(
    name_counts: dict[str, int],
    total: int,
) -> None:
    """Pretty-print a sampling distribution table."""
    print(f"\n{'Sub-dataset':<45} {'Count':>6}")
    print("-" * 53)
    for name, count in sorted(
        name_counts.items(), key=lambda x: -x[1]
    ):
        print(f"  {name:<43} {count:>6}")
    print("-" * 53)
    print(f"  {'TOTAL':<43} {total:>6}\n")


# ── omni-med-vqa-mini-v2 ────────────────────────────────────────────


def build_omni_med_vqa_mini(dry_run: bool = False) -> None:
    """Build simwit/omni-med-vqa-mini-v2 via stratified sampling."""
    print("=== Building omni-med-vqa-mini-v2 ===")

    print("Loading source dataset (this may take a while)...")
    ds = load_dataset(OMNI_SRC_REPO, split="test")
    total_n = len(ds)
    print(f"Source size: {total_n}")

    # Group indices by the `dataset` column
    group_indices: dict[str, list[int]] = {}
    dataset_col: list[str] = ds["dataset"]
    for idx, label in enumerate(dataset_col):
        group_indices.setdefault(label, []).append(idx)

    num_groups = len(group_indices)
    print(f"Found {num_groups} sub-datasets")

    # Compute per-group quotas
    quotas: dict[str, int] = {
        name: _compute_quota(len(indices), total_n)
        for name, indices in group_indices.items()
    }
    planned_total = sum(quotas.values())
    print(f"Planned total: {planned_total}")

    _print_distribution(quotas, planned_total)

    assert planned_total <= OMNI_MAX_TOTAL, (
        f"Total {planned_total} exceeds cap {OMNI_MAX_TOTAL}"
    )

    if dry_run:
        print("[dry-run] Skipping upload.\n")
        return

    # Sample from each group and concatenate
    from datasets import concatenate_datasets

    sampled_parts: list[Dataset] = []
    for name in sorted(group_indices):
        indices = group_indices[name]
        subset = ds.select(indices)
        n_select = quotas[name]
        sampled = subset.shuffle(seed=SEED).select(range(n_select))
        sampled_parts.append(sampled)

    mini_ds = concatenate_datasets(sampled_parts)
    mini_ds = mini_ds.shuffle(seed=SEED)
    print(f"Final dataset size: {len(mini_ds)}")

    # Verify distribution
    final_counts = Counter(mini_ds["dataset"])
    for name, expected in quotas.items():
        actual = final_counts.get(name, 0)
        assert actual == expected, (
            f"Mismatch for {name}: expected {expected}, got {actual}"
        )

    login(token=HF_TOKEN)
    print(f"Uploading to {OMNI_DST_REPO} (split=test)...")
    mini_ds.push_to_hub(
        OMNI_DST_REPO, split="test", token=HF_TOKEN
    )
    print(f"Uploaded {len(mini_ds)} rows to {OMNI_DST_REPO}\n")


# ── path-vqa-mini ────────────────────────────────────────────────────


def build_pathvqa_mini(dry_run: bool = False) -> None:
    """Build simwit/medmoe-path-vqa-mini by down-sampling."""
    print("=== Building medmoe-path-vqa-mini ===")

    login_done = False

    for split_name in PATHVQA_SPLITS:
        print(f"\nLoading {PATHVQA_SRC_REPO} split={split_name}...")
        ds = load_dataset(PATHVQA_SRC_REPO, split=split_name)
        src_size = len(ds)
        print(f"  Source size: {src_size}")

        n_select = min(PATHVQA_SAMPLE_SIZE, src_size)
        print(f"  Sampling {n_select} rows (seed={SEED})")

        if dry_run:
            print(f"  [dry-run] Would upload {n_select} rows "
                  f"to {PATHVQA_DST_REPO} split={split_name}")
            continue

        sampled = ds.shuffle(seed=SEED).select(range(n_select))
        print(f"  Sampled size: {len(sampled)}")

        if not login_done:
            login(token=HF_TOKEN)
            login_done = True

        print(f"  Uploading to {PATHVQA_DST_REPO} "
              f"split={split_name}...")
        sampled.push_to_hub(
            PATHVQA_DST_REPO,
            split=split_name,
            token=HF_TOKEN,
        )
        print(f"  Uploaded {len(sampled)} rows.\n")

    if dry_run:
        print("[dry-run] Skipping upload.\n")
    else:
        print(f"Done: {PATHVQA_DST_REPO}\n")


# ── CLI ──────────────────────────────────────────────────────────────

DatasetChoice = Literal["omni", "pathvqa", "all"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Build mini medical-VQA evaluation datasets "
            "and upload to HuggingFace."
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=["omni", "pathvqa", "all"],
        default="all",
        help="Which dataset to build (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print sampling statistics; do not upload.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()

    builders: dict[str, callable] = {
        "omni": build_omni_med_vqa_mini,
        "pathvqa": build_pathvqa_mini,
    }

    targets = (
        list(builders.keys())
        if args.dataset == "all"
        else [args.dataset]
    )

    for name in targets:
        builders[name](dry_run=args.dry_run)

    print("All done.")


if __name__ == "__main__":
    main()
