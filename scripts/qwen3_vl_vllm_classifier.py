#!/usr/bin/env python3
"""
Medical Image Modality Classifier using Qwen3-VL via vLLM.

Uses vLLM for faster and more stable inference.
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
from tqdm import tqdm
from vllm import LLM, SamplingParams

# 10-class modality options
MODALITY_OPTIONS = [
    "CT scan",
    "MRI scan",
    "X-ray",
    "fundus photograph",
    "OCT scan",
    "dermoscopy",
    "histopathology",
    "ultrasound",
    "endoscopy",
    "medical diagram",
]

OPTION_TO_MODALITY = {
    "CT scan": "ct",
    "MRI scan": "mri",
    "X-ray": "xray",
    "fundus photograph": "fundus",
    "OCT scan": "oct",
    "dermoscopy": "dermoscopy",
    "histopathology": "histopathology",
    "ultrasound": "ultrasound",
    "endoscopy": "endoscopy",
    "medical diagram": "other",
}

# Reverse mapping for parsing
LETTER_TO_OPTION = {chr(65+i): opt for i, opt in enumerate(MODALITY_OPTIONS)}


class Qwen3VLClassifier:
    """Qwen3-VL classifier using vLLM backend."""

    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-VL-4B-Instruct",
        tensor_parallel_size: int = 1,
        max_pixels: int = 1024 * 28 * 28,
    ):
        print(f"Loading Qwen3-VL model via vLLM: {model_path}")
        print(f"  tensor_parallel_size={tensor_parallel_size}, max_pixels={max_pixels}")

        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.85,
            trust_remote_code=True,
            disable_log_stats=True,
            limit_mm_per_prompt={"image": 1},
        )
        self.max_pixels = max_pixels

        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=16,
        )

        print("Model loaded successfully.")

    def classify_batch(self, image_paths: list[str]) -> list[tuple[str, str]]:
        """
        Classify a batch of images.

        Returns:
            List of (modality, raw_response) tuples
        """
        # Build prompts
        options_text = "\n".join(
            [f"{chr(65+i)}. {opt}" for i, opt in enumerate(MODALITY_OPTIONS)]
        )
        prompt_text = f"""Look at this medical image and identify its type.

Options:
{options_text}

Answer with just the letter (A-J):"""

        # Build messages for each image
        prompts = []
        valid_indices = []

        for i, img_path in enumerate(image_paths):
            try:
                # vLLM expects image path or PIL Image
                img = Image.open(img_path).convert("RGB")

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ]
                prompts.append(messages)
                valid_indices.append(i)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")

        if not prompts:
            return [(None, "")] * len(image_paths)

        # Run inference
        outputs = self.llm.chat(prompts, self.sampling_params, use_tqdm=False)

        # Parse results
        results = [(None, "")] * len(image_paths)
        for j, output in enumerate(outputs):
            orig_idx = valid_indices[j]
            response = output.outputs[0].text.strip()

            # Parse the letter from response
            modality = None
            for char in response.upper():
                if char in LETTER_TO_OPTION:
                    option = LETTER_TO_OPTION[char]
                    modality = OPTION_TO_MODALITY[option]
                    break

            results[orig_idx] = (modality, response)

        return results


# Test sets
TEST_SETS = {
    "mimic-cxr-dataset-cleaned": "xray",
    "biomedica-histopathology": "histopathology",
    "biomedica-dermatology": "dermoscopy",
    "path-vqa": "histopathology",
}

GMAI_TEST_SETS = {
    "gmai-reasoning10k/2d/cls/ct": "ct",
    "gmai-reasoning10k/2d/cls/ct_2d": "ct",
    "gmai-reasoning10k/2d/cls/dermoscopy": "dermoscopy",
    "gmai-reasoning10k/2d/cls/endoscopy": "endoscopy",
    "gmai-reasoning10k/2d/cls/fundus_photography": "fundus",
    "gmai-reasoning10k/2d/cls/histopathology": "histopathology",
    "gmai-reasoning10k/2d/cls/pathology": "histopathology",
    "gmai-reasoning10k/2d/cls/oct": "oct",
    "gmai-reasoning10k/2d/cls/ultrasound": "ultrasound",
    "gmai-reasoning10k/2d/cls/x_ray": "xray",
    "gmai-reasoning10k/2d/cls/bone_radiograph": "xray",
    "gmai-reasoning10k/2d/cls/mr_unknown": "mri",
}


def get_test_images(image_root: Path, dataset: str, n_samples: int = 100):
    """Get random sample of images from a dataset."""
    dataset_dir = image_root / dataset
    if not dataset_dir.exists():
        return []

    extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp"]
    all_images = []
    for ext in extensions:
        all_images.extend(dataset_dir.rglob(f"*{ext}"))
        all_images.extend(dataset_dir.rglob(f"*{ext.upper()}"))

    if not all_images:
        return []

    if len(all_images) > n_samples:
        all_images = random.sample(all_images, n_samples)

    return [str(p) for p in all_images]


def evaluate(classifier, image_root: str, samples_per_dataset: int = 50, batch_size: int = 16):
    """Run evaluation on test sets."""
    image_root = Path(image_root)
    all_results = []
    per_dataset_results = defaultdict(list)

    all_test_sets = {**TEST_SETS, **GMAI_TEST_SETS}

    for dataset, expected_modality in all_test_sets.items():
        print(f"\nEvaluating {dataset} (expected: {expected_modality})...")
        images = get_test_images(image_root, dataset, samples_per_dataset)

        if not images:
            print(f"  No images found, skipping")
            continue

        print(f"  Found {len(images)} test images")

        # Batch processing
        for i in tqdm(range(0, len(images), batch_size), desc=f"  {dataset}"):
            batch_paths = images[i:i+batch_size]
            results = classifier.classify_batch(batch_paths)

            for (modality, _) in results:
                if modality is not None:
                    all_results.append((expected_modality, modality))
                    per_dataset_results[dataset].append((expected_modality, modality))

    # Compute metrics
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS (Qwen3-VL via vLLM)")
    print("=" * 70)

    print("\nPer-dataset accuracy:")
    for dataset, results in per_dataset_results.items():
        correct = sum(1 for gt, pred in results if gt == pred)
        total = len(results)
        acc = correct / total * 100 if total > 0 else 0
        print(f"  {dataset}: {correct}/{total} = {acc:.1f}%")

    correct = sum(1 for gt, pred in all_results if gt == pred)
    total = len(all_results)
    overall_acc = correct / total * 100 if total > 0 else 0
    print(f"\nOverall accuracy: {correct}/{total} = {overall_acc:.1f}%")

    # Error analysis
    print("\nMisclassification patterns:")
    errors = [(gt, pred) for gt, pred in all_results if gt != pred]
    error_counts = Counter(errors)
    for (gt, pred), count in error_counts.most_common(10):
        print(f"  {gt} -> {pred}: {count}")

    return overall_acc


def main():
    parser = argparse.ArgumentParser(description="Classify medical images using Qwen3-VL via vLLM")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-VL-4B-Instruct",
        help="Model path",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default="/root/paddlejob/workspace/env_run/hqguo/data/afs_upload/images",
        help="Root directory for images",
    )
    parser.add_argument("--samples", type=int, default=50, help="Samples per dataset")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--max-pixels", type=int, default=1024*28*28, help="Max pixels")
    parser.add_argument("--test-image", type=str, help="Test on a single image")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    random.seed(args.seed)

    classifier = Qwen3VLClassifier(
        model_path=args.model,
        tensor_parallel_size=args.tp,
        max_pixels=args.max_pixels,
    )

    if args.test_image:
        results = classifier.classify_batch([args.test_image])
        modality, response = results[0]
        print(f"\nImage: {args.test_image}")
        print(f"Modality: {modality}")
        print(f"Raw response: {response}")
    else:
        evaluate(classifier, args.image_root, args.samples, args.batch_size)


if __name__ == "__main__":
    main()
