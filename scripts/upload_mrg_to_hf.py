"""
MRG (Medical Report Generation) Dataset Upload Script

Uploads local IU-XRAY and MIMIC-CXR datasets to HuggingFace.

Usage:
    python scripts/upload_mrg_to_hf.py --dataset iu_xray
    python scripts/upload_mrg_to_hf.py --dataset mimic_cxr
    python scripts/upload_mrg_to_hf.py --dataset all
"""
import os
import json
import argparse
from pathlib import Path
from PIL import Image
from datasets import Dataset, Features, Value, Image as HFImage
from huggingface_hub import login
from tqdm import tqdm

# Configuration
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_ORG = "simwit"
LOCAL_DATA_ROOT = "/root/paddlejob/workspace/env_run/hqguo/eval-kit/datas"

DATASETS = {
    "iu_xray": {
        "local_dir": "IU_XRAY",
        "hf_repo": "iu-xray-lmms",
        "image_is_list": True,
    },
    "mimic_cxr": {
        "local_dir": "MIMIC_CXR",
        "hf_repo": "mimic-cxr-lmms",
        "image_is_list": False,
    }
}


def load_local_data(dataset_name: str) -> list:
    """Load local dataset and prepare for HF upload."""
    config = DATASETS[dataset_name]
    local_dir = Path(LOCAL_DATA_ROOT) / config["local_dir"]
    images_dir = local_dir / "images"
    test_json = local_dir / "test.json"

    with open(test_json) as f:
        data = json.load(f)

    samples = []
    for item in tqdm(data, desc=f"Processing {dataset_name}"):
        # Handle image field (list for IU-XRAY, string for MIMIC-CXR)
        if config["image_is_list"]:
            img_name = item["image"][0] if isinstance(item["image"], list) else item["image"]
        else:
            img_name = item["image"]

        img_path = images_dir / img_name
        if not img_path.exists():
            print(f"Warning: Image not found: {img_path}")
            continue

        # Build report from findings + impression
        findings = item.get("findings", "").strip()
        impression = item.get("impression", "").strip()
        report = f"FINDINGS: {findings}\n\nIMPRESSION: {impression}"

        samples.append({
            "image": str(img_path),
            "findings": findings,
            "impression": impression,
            "report": report,
        })

    return samples


def upload_to_hf(dataset_name: str):
    """Upload dataset to HuggingFace."""
    config = DATASETS[dataset_name]

    # Login to HuggingFace
    login(token=HF_TOKEN)

    # Load data
    samples = load_local_data(dataset_name)
    print(f"Loaded {len(samples)} samples for {dataset_name}")

    # Define features
    features = Features({
        "image": HFImage(),
        "findings": Value("string"),
        "impression": Value("string"),
        "report": Value("string"),
    })

    # Create HF dataset with generator
    def gen():
        for s in tqdm(samples, desc="Creating dataset"):
            try:
                img = Image.open(s["image"]).convert("RGB")
                yield {
                    "image": img,
                    "findings": s["findings"],
                    "impression": s["impression"],
                    "report": s["report"],
                }
            except Exception as e:
                print(f"Error processing {s['image']}: {e}")

    dataset = Dataset.from_generator(gen, features=features)

    # Upload to HuggingFace
    repo_id = f"{HF_ORG}/{config['hf_repo']}"
    print(f"Uploading to {repo_id}...")
    dataset.push_to_hub(repo_id, split="test", token=HF_TOKEN)
    print(f"Successfully uploaded to {repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Upload MRG datasets to HuggingFace")
    parser.add_argument("--dataset", choices=["iu_xray", "mimic_cxr", "all"], required=True,
                        help="Dataset to upload: iu_xray, mimic_cxr, or all")
    args = parser.parse_args()

    if args.dataset == "all":
        for name in DATASETS:
            upload_to_hf(name)
    else:
        upload_to_hf(args.dataset)


if __name__ == "__main__":
    main()
