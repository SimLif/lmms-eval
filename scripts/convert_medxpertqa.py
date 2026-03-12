"""
Convert TsinghuaC3I/MedXpertQA to HuggingFace format with embedded images.
Upload to simwit/medxpertqa-mm and simwit/medxpertqa-text
"""

import os
import zipfile

from datasets import Dataset, Features, Image, Value, load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image as PILImage
from tqdm import tqdm

# Cache directory
HF_HOME = os.getenv("HF_HOME", "~/.cache/huggingface/")
CACHE_DIR = os.path.join(os.path.expanduser(HF_HOME), "medxpertqa_convert")


def download_and_extract_images():
    """Download and extract images.zip."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    images_dir = os.path.join(CACHE_DIR, "images")

    if os.path.exists(images_dir) and len(os.listdir(images_dir)) > 0:
        print(f"Images already extracted at {images_dir}")
        return images_dir

    print("Downloading images.zip...")
    zip_path = hf_hub_download(
        repo_id="TsinghuaC3I/MedXpertQA",
        filename="images.zip",
        repo_type="dataset",
        token=True,
    )

    print("Extracting images...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(CACHE_DIR)

    return images_dir


def convert_mm_dataset():
    """Convert MM (multimodal) dataset."""
    print("\n=== Converting MM dataset ===")

    # Download images
    images_dir = download_and_extract_images()

    # Load original dataset
    print("Loading original dataset...")
    ds = load_dataset("TsinghuaC3I/MedXpertQA", "MM", token=True)

    def process_split(split_name):
        split_ds = ds[split_name]
        records = []

        for item in tqdm(split_ds, desc=f"Processing {split_name}"):
            # Load images
            image_names = item.get("images", [])
            images = []
            for img_name in image_names:
                img_path = os.path.join(images_dir, img_name)
                if os.path.exists(img_path):
                    try:
                        img = PILImage.open(img_path).convert("RGB")
                        images.append(img)
                    except Exception as e:
                        print(f"Failed to load {img_path}: {e}")

            # Format options as string
            options = item.get("options", {})
            options_list = []
            for key in ["A", "B", "C", "D", "E"]:
                if key in options and options[key]:
                    options_list.append(f"{key}. {options[key]}")
            options_str = "\n".join(options_list)

            record = {
                "id": item["id"],
                "question": item["question"],
                "options": options_str,
                "answer": item["label"],
                "image": images[0] if images else None,  # Primary image
                "images": images,  # All images
                "num_images": len(images),
                "medical_task": item.get("medical_task", ""),
                "body_system": item.get("body_system", ""),
                "question_type": item.get("question_type", ""),
            }
            records.append(record)

        return records

    # Process each split
    processed = {}
    for split in ds.keys():
        processed[split] = process_split(split)

    # Create HuggingFace dataset
    print("\nCreating HuggingFace dataset...")
    hf_datasets = {}
    for split, records in processed.items():
        # Filter out records without images
        records_with_images = [r for r in records if r["image"] is not None]
        print(f"{split}: {len(records_with_images)}/{len(records)} samples have images")

        hf_datasets[split] = Dataset.from_list(records_with_images)

    return hf_datasets


def convert_text_dataset():
    """Convert Text (text-only) dataset."""
    print("\n=== Converting Text dataset ===")

    # Load original dataset
    print("Loading original dataset...")
    ds = load_dataset("TsinghuaC3I/MedXpertQA", "Text", token=True)

    def process_split(split_name):
        split_ds = ds[split_name]
        records = []

        for item in tqdm(split_ds, desc=f"Processing {split_name}"):
            # Format options as string
            options = item.get("options", {})
            options_list = []
            for key in ["A", "B", "C", "D", "E"]:
                if key in options and options[key]:
                    options_list.append(f"{key}. {options[key]}")
            options_str = "\n".join(options_list)

            record = {
                "id": item["id"],
                "question": item["question"],
                "options": options_str,
                "answer": item["label"],
                "medical_task": item.get("medical_task", ""),
                "body_system": item.get("body_system", ""),
                "question_type": item.get("question_type", ""),
            }
            records.append(record)

        return records

    # Process each split
    processed = {}
    for split in ds.keys():
        processed[split] = process_split(split)

    # Create HuggingFace dataset
    print("\nCreating HuggingFace dataset...")
    hf_datasets = {}
    for split, records in processed.items():
        hf_datasets[split] = Dataset.from_list(records)
        print(f"{split}: {len(records)} samples")

    return hf_datasets


def main():
    import sys
    auto_upload = "--upload" in sys.argv

    print("Converting MedXpertQA dataset to HuggingFace format...")

    # Convert MM dataset
    mm_datasets = convert_mm_dataset()

    # Convert Text dataset
    text_datasets = convert_text_dataset()

    # Preview
    print("\n=== Preview MM dataset ===")
    if "test" in mm_datasets:
        sample = mm_datasets["test"][0]
        print(f"Keys: {list(sample.keys())}")
        print(f"Question: {sample['question'][:100]}...")
        print(f"Image type: {type(sample['image'])}")

    print("\n=== Preview Text dataset ===")
    if "test" in text_datasets:
        sample = text_datasets["test"][0]
        print(f"Keys: {list(sample.keys())}")
        print(f"Question: {sample['question'][:100]}...")

    # Ask for upload confirmation
    print("\n" + "=" * 50)
    print("Ready to upload to HuggingFace Hub:")
    print("  - simwit/medxpertqa-mm")
    print("  - simwit/medxpertqa-text")

    if auto_upload:
        response = "y"
        print("Auto-upload enabled via --upload flag")
    else:
        response = input("Upload? [y/N]: ")

    if response.lower() == "y":
        print("\nUploading MM dataset...")
        for split, dataset in mm_datasets.items():
            dataset.push_to_hub("simwit/medxpertqa-mm", split=split, token=True)
        print("MM dataset uploaded!")

        print("\nUploading Text dataset...")
        for split, dataset in text_datasets.items():
            dataset.push_to_hub("simwit/medxpertqa-text", split=split, token=True)
        print("Text dataset uploaded!")
    else:
        print("Upload cancelled.")

        # Save locally instead
        local_path = os.path.join(CACHE_DIR, "converted")
        os.makedirs(local_path, exist_ok=True)

        print(f"\nSaving locally to {local_path}...")
        for split, dataset in mm_datasets.items():
            dataset.save_to_disk(os.path.join(local_path, f"mm_{split}"))
        for split, dataset in text_datasets.items():
            dataset.save_to_disk(os.path.join(local_path, f"text_{split}"))
        print("Saved locally!")


if __name__ == "__main__":
    main()
