#!/usr/bin/env python3
"""Download and prepare spatial reasoning datasets for Stage 2 training.

Datasets:
  1. VSR (Visual Spatial Reasoning) - ~10K binary spatial relation samples (uses COCO images)
  2. What'sUp - orientation understanding (~600 samples)
  3. GQA Spatial - spatial subset of GQA (~50K samples, uses existing GQA images)
  4. SpatialBench - proxy via Q-Spatial-Bench or RussRobin/SpatialBench

Usage:
  python scripts/download_spatial_datasets.py --data_root /data/datasets
  python scripts/download_spatial_datasets.py --data_root /data/datasets --datasets vsr gqa_spatial
"""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_json(path: str, data: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  Wrote {len(data)} samples → {path}")


def _write_jsonl(path: str, data: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(data)} samples → {path}")


def _symlink_or_copy(src: str, dst: str) -> None:
    """Create a symlink from dst -> src."""
    if os.path.exists(dst):
        print(f"  Already exists: {dst}")
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.symlink(os.path.abspath(src), dst)
        print(f"  Symlinked {dst} → {src}")
    except OSError:
        print(f"  Copying {src} → {dst}")
        shutil.copytree(src, dst)


# ---------------------------------------------------------------------------
# 1. VSR (Visual Spatial Reasoning)
#    HuggingFace: cambridgeltl/vsr_random
#    Images: COCO train2017 (already at /data/datasets/coco/train2017/)
# ---------------------------------------------------------------------------


def download_vsr(data_root: str) -> None:
    """Download VSR dataset from HuggingFace.

    VSR images come from COCO train2017. The 'image' field in the HF dataset
    is a filename like '000000296471.jpg' which corresponds to a COCO image.
    """
    from datasets import load_dataset

    out_dir = os.path.join(data_root, "vsr")
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(out_dir, exist_ok=True)

    print("\n[1/4] Downloading VSR (Visual Spatial Reasoning)...")

    # VSR uses COCO train2017 images - symlink them
    coco_dir = os.path.join(data_root, "coco", "train2017")
    if os.path.isdir(coco_dir):
        _symlink_or_copy(coco_dir, img_dir)
    else:
        print(f"  WARNING: COCO images not found at {coco_dir}")
        print("  Will attempt to download individual images from image_link field.")
        os.makedirs(img_dir, exist_ok=True)

    ds = load_dataset("cambridgeltl/vsr_random")

    split_map = {"train": "train", "validation": "val", "test": "test"}

    for hf_split, out_split in split_map.items():
        if hf_split not in ds:
            print(f"  Split '{hf_split}' not found, skipping")
            continue

        split_data = ds[hf_split]
        samples = []
        missing_images = []

        for row in split_data:
            img_filename = row["image"]  # e.g. '000000296471.jpg'
            caption = row["caption"]
            label = row["label"]  # 0=False, 1=True

            # Check if image exists
            full_img_path = os.path.join(img_dir, img_filename)
            if not os.path.exists(full_img_path):
                # Try downloading from image_link
                image_link = row.get("image_link", "")
                if image_link and not os.path.exists(full_img_path):
                    missing_images.append((img_filename, image_link))
                continue

            samples.append({
                "image": img_filename,
                "question": f'Is the following statement true or false about the image? "{caption}" Answer with just True or False.',
                "answer": "True" if label == 1 else "False",
                "split": out_split,
            })

        if missing_images:
            print(f"  {len(missing_images)} images missing for {out_split} split, downloading...")
            import urllib.request
            downloaded = 0
            for img_filename, image_link in missing_images:
                try:
                    urllib.request.urlretrieve(image_link, os.path.join(img_dir, img_filename))
                    downloaded += 1
                except Exception:
                    pass
            print(f"  Downloaded {downloaded}/{len(missing_images)} missing images")

            # Re-process to include newly downloaded images
            samples = []
            for row in split_data:
                img_filename = row["image"]
                if not os.path.exists(os.path.join(img_dir, img_filename)):
                    continue
                caption = row["caption"]
                label = row["label"]
                samples.append({
                    "image": img_filename,
                    "question": f'Is the following statement true or false about the image? "{caption}" Answer with just True or False.',
                    "answer": "True" if label == 1 else "False",
                    "split": out_split,
                })

        out_file = os.path.join(out_dir, f"{out_split}.jsonl")
        _write_jsonl(out_file, samples)

    print(f"  VSR complete → {out_dir}")


# ---------------------------------------------------------------------------
# 2. What'sUp
#    HuggingFace: Mayfull/whats_up_vlms or AsphyXIA/whatsup
# ---------------------------------------------------------------------------


def download_whatsup(data_root: str) -> None:
    """Download What'sUp dataset from HuggingFace."""
    from datasets import load_dataset

    out_dir = os.path.join(data_root, "whatsup")
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    print("\n[2/4] Downloading What'sUp...")

    # Try multiple HF sources
    hf_sources = [
        "Mayfull/whats_up_vlms",
        "AsphyXIA/whatsup",
    ]

    ds = None
    for hf_name in hf_sources:
        try:
            print(f"  Trying {hf_name}...")
            ds = load_dataset(hf_name)
            print(f"  Loaded from {hf_name}")
            # Print available splits and columns
            for split in ds:
                print(f"    Split '{split}': {len(ds[split])} samples, columns: {ds[split].column_names}")
            break
        except Exception as e:
            print(f"  Failed: {e}")
            continue

    if ds is None:
        print("  Could not find What'sUp on HuggingFace.")
        print("  Creating empty placeholder - please download manually from https://github.com/amitakamath/whatsup_vlms")
        _write_json(os.path.join(out_dir, "train.json"), [])
        return

    all_samples = []
    for split_name in ds:
        split_data = ds[split_name]
        for i, row in enumerate(split_data):
            img_filename = f"{split_name}_{i:06d}.jpg"
            img_path = os.path.join(img_dir, img_filename)

            # Handle image fields — AsphyXIA/whatsup uses 'images' (list of PIL),
            # other sources may use 'image' (single PIL or path string).
            images_field = row.get("images", None)
            image = row.get("image", None)
            if images_field is not None and isinstance(images_field, list) and len(images_field) > 0:
                pil_img = images_field[0]
                if hasattr(pil_img, "save") and not os.path.exists(img_path):
                    pil_img.save(img_path)
            elif image is not None:
                if hasattr(image, "save"):
                    if not os.path.exists(img_path):
                        image.save(img_path)
                elif isinstance(image, str):
                    img_filename = os.path.basename(image)
            elif "image_path" in row:
                img_filename = os.path.basename(row["image_path"])

            # Extract question and answer.
            # AsphyXIA/whatsup has positive_caption / negative_caption (lists).
            question = row.get("question", row.get("text", ""))
            answer = row.get("answer", row.get("label", row.get("correct_answer", "")))

            if not question:
                pos_caps = row.get("positive_caption", [])
                neg_caps = row.get("negative_caption", [])
                if pos_caps and neg_caps:
                    pos = pos_caps[0] if isinstance(pos_caps, list) else pos_caps
                    neg = neg_caps[0] if isinstance(neg_caps, list) else neg_caps
                    question = (
                        f"Which description best matches the spatial arrangement "
                        f"in the image?\nA: {pos}\nB: {neg}"
                    )
                    answer = "A"
                elif "caption" in row:
                    question = f"Which description best matches the spatial arrangement in the image? {row['caption']}"

            all_samples.append({
                "image": img_filename,
                "question": question,
                "answer": str(answer),
                "split": "train",
            })

    # Split 80/10/10 for train/val/test
    import random
    random.seed(42)
    random.shuffle(all_samples)
    n = len(all_samples)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    for s in all_samples[:n_train]:
        s["split"] = "train"
    for s in all_samples[n_train:n_train + n_val]:
        s["split"] = "val"
    for s in all_samples[n_train + n_val:]:
        s["split"] = "test"

    train_samples = [s for s in all_samples if s["split"] == "train"]
    val_samples = [s for s in all_samples if s["split"] == "val"]
    test_samples = [s for s in all_samples if s["split"] == "test"]

    if train_samples:
        _write_json(os.path.join(out_dir, "train.json"), train_samples)
    if val_samples:
        _write_json(os.path.join(out_dir, "val.json"), val_samples)
    if test_samples:
        _write_json(os.path.join(out_dir, "test.json"), test_samples)

    if not any([train_samples, val_samples, test_samples]) and all_samples:
        for s in all_samples:
            s["split"] = "train"
        _write_json(os.path.join(out_dir, "train.json"), all_samples)

    print(f"  What'sUp complete → {out_dir} ({len(all_samples)} total samples)")


# ---------------------------------------------------------------------------
# 3. GQA Spatial
#    HuggingFace: lmms-lab/GQA (train_balanced_instructions config)
#    Images: existing GQA images at data_root/gqa/images/
# ---------------------------------------------------------------------------

SPATIAL_KEYWORDS = [
    "left", "right", "above", "below", "behind", "front",
    "top", "bottom", "near", "far", "next to", "beside",
    "between", "under", "over", "on top of", "in front of",
    "to the left", "to the right", "on the left", "on the right",
    "above the", "below the", "underneath", "adjacent",
    "closer", "farther", "nearest", "farthest",
]

SPATIAL_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(kw) for kw in SPATIAL_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


def _is_spatial_question(question: str) -> bool:
    return bool(SPATIAL_PATTERN.search(question))


def download_gqa_spatial(data_root: str) -> None:
    """Download GQA spatial subset using existing GQA images and HuggingFace questions."""
    from datasets import load_dataset

    gqa_images = os.path.join(data_root, "gqa", "images")
    out_dir = os.path.join(data_root, "gqa_spatial")
    out_img_dir = os.path.join(out_dir, "images")
    os.makedirs(out_dir, exist_ok=True)

    print("\n[3/4] Preparing GQA Spatial subset...")

    if not os.path.isdir(gqa_images):
        print(f"  ERROR: GQA images not found at {gqa_images}")
        print("  Please download GQA images first.")
        _write_json(os.path.join(out_dir, "train.json"), [])
        return

    # Symlink images
    _symlink_or_copy(gqa_images, out_img_dir)

    # Get set of available images
    print("  Indexing available GQA images...")
    available_images = set(os.listdir(gqa_images))
    print(f"  Found {len(available_images)} images")

    # Load questions from HuggingFace
    hf_configs = {
        "train": "train_balanced_instructions",
        "val": "val_balanced_instructions",
        "test": "testdev_balanced_instructions",
    }

    hf_split_names = {
        "train": "train",
        "val": "val",
        "test": "testdev",
    }

    for split_name, hf_config in hf_configs.items():
        print(f"  Loading GQA {hf_config}...")
        hf_split = hf_split_names[split_name]
        try:
            ds = load_dataset("lmms-lab/GQA", hf_config, split=hf_split)
        except Exception as e:
            print(f"  Failed to load {hf_config} split={hf_split}: {e}")
            continue

        print(f"  Loaded {len(ds)} questions, filtering for spatial...")
        spatial_samples = []
        for row in ds:
            question = row.get("question", "")
            if not _is_spatial_question(question):
                continue

            image_id = str(row.get("imageId", ""))
            img_filename = f"{image_id}.jpg"

            if img_filename not in available_images:
                continue

            answer = str(row.get("answer", ""))
            if not answer:
                continue

            spatial_samples.append({
                "image": img_filename,
                "question": question,
                "answer": answer,
                "split": split_name,
            })

        out_file = os.path.join(out_dir, f"{split_name}.json")
        _write_json(out_file, spatial_samples)

    print(f"  GQA Spatial complete → {out_dir}")


# ---------------------------------------------------------------------------
# 4. SpatialBench
#    HuggingFace: RussRobin/SpatialBench or andrewliao11/Q-Spatial-Bench
# ---------------------------------------------------------------------------


def download_spatialbench(data_root: str) -> None:
    """Download SpatialBench or a suitable proxy dataset."""
    from datasets import load_dataset

    out_dir = os.path.join(data_root, "spatialbench")
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    print("\n[4/4] Downloading SpatialBench...")

    hf_sources = [
        ("RussRobin/SpatialBench", None),
        ("andrewliao11/Q-Spatial-Bench", None),
        ("andrewliao11/Q-Spatial-Bench", "Q-Spatial-ScanNet"),
    ]

    ds = None
    source_name = None
    for hf_name, config in hf_sources:
        try:
            label = f"{hf_name}" + (f" ({config})" if config else "")
            print(f"  Trying {label}...")
            ds = load_dataset(hf_name, config)
            source_name = label
            print(f"  Loaded from {label}")
            for split in ds:
                print(f"    Split '{split}': {len(ds[split])} samples, columns: {ds[split].column_names}")
            break
        except Exception as e:
            print(f"  Failed: {e}")
            continue

    if ds is None:
        print("  Could not find SpatialBench on HuggingFace.")
        print("  Creating empty placeholder.")
        _write_json(os.path.join(out_dir, "train.json"), [])
        return

    all_samples = []
    for split_name in ds:
        split_data = ds[split_name]
        for i, row in enumerate(split_data):
            img_filename = f"{split_name}_{i:06d}.jpg"
            img_path = os.path.join(img_dir, img_filename)

            # Handle image field
            image = row.get("image", None)
            if image is not None and hasattr(image, "save"):
                if not os.path.exists(img_path):
                    # Convert non-RGB modes (e.g. I;16 depth maps) to RGB for JPEG
                    if image.mode not in ("RGB", "L"):
                        image = image.convert("RGB")
                    image.save(img_path)
            elif isinstance(image, str):
                img_filename = os.path.basename(image)
            elif "image_path" in row:
                img_filename = os.path.basename(row["image_path"])

            question = row.get("question", row.get("text", row.get("prompt", "")))
            answer = str(row.get("answer", row.get("label", row.get("ground_truth", ""))))
            category = row.get("category", row.get("type", row.get("task_type", "spatial")))

            all_samples.append({
                "image": img_filename,
                "question": question,
                "answer": answer,
                "category": str(category),
                "split": split_name,
            })

    # If only one split, create train/val/test
    if len(ds) == 1:
        import random
        random.seed(42)
        random.shuffle(all_samples)
        n = len(all_samples)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        for s in all_samples[:n_train]:
            s["split"] = "train"
        for s in all_samples[n_train:n_train + n_val]:
            s["split"] = "val"
        for s in all_samples[n_train + n_val:]:
            s["split"] = "test"

    splits = set(s["split"] for s in all_samples)
    for split_name in splits:
        split_samples = [s for s in all_samples if s["split"] == split_name]
        out_file = os.path.join(out_dir, f"{split_name}.json")
        _write_json(out_file, split_samples)

    print(f"  SpatialBench complete → {out_dir} ({len(all_samples)} total from {source_name})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DOWNLOADERS = {
    "vsr": download_vsr,
    "whatsup": download_whatsup,
    "gqa_spatial": download_gqa_spatial,
    "spatialbench": download_spatialbench,
}


def main():
    parser = argparse.ArgumentParser(
        description="Download spatial reasoning datasets for Stage 2 training."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/data/datasets",
        help="Root directory for datasets (default: /data/datasets)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DOWNLOADERS.keys()),
        choices=list(DOWNLOADERS.keys()),
        help="Which datasets to download (default: all)",
    )
    args = parser.parse_args()

    print(f"Data root: {args.data_root}")
    print(f"Datasets to download: {args.datasets}")
    print("=" * 60)

    for name in args.datasets:
        try:
            DOWNLOADERS[name](args.data_root)
        except Exception as e:
            print(f"\n  ERROR downloading {name}: {e}")
            import traceback
            traceback.print_exc()
            print(f"  Continuing with remaining datasets...")

    print("\n" + "=" * 60)
    print("Done! Summary:")
    for name in args.datasets:
        ds_dir = os.path.join(args.data_root, name)
        if os.path.isdir(ds_dir):
            files = os.listdir(ds_dir)
            json_files = [f for f in files if f.endswith((".json", ".jsonl"))]
            total = 0
            for jf in json_files:
                path = os.path.join(ds_dir, jf)
                try:
                    if jf.endswith(".jsonl"):
                        with open(path) as f:
                            total += sum(1 for _ in f)
                    else:
                        with open(path) as f:
                            total += len(json.load(f))
                except Exception:
                    pass
            print(f"  ✓ {name}: {json_files} ({total} total samples)")
        else:
            print(f"  ✗ {name}: not found")


if __name__ == "__main__":
    main()
