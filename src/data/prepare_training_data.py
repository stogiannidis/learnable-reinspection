"""Download and convert training datasets to the project's standard formats.

Stage 1 (grounding):  {image, expression, bbox, image_w, image_h} JSON
Stage 2 (spatial VQA): {image, question, answer, split} JSON/JSONL

Datasets:
  - grit:           GRIT grounding (Stage 1, streamed from zzliang/GRIT)
  - vg_grounding:   Visual Genome region descriptions → Stage 1 grounding (Stage 1)
                    Requires /data/datasets/vg/region_descriptions.json, image_data.json,
                    and images in /data/datasets/vg/VG_100K/ + VG_100K_2/.
  - rel3d:          Rel3D 3D spatial relations (Stage 2)
  - cambrian:       Cambrian-style spatial VQA mix (Stage 2)
  - clevr_spatial:  Synthetic CLEVR-like spatial reasoning (Stage 2)
  - vg_spatial:     Visual Genome relationships → spatial VQA (Stage 2)
                    Requires /data/datasets/vg/relationships.json and image_data.json
                    (downloaded from https://homes.cs.washington.edu/~ranjay/visualgenome/api.html)
                    and images in /data/datasets/vg/VG_100K/ + VG_100K_2/.

Usage:
    python -m src.data.prepare_training_data --output_dir /data/datasets --datasets all
    python -m src.data.prepare_training_data --output_dir /data/datasets --datasets grit --grit_max_samples 200000
    python -m src.data.prepare_training_data --output_dir /data/datasets --datasets clevr_spatial --clevr_num_scenes 50000
    python -m src.data.prepare_training_data --datasets cambrian_spatial --cambrian_source fallback
    python -m src.data.prepare_training_data --datasets vg_spatial --vg_input_dir /data/datasets/vg
    python -m src.data.prepare_training_data --datasets vg_grounding --vg_input_dir /data/datasets/vg

Cambrian: nyu-visionx/Cambrian-10M via HuggingFace streaming often raises KeyError('jpg')
    inside the WebDataset loader (inconsistent shards). Default --cambrian_source fallback
    uses GQA + Visual Genome only; use auto to try HF first, or hf for HF-only (no fallback).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import warnings
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlopen, urlretrieve

from PIL import Image, ImageDraw
from tqdm import tqdm

ALL_DATASETS = ["grit", "rel3d", "cambrian_spatial", "clevr_spatial", "vg_spatial", "vg_grounding", "grefcoco"]


# =========================================================================== #
# Helpers                                                                      #
# =========================================================================== #

def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _save_json(data: list, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(data)} samples to {path}")


def _save_jsonl(data: list, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(data)} samples to {path}")


def _download_image(url: str, dest: str, timeout: int = 10) -> bool:
    """Download an image from URL, return True on success."""
    try:
        resp = urlopen(url, timeout=timeout)
        data = resp.read()
        img = Image.open(BytesIO(data)).convert("RGB")
        img.save(dest)
        return True
    except Exception:
        return False


# =========================================================================== #
# Visual Genome region descriptions — Stage 1 grounding                       #
# =========================================================================== #

def prepare_vg_grounding(
    output_dir: str,
    vg_input_dir: str = "/data/datasets/vg",
    max_samples: int = 500_000,
    min_bbox_area: float = 0.002,
    seed: int = 42,
) -> None:
    """Convert VG region descriptions to Stage-1 grounding format.

    Reads region_descriptions.json and image_data.json from ``vg_input_dir``.
    Each region has a free-text phrase and an absolute bounding box — maps
    directly to the RefCOCODataset schema: {image, expression, bbox, image_w, image_h}.
    Images are symlinked (not copied) from VG_100K / VG_100K_2 to save disk space.
    Output goes to ``output_dir/vg_grounding/``.

    Args:
        output_dir: Root data directory (the pipeline's ``data_root``).
        vg_input_dir: Directory with region_descriptions.json, image_data.json,
            VG_100K/, and VG_100K_2/.
        max_samples: Maximum grounding pairs to emit.
        min_bbox_area: Minimum normalized bbox area; filters out tiny/degenerate regions.
        seed: Random seed for shuffling.
    """
    print("\n=== Preparing Visual Genome grounding (Stage 1) ===")

    reg_path = os.path.join(vg_input_dir, "region_descriptions.json")
    meta_path = os.path.join(vg_input_dir, "image_data.json")

    if not os.path.exists(reg_path):
        print(f"  region_descriptions.json not found at {reg_path}. Skipping.")
        return
    if not os.path.exists(meta_path):
        print(f"  image_data.json not found at {meta_path}. Skipping.")
        return

    vg_dir = _ensure_dir(os.path.join(output_dir, "vg_grounding"))
    img_out_dir = _ensure_dir(os.path.join(vg_dir, "images"))

    print("  Building image_id → metadata index ...")
    with open(meta_path, "r", encoding="utf-8") as f:
        image_meta = json.load(f)

    id_to_meta: Dict[int, Dict] = {}
    for entry in image_meta:
        id_to_meta[entry["image_id"]] = entry

    vg_image_dirs = [
        os.path.join(vg_input_dir, "VG_100K"),
        os.path.join(vg_input_dir, "VG_100K_2"),
    ]

    def _find_image(fname: str) -> Optional[str]:
        for d in vg_image_dirs:
            p = os.path.join(d, fname)
            if os.path.exists(p):
                return p
        return None

    print("  Scanning region descriptions ...")
    with open(reg_path, "r", encoding="utf-8") as f:
        reg_data = json.load(f)

    rng = random.Random(seed)
    samples = []

    for img_entry in tqdm(reg_data, desc="VG grounding"):
        if len(samples) >= max_samples:
            break

        iid = img_entry["id"]
        meta = id_to_meta.get(iid)
        if meta is None:
            continue

        img_w = meta["width"]
        img_h = meta["height"]
        if img_w <= 0 or img_h <= 0:
            continue

        url = meta.get("url", "")
        fname = os.path.basename(url) if url else f"{iid}.jpg"
        src_path = _find_image(fname)
        if src_path is None:
            continue

        out_fname = f"vg_{iid}.jpg"
        out_path = os.path.join(img_out_dir, out_fname)

        # Symlink instead of copy — images are already on disk
        if not os.path.exists(out_path):
            try:
                os.symlink(os.path.abspath(src_path), out_path)
            except Exception:
                continue

        for region in img_entry.get("regions", []):
            phrase = (region.get("phrase") or "").strip()
            if not phrase:
                continue

            x = region.get("x", 0)
            y = region.get("y", 0)
            w = region.get("width", 0)
            h = region.get("height", 0)

            if w <= 0 or h <= 0:
                continue

            # Filter tiny regions by normalized area
            norm_area = (w / img_w) * (h / img_h)
            if norm_area < min_bbox_area:
                continue

            # Clamp to image bounds
            x = max(0, min(x, img_w - 1))
            y = max(0, min(y, img_h - 1))
            w = max(1, min(w, img_w - x))
            h = max(1, min(h, img_h - y))

            samples.append({
                "image": out_fname,
                "expression": phrase,
                "bbox": [float(x), float(y), float(w), float(h)],
                "image_w": img_w,
                "image_h": img_h,
            })

            if len(samples) >= max_samples:
                break

    rng.shuffle(samples)
    split_idx = int(len(samples) * 0.95)
    _save_json(samples[:split_idx], os.path.join(vg_dir, "train.json"))
    _save_json(samples[split_idx:], os.path.join(vg_dir, "val.json"))
    print(f"  Produced {len(samples)} grounding pairs from VG region descriptions")


# =========================================================================== #
# gRefCOCO — Stage 1 grounding                                                #
# =========================================================================== #

def prepare_grefcoco(
    output_dir: str,
    coco_image_dir: str = "/data/datasets/coco/train2017",
    seed: int = 42,
) -> None:
    """Convert gRefCOCO to Stage-1 grounding format.

    Reads from the HuggingFace cache populated by FudanCVL/gRefCOCO (the parquet
    loader fails due to mixed segmentation types, so we read the raw JSONs directly).
    Images are symlinked from the COCO train2017 directory (image_id zero-padded to
    12 digits, e.g. 000000000072.jpg). Skips ``no_target`` entries (no bbox).

    Output: ``output_dir/grefcoco/{train,val}.json`` in RefCOCODataset schema.

    Args:
        output_dir: Root data directory.
        coco_image_dir: Path to COCO train2017 images.
        seed: Random seed for shuffling.
    """
    print("\n=== Preparing gRefCOCO (Stage 1 grounding) ===")

    import glob
    snapshots = glob.glob(
        os.path.expanduser(
            "~/.cache/huggingface/hub/datasets--FudanCVL--gRefCOCO/snapshots/*/grefs(unc).json"
        )
    )
    if not snapshots:
        print("  FudanCVL/gRefCOCO not in HF cache. Downloading ...")
        try:
            from datasets import load_dataset
            load_dataset("FudanCVL/gRefCOCO")
        except Exception:
            pass
        snapshots = glob.glob(
            os.path.expanduser(
                "~/.cache/huggingface/hub/datasets--FudanCVL--gRefCOCO/snapshots/*/grefs(unc).json"
            )
        )

    if not snapshots:
        print("  Could not locate gRefCOCO cache. Skipping.")
        return

    cache_dir = os.path.dirname(snapshots[0])
    grefs_path = snapshots[0]
    instances_path = os.path.join(cache_dir, "instances.json")

    print(f"  Reading from {cache_dir}")
    with open(grefs_path, "r", encoding="utf-8") as f:
        grefs = json.load(f)
    with open(instances_path, "r", encoding="utf-8") as f:
        instances = json.load(f)

    # Build ann_id → bbox + image dimensions lookup
    ann_id_to_info: Dict[int, Dict] = {}
    image_id_to_dims: Dict[int, Tuple[int, int]] = {}

    for img in instances["images"]:
        image_id_to_dims[img["id"]] = (img["width"], img["height"])
    for ann in instances["annotations"]:
        ann_id_to_info[ann["id"]] = {
            "bbox": ann["bbox"],  # [x, y, w, h] absolute COCO format
            "image_id": ann["image_id"],
        }

    gref_dir = _ensure_dir(os.path.join(output_dir, "grefcoco"))
    img_out_dir = _ensure_dir(os.path.join(gref_dir, "images"))

    train_samples, val_samples = [], []
    skipped_no_target = skipped_missing = 0

    for entry in tqdm(grefs, desc="gRefCOCO"):
        if entry.get("no_target"):
            skipped_no_target += 1
            continue

        image_id = entry["image_id"]
        split = entry.get("split", "train")

        # Resolve bbox from first ann_id
        ann_ids = entry.get("ann_id", [])
        if not ann_ids:
            skipped_missing += 1
            continue
        ann_info = ann_id_to_info.get(ann_ids[0])
        if ann_info is None:
            skipped_missing += 1
            continue

        dims = image_id_to_dims.get(image_id)
        if dims is None:
            skipped_missing += 1
            continue
        img_w, img_h = dims

        # Symlink image
        src_fname = f"{image_id:012d}.jpg"
        src_path = os.path.join(coco_image_dir, src_fname)
        if not os.path.exists(src_path):
            skipped_missing += 1
            continue

        out_fname = f"grefcoco_{image_id:012d}.jpg"
        out_path = os.path.join(img_out_dir, out_fname)
        if not os.path.exists(out_path):
            try:
                os.symlink(os.path.abspath(src_path), out_path)
            except Exception:
                skipped_missing += 1
                continue

        bbox = ann_info["bbox"]  # [x, y, w, h]

        # One sample per sentence
        for sent_info in entry.get("sentences", []):
            expression = sent_info.get("sent", sent_info.get("raw", "")).strip()
            if not expression:
                continue
            sample = {
                "image": out_fname,
                "expression": expression,
                "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                "image_w": img_w,
                "image_h": img_h,
            }
            if split in ("val", "testA", "testB"):
                val_samples.append(sample)
            else:
                train_samples.append(sample)

    rng = random.Random(seed)
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)

    _save_json(train_samples, os.path.join(gref_dir, "train.json"))
    _save_json(val_samples, os.path.join(gref_dir, "val.json"))
    print(f"  Skipped {skipped_no_target} no-target and {skipped_missing} missing entries")


# =========================================================================== #
# GRIT — Stage 1 grounding                                                    #
# =========================================================================== #

def prepare_grit(
    output_dir: str,
    max_samples: int = 200_000,
    min_bbox_area: float = 0.001,
    max_download_failures: int = 5000,
    seed: int = 42,
) -> None:
    """Download and convert GRIT grounding data for Stage 1.

    Source: HuggingFace zzliang/GRIT (config default, streamed).
    Each row has caption, url, and ref_exps: list of
    [char_start, char_end, x1, y1, x2, y2, confidence] with normalized boxes.
    Text spans are caption[start:end]. Output is RefCOCO-style JSON.

    Args:
        output_dir: Root data directory.
        max_samples: Maximum number of (expression, bbox) pairs to keep.
        min_bbox_area: Minimum normalized bbox area to filter tiny objects.
        max_download_failures: Stop after this many consecutive download failures.
        seed: Random seed for shuffling.
    """
    from datasets import load_dataset

    print("\n=== Preparing GRIT (Stage 1 grounding) ===")
    grit_dir = _ensure_dir(os.path.join(output_dir, "grit"))
    img_dir = _ensure_dir(os.path.join(grit_dir, "images"))

    # Stream the dataset to avoid downloading the full corpus
    print("  Streaming from zzliang/GRIT (config=default) ...")
    ds = load_dataset("zzliang/GRIT", "default", split="train", streaming=True)

    samples = []
    consecutive_failures = 0
    images_downloaded = 0

    for row in tqdm(ds, desc="GRIT", total=max_samples):
        if len(samples) >= max_samples:
            break

        url = row.get("url", "") or ""
        caption = row.get("caption", "") or ""
        ref_exps = row.get("ref_exps", []) or []

        if not url or not caption or not ref_exps:
            continue

        # Download image
        image_id = f"grit_{images_downloaded:07d}"
        image_filename = f"{image_id}.jpg"
        image_path = os.path.join(img_dir, image_filename)

        if not os.path.exists(image_path):
            if not _download_image(url, image_path):
                consecutive_failures += 1
                if consecutive_failures >= max_download_failures:
                    print(f"  Stopping: {max_download_failures} consecutive download failures")
                    break
                continue
        consecutive_failures = 0

        # Get image dimensions
        try:
            with Image.open(image_path) as img:
                img_w, img_h = img.size
        except Exception:
            os.remove(image_path)
            continue

        images_downloaded += 1

        # Each ref_exp: [char_start, char_end, x1, y1, x2, y2, confidence?]
        for ref in ref_exps:
            if not isinstance(ref, (list, tuple)) or len(ref) < 6:
                continue
            try:
                s, e = int(ref[0]), int(ref[1])
            except (TypeError, ValueError):
                continue
            s = max(0, min(s, len(caption)))
            e = max(s, min(e, len(caption)))
            x1, y1, x2, y2 = float(ref[2]), float(ref[3]), float(ref[4]), float(ref[5])

            # Filter degenerate / tiny boxes
            area = (x2 - x1) * (y2 - y1)
            if area < min_bbox_area or x2 <= x1 or y2 <= y1:
                continue

            # Convert to COCO format [x, y, w, h] in absolute pixels
            abs_x = x1 * img_w
            abs_y = y1 * img_h
            abs_w = (x2 - x1) * img_w
            abs_h = (y2 - y1) * img_h

            expression = caption[s:e].strip()
            if not expression or len(expression) < 2:
                continue

            samples.append({
                "image": image_filename,
                "expression": expression,
                "bbox": [abs_x, abs_y, abs_w, abs_h],
                "image_w": img_w,
                "image_h": img_h,
            })

            if len(samples) >= max_samples:
                break

    # Shuffle and save
    rng = random.Random(seed)
    rng.shuffle(samples)

    # 95/5 train/val split
    split_idx = int(len(samples) * 0.95)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    _save_json(train_samples, os.path.join(grit_dir, "train.json"))
    _save_json(val_samples, os.path.join(grit_dir, "val.json"))
    print(f"  Downloaded {images_downloaded} images, extracted {len(samples)} grounding pairs")


# =========================================================================== #
# Rel3D — Stage 2 spatial VQA                                                 #
# =========================================================================== #

_REL3D_RELATIONS = [
    "above", "below", "in front of", "behind",
    "left of", "right of", "on", "under",
]


def prepare_rel3d(output_dir: str) -> None:
    """Download and convert Rel3D for Stage 2 spatial VQA.

    Source: Rel3D from HuggingFace (hsiaotung/Rel3D).
    Each sample has an image with two objects and a 3D spatial relationship.
    We convert to binary yes/no questions + contrastive negatives.
    """
    from datasets import load_dataset

    print("\n=== Preparing Rel3D (Stage 2 spatial VQA) ===")
    rel3d_dir = _ensure_dir(os.path.join(output_dir, "rel3d"))
    img_dir = _ensure_dir(os.path.join(rel3d_dir, "images"))

    # Try loading from HuggingFace
    try:
        ds = load_dataset("hsiaotung/Rel3D", split="train")
    except Exception:
        print("  Primary source failed, trying alternative ...")
        try:
            ds = load_dataset("Rel3D/Rel3D", split="train")
        except Exception as e:
            print(f"  Could not load Rel3D: {e}")
            print("  Please download manually from the project page and place at:")
            print(f"    {rel3d_dir}/raw/")
            return

    # Relation antonyms for contrastive negatives
    antonyms = {
        "above": "below", "below": "above",
        "in front of": "behind", "behind": "in front of",
        "left of": "right of", "right of": "left of",
        "on": "under", "under": "on",
        "left": "right", "right": "left",
        "on top of": "below", "beneath": "above",
    }

    samples = []
    for i, row in enumerate(tqdm(ds, desc="Rel3D")):
        # Extract fields (adapt to actual schema)
        img = row.get("image")
        subject = row.get("subject", row.get("object1", ""))
        obj = row.get("object", row.get("object2", ""))
        relation = row.get("relation", row.get("predicate", ""))
        label = row.get("label", row.get("answer", 1))

        if not subject or not obj or not relation:
            continue

        # Save image
        image_filename = f"rel3d_{i:05d}.jpg"
        image_path = os.path.join(img_dir, image_filename)
        if not os.path.exists(image_path):
            if isinstance(img, Image.Image):
                img.convert("RGB").save(image_path)
            elif isinstance(img, str) and os.path.exists(img):
                Image.open(img).convert("RGB").save(image_path)
            else:
                continue

        # Positive question
        question = f"Is the {subject} {relation} the {obj}?"
        answer = "Yes" if label in (1, True, "True", "yes", "Yes") else "No"
        samples.append({
            "image": image_filename,
            "question": question,
            "answer": answer,
            "split": "train",
        })

        # Contrastive negative (swap relation)
        rel_lower = relation.lower().strip()
        if rel_lower in antonyms:
            neg_relation = antonyms[rel_lower]
            neg_question = f"Is the {subject} {neg_relation} the {obj}?"
            neg_answer = "No" if answer == "Yes" else "Yes"
            samples.append({
                "image": image_filename,
                "question": neg_question,
                "answer": neg_answer,
                "split": "train",
            })

    # Shuffle and split
    rng = random.Random(42)
    rng.shuffle(samples)
    split_idx = int(len(samples) * 0.9)

    _save_json(samples[:split_idx], os.path.join(rel3d_dir, "train.json"))
    _save_json(samples[split_idx:], os.path.join(rel3d_dir, "val.json"))


# =========================================================================== #
# Cambrian-style spatial VQA mix — Stage 2                                     #
# =========================================================================== #

# Spatial-relevant source datasets within Cambrian-10M
_CAMBRIAN_SPATIAL_SOURCES = [
    "gqa",          # scene graph QA (many spatial questions)
    "vg",           # visual genome
    "clevr",        # synthetic spatial reasoning
    "sqa",          # spatial QA
    "tallyqa",      # counting (related to spatial understanding)
]

# Keywords to filter for spatial content
_SPATIAL_KEYWORDS = {
    "left", "right", "above", "below", "behind", "front",
    "next to", "beside", "between", "near", "far",
    "top", "bottom", "under", "over", "inside", "outside",
    "closer", "farther", "nearest", "farthest",
    "on top of", "in front of", "to the left", "to the right",
    "adjacent", "opposite", "surrounding",
}


def _is_spatial_question(question: str) -> bool:
    """Check if a question involves spatial reasoning."""
    q_lower = question.lower()
    return any(kw in q_lower for kw in _SPATIAL_KEYWORDS)


def _prepare_cambrian_hf_stream(
    output_dir: str,
    max_samples: int,
    seed: int,
) -> None:
    """Stream nyu-visionx/Cambrian-10M and write train/val JSON. Raises on load/iter errors."""
    from datasets import load_dataset

    camb_dir = _ensure_dir(os.path.join(output_dir, "cambrian_spatial"))
    img_dir = _ensure_dir(os.path.join(camb_dir, "images"))

    print("  Streaming from nyu-visionx/Cambrian-10M ...")
    ds = load_dataset("nyu-visionx/Cambrian-10M", split="train", streaming=True)

    samples = []
    images_saved = 0
    skipped = 0
    buffer_multiplier = 3  # collect more, then subsample

    for row in tqdm(ds, desc="Cambrian", total=max_samples * buffer_multiplier):
        if len(samples) >= max_samples * buffer_multiplier:
            break

        source = row.get("source", row.get("dataset", "")).lower()
        question = row.get("question", row.get("conversations", [{}])[0].get("value", ""))
        answer = row.get("answer", "")

        # Extract from conversations format if needed
        convs = row.get("conversations", [])
        if not question and len(convs) >= 2:
            question = convs[0].get("value", "")
            answer = convs[1].get("value", "")

        if not question or not answer:
            continue

        # Filter for spatial content
        source_match = any(s in source for s in _CAMBRIAN_SPATIAL_SOURCES)
        spatial_match = _is_spatial_question(question)

        if not (source_match and spatial_match):
            # For non-source-match, require strong spatial signal
            if not spatial_match:
                skipped += 1
                continue

        # Handle image
        img = row.get("image", None)
        image_filename = f"cambrian_{images_saved:06d}.jpg"
        image_path = os.path.join(img_dir, image_filename)

        if not os.path.exists(image_path):
            if isinstance(img, Image.Image):
                img.convert("RGB").save(image_path)
            elif isinstance(img, str):
                if os.path.exists(img):
                    Image.open(img).convert("RGB").save(image_path)
                elif img.startswith("http"):
                    if not _download_image(img, image_path):
                        continue
                else:
                    continue
            else:
                continue

        images_saved += 1

        # Clean question: remove <image> tokens if present
        question_clean = question.replace("<image>", "").replace("<image>\n", "").strip()

        samples.append({
            "image": image_filename,
            "question": question_clean,
            "answer": answer.strip(),
            "split": "train",
            "source": source,
        })

    # Subsample to target size
    rng = random.Random(seed)
    if len(samples) > max_samples:
        samples = rng.sample(samples, max_samples)
    rng.shuffle(samples)

    split_idx = int(len(samples) * 0.95)
    _save_json(samples[:split_idx], os.path.join(camb_dir, "train.json"))
    _save_json(samples[split_idx:], os.path.join(camb_dir, "val.json"))
    print(f"  Kept {len(samples)} spatial samples from {images_saved} images (skipped {skipped} non-spatial)")


def prepare_cambrian_spatial(
    output_dir: str,
    max_samples: int = 100_000,
    seed: int = 42,
    source: str = "fallback",
) -> None:
    """Build a Cambrian-style spatial VQA mix for Stage 2.

    By default uses GQA + Visual Genome only (``source="fallback"``), because
    ``nyu-visionx/Cambrian-10M`` often fails inside HuggingFace's WebDataset
    loader with ``KeyError('jpg')`` on inconsistent shards.

    Args:
        output_dir: Root data directory.
        max_samples: Target number of spatial VQA samples.
        seed: Random seed.
        source: ``fallback`` — GQA + VG only; ``auto`` — try HF Cambrian-10M stream
            then fallback on error; ``hf`` — HF stream only (raises if it fails).
    """
    print("\n=== Preparing Cambrian-style spatial VQA mix ===")
    _ensure_dir(os.path.join(output_dir, "cambrian_spatial"))

    if source == "fallback":
        print(
            "  Using --cambrian_source fallback: skipping nyu-visionx/Cambrian-10M "
            "(HF WebDataset loader often raises KeyError('jpg') on bad shards). "
            "Building from GQA + Visual Genome.",
        )
        _prepare_cambrian_fallback(output_dir, max_samples, seed)
        return

    try:
        _prepare_cambrian_hf_stream(output_dir, max_samples, seed)
    except Exception as e:
        if source == "auto":
            print(f"  Cambrian-10M HuggingFace stream failed: {e}")
            print("  Falling back to individual source datasets ...")
            _prepare_cambrian_fallback(output_dir, max_samples, seed)
            return
        raise


def _prepare_cambrian_fallback(
    output_dir: str,
    max_samples: int = 100_000,
    seed: int = 42,
) -> None:
    """Fallback: build a spatial mix from GQA and Visual Genome directly."""
    from datasets import load_dataset

    camb_dir = _ensure_dir(os.path.join(output_dir, "cambrian_spatial"))
    img_dir = _ensure_dir(os.path.join(camb_dir, "images"))

    samples = []

    # --- GQA balanced (spatial subset) ---
    # lmms-lab/GQA splits images vs QA into *_images / *_instructions configs; merge on imageId.
    print("  Loading GQA (spatial subset) ...")
    try:
        per_source = max_samples // 2
        gqa_count = 0

        def _gqa_merge_split(split: str) -> None:
            nonlocal gqa_count
            img_ds = load_dataset(
                "lmms-lab/GQA",
                f"{split}_balanced_images",
                split=split,
            )
            id_to_image = {row["id"]: row["image"] for row in img_ds}
            inst_ds = load_dataset(
                "lmms-lab/GQA",
                f"{split}_balanced_instructions",
                split=split,
            )
            for row in tqdm(inst_ds, desc=f"GQA spatial ({split})"):
                if gqa_count >= per_source:
                    break
                question = row.get("question", "") or ""
                answer = row.get("answer", "") or row.get("fullAnswer", "") or ""
                img = id_to_image.get(row.get("imageId"))

                if not question or not answer or img is None:
                    continue
                if not _is_spatial_question(question):
                    continue

                image_filename = f"gqa_{gqa_count:06d}.jpg"
                image_path = os.path.join(img_dir, image_filename)

                if not os.path.exists(image_path):
                    if isinstance(img, Image.Image):
                        img.convert("RGB").save(image_path)
                    else:
                        continue

                samples.append({
                    "image": image_filename,
                    "question": question,
                    "answer": answer,
                    "split": "train",
                    "source": "gqa",
                })
                gqa_count += 1

        # testdev is small (~400 images, ~12k QA); train_balanced_images is ~72k/~10GB — avoid full merge here
        _gqa_merge_split("testdev")

        print(f"    Extracted {gqa_count} spatial questions from GQA")
    except Exception as e:
        print(f"    GQA load failed: {e}")

    # --- Visual Genome relationships (from local annotations) ---
    print("  Loading Visual Genome relationships (spatial subset, local) ...")
    vg_input_dir = "/data/datasets/vg"
    rel_path = os.path.join(vg_input_dir, "relationships.json")
    meta_path = os.path.join(vg_input_dir, "image_data.json")

    if os.path.exists(rel_path) and os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                image_meta_list = json.load(f)
            id_to_fname: Dict[str, str] = {}
            for entry in image_meta_list:
                url = entry.get("url", "")
                fname = os.path.basename(url) if url else f"{entry['image_id']}.jpg"
                id_to_fname[entry["image_id"]] = fname

            vg_image_dirs = [
                os.path.join(vg_input_dir, "VG_100K"),
                os.path.join(vg_input_dir, "VG_100K_2"),
            ]

            def _find_vg_img(fname: str) -> Optional[str]:
                for d in vg_image_dirs:
                    p = os.path.join(d, fname)
                    if os.path.exists(p):
                        return p
                return None

            with open(rel_path, "r", encoding="utf-8") as f:
                rel_data = json.load(f)

            per_source = max_samples // 2
            vg_count = 0

            for img_entry in tqdm(rel_data, desc="VG spatial"):
                if vg_count >= per_source:
                    break
                iid = img_entry["image_id"]
                fname = id_to_fname.get(iid)
                if fname is None:
                    continue
                src = _find_vg_img(fname)
                if src is None:
                    continue

                image_filename = f"cambrian_vg_{iid}.jpg"
                image_path = os.path.join(img_dir, image_filename)
                if not os.path.exists(image_path):
                    try:
                        import shutil
                        shutil.copy2(src, image_path)
                    except Exception:
                        continue

                for rel in img_entry.get("relationships", [])[:2]:
                    pred = rel.get("predicate", "").lower().strip()
                    if not any(kw in pred for kw in _SPATIAL_KEYWORDS):
                        continue
                    subj_info = rel.get("subject", {})
                    obj_info = rel.get("object", {})
                    subj = (subj_info.get("name") or (subj_info.get("names") or [""])[0]).strip().lower()
                    obj_name = (obj_info.get("name") or (obj_info.get("names") or [""])[0]).strip().lower()
                    if not subj or not obj_name:
                        continue
                    samples.append({
                        "image": image_filename,
                        "question": f"What is the spatial relationship between the {subj} and the {obj_name}?",
                        "answer": f"The {subj} is {pred} the {obj_name}.",
                        "split": "train",
                        "source": "visual_genome",
                    })
                    vg_count += 1

            print(f"    Extracted {vg_count} spatial relationships from VG (local)")
        except Exception as e:
            print(f"    VG local load failed: {e}")
    else:
        print(f"    VG annotations not found at {vg_input_dir}. Skipping VG.")

    rng = random.Random(seed)
    rng.shuffle(samples)
    split_idx = int(len(samples) * 0.95)

    _save_json(samples[:split_idx], os.path.join(camb_dir, "train.json"))
    _save_json(samples[split_idx:], os.path.join(camb_dir, "val.json"))


# =========================================================================== #
# Visual Genome spatial VQA — Stage 2                                          #
# =========================================================================== #

_VG_SPATIAL_PREDICATES = {
    "left of", "to the left of", "right of", "to the right of",
    "above", "below", "on top of", "under", "underneath", "beneath",
    "in front of", "behind", "next to", "beside", "near", "far from",
    "between", "inside", "outside", "on", "over",
}

_VG_QA_TEMPLATES = [
    ("What is the spatial relationship between the {subj} and the {obj}?",
     "The {subj} is {pred} the {obj}."),
    ("Where is the {subj} relative to the {obj}?",
     "The {subj} is {pred} the {obj}."),
    ("Is the {subj} {pred} the {obj}?", "Yes"),
]

_VG_ANTONYMS = {
    "left of": "right of", "to the left of": "to the right of",
    "right of": "left of", "to the right of": "to the left of",
    "above": "below", "below": "above",
    "on top of": "under", "under": "on top of",
    "in front of": "behind", "behind": "in front of",
}


def prepare_vg_spatial(
    output_dir: str,
    vg_input_dir: str = "/data/datasets/vg",
    max_samples: int = 200_000,
    seed: int = 42,
) -> None:
    """Convert locally downloaded Visual Genome annotations to Stage-2 spatial VQA.

    Reads relationships.json and image_data.json from ``vg_input_dir`` (both
    downloaded from https://homes.cs.washington.edu/~ranjay/visualgenome/api.html).
    Images are expected in ``vg_input_dir/VG_100K/`` and ``vg_input_dir/VG_100K_2/``.

    Each relationship whose predicate matches a spatial keyword is turned into
    a natural-language QA pair. A contrastive negative is added for antonym
    predicates. Outputs are saved to ``output_dir/vg_spatial/`` with symlinked
    (or copied) images.

    Args:
        output_dir: Root data directory (the pipeline's ``data_root``).
        vg_input_dir: Directory containing relationships.json, image_data.json,
            VG_100K/, and VG_100K_2/.
        max_samples: Maximum QA pairs to emit.
        seed: Random seed for shuffling / subsampling.
    """
    print("\n=== Preparing Visual Genome spatial VQA ===")

    rel_path = os.path.join(vg_input_dir, "relationships.json")
    meta_path = os.path.join(vg_input_dir, "image_data.json")

    if not os.path.exists(rel_path):
        print(f"  relationships.json not found at {rel_path}. Skipping.")
        return
    if not os.path.exists(meta_path):
        print(f"  image_data.json not found at {meta_path}. Skipping.")
        return

    vg_dir = _ensure_dir(os.path.join(output_dir, "vg_spatial"))
    img_out_dir = _ensure_dir(os.path.join(vg_dir, "images"))

    # Build image_id -> filename map from the two image directories
    print("  Building image_id → filename index ...")
    with open(meta_path, "r", encoding="utf-8") as f:
        image_meta = json.load(f)

    id_to_filename: Dict[int, str] = {}
    for entry in image_meta:
        iid = entry["image_id"]
        url = entry.get("url", "")
        fname = os.path.basename(url) if url else f"{iid}.jpg"
        id_to_filename[iid] = fname

    # Locate each image across the two VG_100K dirs
    vg_image_dirs = [
        os.path.join(vg_input_dir, "VG_100K"),
        os.path.join(vg_input_dir, "VG_100K_2"),
    ]

    def _find_image(fname: str) -> Optional[str]:
        for d in vg_image_dirs:
            p = os.path.join(d, fname)
            if os.path.exists(p):
                return p
        return None

    print("  Scanning relationships for spatial predicates ...")
    with open(rel_path, "r", encoding="utf-8") as f:
        rel_data = json.load(f)

    rng = random.Random(seed)
    samples = []

    for img_entry in tqdm(rel_data, desc="VG images"):
        if len(samples) >= max_samples:
            break

        image_id = img_entry["image_id"]
        fname = id_to_filename.get(image_id)
        if fname is None:
            continue

        src_path = _find_image(fname)
        if src_path is None:
            continue

        out_fname = f"vg_{image_id}.jpg"
        out_path = os.path.join(img_out_dir, out_fname)

        # Only copy when not already present
        if not os.path.exists(out_path):
            try:
                import shutil
                shutil.copy2(src_path, out_path)
            except Exception:
                continue

        for rel in img_entry.get("relationships", []):
            pred_raw = rel.get("predicate", "").lower().strip()

            # Normalise minor variants
            pred = pred_raw
            if pred == "left":
                pred = "left of"
            elif pred == "right":
                pred = "right of"

            if not any(sp in pred for sp in _VG_SPATIAL_PREDICATES):
                continue

            subj_info = rel.get("subject", {})
            obj_info = rel.get("object", {})

            subj = (subj_info.get("name") or
                    (subj_info.get("names") or [""])[0]).strip().lower()
            obj = (obj_info.get("name") or
                   (obj_info.get("names") or [""])[0]).strip().lower()

            if not subj or not obj or subj == obj:
                continue

            # Pick a random QA template for variety
            tmpl_q, tmpl_a = rng.choice(_VG_QA_TEMPLATES[:2])
            question = tmpl_q.format(subj=subj, obj=obj, pred=pred)
            answer = tmpl_a.format(subj=subj, obj=obj, pred=pred)

            samples.append({
                "image": out_fname,
                "question": question,
                "answer": answer,
                "split": "train",
                "source": "visual_genome",
            })

            # Contrastive negative for antonym predicates
            if pred in _VG_ANTONYMS and len(samples) < max_samples * 2:
                neg_pred = _VG_ANTONYMS[pred]
                neg_q = f"Is the {subj} {neg_pred} the {obj}?"
                samples.append({
                    "image": out_fname,
                    "question": neg_q,
                    "answer": "No",
                    "split": "train",
                    "source": "visual_genome",
                })

    rng.shuffle(samples)
    split_idx = int(len(samples) * 0.95)
    _save_json(samples[:split_idx], os.path.join(vg_dir, "train.json"))
    _save_json(samples[split_idx:], os.path.join(vg_dir, "val.json"))
    print(f"  Produced {len(samples)} spatial QA pairs from Visual Genome")


# =========================================================================== #
# CLEVR-based synthetic spatial reasoning — Stage 2                            #
# =========================================================================== #

_SHAPES = ["circle", "square", "triangle", "diamond", "pentagon", "hexagon"]
_COLORS = [
    ("red", (220, 50, 50)),
    ("blue", (50, 80, 220)),
    ("green", (50, 180, 60)),
    ("yellow", (230, 210, 40)),
    ("purple", (160, 50, 200)),
    ("orange", (240, 150, 30)),
    ("cyan", (40, 200, 220)),
    ("brown", (140, 90, 50)),
]
_SIZES = ["small", "large"]
_SIZE_RADII = {"small": (18, 28), "large": (35, 55)}

_SPATIAL_RELATIONS = [
    ("to the left of", lambda a, b: a[0] < b[0]),
    ("to the right of", lambda a, b: a[0] > b[0]),
    ("above", lambda a, b: a[1] < b[1]),
    ("below", lambda a, b: a[1] > b[1]),
    ("near", lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1]) < 120),
    ("far from", lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1]) > 200),
]


def _draw_shape(
    draw: ImageDraw.ImageDraw,
    shape: str,
    center: Tuple[int, int],
    radius: int,
    color: Tuple[int, int, int],
) -> None:
    """Draw a geometric shape on the canvas."""
    cx, cy = center
    if shape == "circle":
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=color, outline=(0, 0, 0), width=2,
        )
    elif shape == "square":
        draw.rectangle(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=color, outline=(0, 0, 0), width=2,
        )
    elif shape == "triangle":
        points = [
            (cx, cy - radius),
            (cx - radius, cy + radius),
            (cx + radius, cy + radius),
        ]
        draw.polygon(points, fill=color, outline=(0, 0, 0), width=2)
    elif shape == "diamond":
        points = [
            (cx, cy - radius),
            (cx + radius, cy),
            (cx, cy + radius),
            (cx - radius, cy),
        ]
        draw.polygon(points, fill=color, outline=(0, 0, 0), width=2)
    elif shape == "pentagon":
        points = []
        for k in range(5):
            angle = math.radians(90 + k * 72)
            points.append((cx + int(radius * math.cos(angle)), cy - int(radius * math.sin(angle))))
        draw.polygon(points, fill=color, outline=(0, 0, 0), width=2)
    elif shape == "hexagon":
        points = []
        for k in range(6):
            angle = math.radians(30 + k * 60)
            points.append((cx + int(radius * math.cos(angle)), cy - int(radius * math.sin(angle))))
        draw.polygon(points, fill=color, outline=(0, 0, 0), width=2)


def _generate_scene(
    rng: random.Random,
    canvas_size: int = 384,
    num_objects: Tuple[int, int] = (3, 6),
    bg_color: Tuple[int, int, int] = (240, 240, 240),
) -> Tuple[Image.Image, List[Dict]]:
    """Generate a synthetic scene with non-overlapping colored shapes.

    Returns the image and a list of object dicts with keys:
        shape, color_name, color_rgb, size, center, radius
    """
    n = rng.randint(*num_objects)
    margin = 60
    max_placement_attempts = 100

    # Pick unique (color, shape) combinations
    available = [(c, s) for c in _COLORS for s in _SHAPES]
    if n > len(available):
        n = len(available)
    chosen = rng.sample(available, n)

    objects = []
    for (color_name, color_rgb), shape in chosen:
        size = rng.choice(_SIZES)
        rmin, rmax = _SIZE_RADII[size]
        radius = rng.randint(rmin, rmax)

        # Place without overlap
        placed = False
        for _ in range(max_placement_attempts):
            cx = rng.randint(margin, canvas_size - margin)
            cy = rng.randint(margin, canvas_size - margin)

            # Check overlap with existing objects
            overlap = False
            for obj in objects:
                dist = math.hypot(cx - obj["center"][0], cy - obj["center"][1])
                if dist < radius + obj["radius"] + 15:
                    overlap = True
                    break
            if not overlap:
                placed = True
                break

        if not placed:
            continue

        objects.append({
            "shape": shape,
            "color_name": color_name,
            "color_rgb": color_rgb,
            "size": size,
            "center": (cx, cy),
            "radius": radius,
        })

    # Render
    img = Image.new("RGB", (canvas_size, canvas_size), bg_color)
    draw = ImageDraw.Draw(img)
    for obj in objects:
        _draw_shape(draw, obj["shape"], obj["center"], obj["radius"], obj["color_rgb"])

    return img, objects


def _describe_object(obj: Dict) -> str:
    """Human-readable description like 'large red circle'."""
    return f"{obj['size']} {obj['color_name']} {obj['shape']}"


def _generate_qa_pairs(
    objects: List[Dict],
    rng: random.Random,
    max_pairs: int = 4,
) -> List[Dict]:
    """Generate spatial QA pairs from a scene's objects."""
    if len(objects) < 2:
        return []

    qa_pairs = []

    # Binary yes/no questions
    for _ in range(max_pairs):
        obj_a, obj_b = rng.sample(objects, 2)
        relation_name, relation_fn = rng.choice(_SPATIAL_RELATIONS)

        desc_a = _describe_object(obj_a)
        desc_b = _describe_object(obj_b)

        result = relation_fn(obj_a["center"], obj_b["center"])
        question = f"Is the {desc_a} {relation_name} the {desc_b}?"
        answer = "Yes" if result else "No"

        qa_pairs.append({"question": question, "answer": answer})

    # "Where is X relative to Y?" questions
    if len(objects) >= 2:
        obj_a, obj_b = rng.sample(objects, 2)
        desc_a = _describe_object(obj_a)
        desc_b = _describe_object(obj_b)

        # Determine true relations
        true_relations = []
        for rel_name, rel_fn in _SPATIAL_RELATIONS[:4]:  # left/right/above/below only
            if rel_fn(obj_a["center"], obj_b["center"]):
                true_relations.append(rel_name)

        if true_relations:
            question = f"Where is the {desc_a} relative to the {desc_b}?"
            answer = f"The {desc_a} is {' and '.join(true_relations)} the {desc_b}."
            qa_pairs.append({"question": question, "answer": answer})

    # "What is [direction] of X?" questions
    if len(objects) >= 3:
        anchor = rng.choice(objects)
        desc_anchor = _describe_object(anchor)
        direction_name, direction_fn = rng.choice(_SPATIAL_RELATIONS[:4])

        matches = [
            o for o in objects
            if o is not anchor and direction_fn(o["center"], anchor["center"])
        ]
        if matches:
            closest = min(matches, key=lambda o: math.hypot(
                o["center"][0] - anchor["center"][0],
                o["center"][1] - anchor["center"][1],
            ))
            question = f"What object is {direction_name} the {desc_anchor}?"
            answer = f"The {_describe_object(closest)}."
            qa_pairs.append({"question": question, "answer": answer})

    # Counting questions
    for attr_type in ["color_name", "shape", "size"]:
        attr_val = rng.choice(objects)[attr_type]
        count = sum(1 for o in objects if o[attr_type] == attr_val)
        if attr_type == "color_name":
            question = f"How many {attr_val} objects are in the image?"
        elif attr_type == "shape":
            question = f"How many {attr_val}s are in the image?"
        else:
            question = f"How many {attr_val} objects are in the image?"
        qa_pairs.append({"question": question, "answer": str(count)})
        break  # one counting question per scene

    return qa_pairs


def prepare_clevr_spatial(
    output_dir: str,
    num_scenes: int = 50_000,
    max_qa_per_scene: int = 4,
    canvas_size: int = 384,
    seed: int = 42,
) -> None:
    """Generate CLEVR-like synthetic spatial reasoning data.

    Creates simple 2D scenes with colored geometric shapes and generates
    spatial relationship questions. Noise-free and unlimited scale.

    Args:
        output_dir: Root data directory.
        num_scenes: Number of scenes to generate.
        max_qa_per_scene: Max QA pairs per scene.
        canvas_size: Image size in pixels.
        seed: Random seed for reproducibility.
    """
    print("\n=== Generating CLEVR-like spatial data ===")
    clevr_dir = _ensure_dir(os.path.join(output_dir, "clevr_spatial"))
    img_dir = _ensure_dir(os.path.join(clevr_dir, "images"))

    rng = random.Random(seed)
    all_samples = []

    for scene_idx in tqdm(range(num_scenes), desc="CLEVR scenes"):
        image_filename = f"clevr_{scene_idx:06d}.jpg"
        image_path = os.path.join(img_dir, image_filename)

        img, objects = _generate_scene(rng, canvas_size=canvas_size)

        if len(objects) < 2:
            continue

        img.save(image_path, quality=95)

        qa_pairs = _generate_qa_pairs(objects, rng, max_pairs=max_qa_per_scene)
        for qa in qa_pairs:
            all_samples.append({
                "image": image_filename,
                "question": qa["question"],
                "answer": qa["answer"],
                "split": "train",
            })

    rng.shuffle(all_samples)
    split_idx = int(len(all_samples) * 0.95)

    _save_json(all_samples[:split_idx], os.path.join(clevr_dir, "train.json"))
    _save_json(all_samples[split_idx:], os.path.join(clevr_dir, "val.json"))
    print(f"  Generated {len(all_samples)} QA pairs from {num_scenes} scenes")


# =========================================================================== #
# CLI                                                                          #
# =========================================================================== #

PREPARE_FNS = {
    "grit": prepare_grit,
    "vg_grounding": prepare_vg_grounding,
    "grefcoco": prepare_grefcoco,
    "rel3d": prepare_rel3d,
    "cambrian_spatial": prepare_cambrian_spatial,
    "clevr_spatial": prepare_clevr_spatial,
    "vg_spatial": prepare_vg_spatial,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare training datasets (GRIT, Rel3D, Cambrian-style, CLEVR-synthetic)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="/data/datasets",
        help="Root directory to save converted data",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["all"],
        help=f"Datasets to prepare: {ALL_DATASETS} or 'all'",
    )

    # GRIT options
    parser.add_argument("--grit_max_samples", type=int, default=200_000,
                        help="Max grounding pairs to extract from GRIT")

    # Cambrian options
    parser.add_argument("--cambrian_max_samples", type=int, default=100_000,
                        help="Max spatial VQA samples from Cambrian")
    parser.add_argument(
        "--cambrian_source",
        choices=("fallback", "auto", "hf"),
        default="fallback",
        help="fallback=GQA+VG only (default; HF Cambrian-10M stream is often broken); "
        "auto=try HF Cambrian-10M then fallback; hf=HF stream only (no fallback)",
    )

    # CLEVR options
    parser.add_argument("--clevr_num_scenes", type=int, default=50_000,
                        help="Number of synthetic scenes to generate")
    parser.add_argument("--clevr_canvas_size", type=int, default=384,
                        help="Canvas size for CLEVR scenes in pixels")
    parser.add_argument("--clevr_max_qa", type=int, default=4,
                        help="Max QA pairs per CLEVR scene")

    # gRefCOCO options
    parser.add_argument("--coco_image_dir", type=str, default="/data/datasets/coco/train2017",
                        help="COCO train2017 image directory for gRefCOCO")

    # VG options
    parser.add_argument("--vg_input_dir", type=str, default="/data/datasets/vg",
                        help="Directory with VG relationships.json, image_data.json, VG_100K/, VG_100K_2/")
    parser.add_argument("--vg_max_samples", type=int, default=200_000,
                        help="Max spatial QA pairs to extract from Visual Genome")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    datasets = ALL_DATASETS if "all" in args.datasets else args.datasets

    for ds_name in datasets:
        if ds_name not in PREPARE_FNS:
            print(f"Unknown dataset: {ds_name}. Available: {ALL_DATASETS}")
            continue

        if ds_name == "grit":
            prepare_grit(args.output_dir, max_samples=args.grit_max_samples, seed=args.seed)
        elif ds_name == "cambrian_spatial":
            prepare_cambrian_spatial(
                args.output_dir,
                max_samples=args.cambrian_max_samples,
                seed=args.seed,
                source=args.cambrian_source,
            )
        elif ds_name == "clevr_spatial":
            prepare_clevr_spatial(
                args.output_dir,
                num_scenes=args.clevr_num_scenes,
                max_qa_per_scene=args.clevr_max_qa,
                canvas_size=args.clevr_canvas_size,
                seed=args.seed,
            )
        elif ds_name == "vg_grounding":
            prepare_vg_grounding(
                args.output_dir,
                vg_input_dir=args.vg_input_dir,
                max_samples=args.vg_max_samples,
                seed=args.seed,
            )
        elif ds_name == "grefcoco":
            prepare_grefcoco(
                args.output_dir,
                coco_image_dir=args.coco_image_dir,
                seed=args.seed,
            )
        elif ds_name == "vg_spatial":
            prepare_vg_spatial(
                args.output_dir,
                vg_input_dir=args.vg_input_dir,
                max_samples=args.vg_max_samples,
                seed=args.seed,
            )
        else:
            PREPARE_FNS[ds_name](args.output_dir)

    print("\n=== Done ===")
    print("Stage 1 data (GRIT) is compatible with RefCOCODataset:")
    print(f"  RefCOCODataset(data_root='{args.output_dir}', dataset_names=['grit', 'refcoco', ...])")
    print("Stage 2 data (Rel3D, Cambrian, CLEVR) — add to build_spatial_dataset config:")
    spatial_new = [d for d in datasets if d != "grit"]
    if spatial_new:
        print(f"  datasets={spatial_new}")


if __name__ == "__main__":
    main()
