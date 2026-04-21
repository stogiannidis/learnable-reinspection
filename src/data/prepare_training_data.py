"""Download and convert training datasets to the project's standard formats.

Stage 1 (grounding):  {image, expression, bbox, image_w, image_h} JSON
Stage 2 (spatial VQA): {image, question, answer, split} JSON/JSONL

Datasets:
  - grit:           GRIT grounding (Stage 1, streamed from zzliang/GRIT)
  - rel3d:          Rel3D 3D spatial relations (Stage 2)
  - cambrian:       Cambrian-style spatial VQA mix (Stage 2)
  - clevr_spatial:  Synthetic CLEVR-like spatial reasoning (Stage 2)

Usage:
    python -m src.data.prepare_training_data --output_dir /data/datasets --datasets all
    python -m src.data.prepare_training_data --output_dir /data/datasets --datasets grit --grit_max_samples 200000
    python -m src.data.prepare_training_data --output_dir /data/datasets --datasets clevr_spatial --clevr_num_scenes 50000
    python -m src.data.prepare_training_data --datasets cambrian --cambrian_source fallback

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

ALL_DATASETS = ["grit", "rel3d", "cambrian", "clevr_spatial"]


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


def prepare_cambrian(
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

    # --- Visual Genome relationships ---
    print("  Loading Visual Genome relationships (spatial subset) ...")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            vg_ds = load_dataset(
                "visual_genome",
                "relationships_v1.2.0",
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
        per_source = max_samples // 2
        vg_count = 0

        for row in tqdm(vg_ds, desc="VG spatial", total=per_source):
            if vg_count >= per_source:
                break

            relationships = row.get("relationships", [])
            img = row.get("image")

            spatial_rels = []
            for rel in relationships:
                predicate = rel.get("predicate", "").lower().strip()
                if any(kw in predicate for kw in _SPATIAL_KEYWORDS):
                    spatial_rels.append(rel)

            if not spatial_rels:
                continue

            image_filename = f"vg_{vg_count:06d}.jpg"
            image_path = os.path.join(img_dir, image_filename)

            if not os.path.exists(image_path):
                if isinstance(img, Image.Image):
                    img.convert("RGB").save(image_path)
                else:
                    continue

            # Create VQA pairs from relationships
            for rel in spatial_rels[:2]:  # limit per image
                subj = rel.get("subject", {}).get("name", rel.get("subject_name", ""))
                obj_name = rel.get("object", {}).get("name", rel.get("object_name", ""))
                predicate = rel.get("predicate", "")

                if not subj or not obj_name:
                    continue

                question = f"What is the spatial relationship between the {subj} and the {obj_name}?"
                answer = f"The {subj} is {predicate} the {obj_name}."

                samples.append({
                    "image": image_filename,
                    "question": question,
                    "answer": answer,
                    "split": "train",
                    "source": "visual_genome",
                })
                vg_count += 1

        print(f"    Extracted {vg_count} spatial relationships from VG")
    except Exception as e:
        print(f"    VG load failed: {e}")

    rng = random.Random(seed)
    rng.shuffle(samples)
    split_idx = int(len(samples) * 0.95)

    _save_json(samples[:split_idx], os.path.join(camb_dir, "train.json"))
    _save_json(samples[split_idx:], os.path.join(camb_dir, "val.json"))


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
    "rel3d": prepare_rel3d,
    "cambrian": prepare_cambrian,
    "clevr_spatial": prepare_clevr_spatial,
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

    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    datasets = ALL_DATASETS if "all" in args.datasets else args.datasets

    for ds_name in datasets:
        if ds_name not in PREPARE_FNS:
            print(f"Unknown dataset: {ds_name}. Available: {ALL_DATASETS}")
            continue

        if ds_name == "grit":
            prepare_grit(args.output_dir, max_samples=args.grit_max_samples, seed=args.seed)
        elif ds_name == "cambrian":
            prepare_cambrian(
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
