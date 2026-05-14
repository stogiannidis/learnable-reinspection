"""Download and convert external benchmarks to the standard {image, question, answer} JSON format.

Usage:
    python -m src.data.prepare_benchmarks --output_dir /data/datasets --benchmarks all
    python -m src.data.prepare_benchmarks --output_dir /data/datasets --benchmarks 3dsrbench blink
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile
from io import BytesIO
from pathlib import Path
from typing import List, Optional
from urllib.request import urlretrieve

from PIL import Image
from tqdm import tqdm

ALL_BENCHMARKS = ["3dsrbench", "mindcube", "blink", "srbench", "qspatial", "embspatial"]


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _save_json(data: list, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(data)} samples to {path}")


def _format_mcq(question: str, choices: dict[str, Optional[str]]) -> str:
    """Format a multiple-choice question with lettered options."""
    parts = [question]
    for letter, text in choices.items():
        if text is not None:
            parts.append(f"({letter}) {text}")
    parts.append("Answer with the option's letter from the given choices directly.")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# 3DSRBench                                                                    #
# --------------------------------------------------------------------------- #
def prepare_3dsrbench(output_dir: str) -> None:
    """Download ccvl/3DSRBench from HuggingFace and convert.

    Format: multiple-choice VQA on COCO images (image URLs).
    Columns: index, question, A, B, C, D, answer (letter), category, image_url
    """
    from datasets import load_dataset

    print("\n=== Preparing 3DSRBench ===")
    ds = load_dataset("ccvl/3DSRBench", split="test")

    bench_dir = _ensure_dir(os.path.join(output_dir, "3dsrbench"))
    img_dir = _ensure_dir(os.path.join(bench_dir, "images"))

    samples = []
    skipped = 0
    for row in tqdm(ds, desc="3DSRBench"):
        # Build choices dict
        choices = {}
        for letter in ["A", "B", "C", "D"]:
            val = row.get(letter)
            if val is not None and str(val).strip():
                choices[letter] = str(val).strip()

        if not choices:
            skipped += 1
            continue

        question_text = _format_mcq(row["question"], choices)
        answer_letter = row["answer"].strip()

        # Handle image: may be a URL or a PIL image from HF
        image_filename = f"{row['index']}.jpg"
        image_path = os.path.join(img_dir, image_filename)

        if not os.path.exists(image_path):
            if "image" in row and row["image"] is not None:
                # HF datasets may provide PIL images directly
                img = row["image"]
                if isinstance(img, Image.Image):
                    img.convert("RGB").save(image_path)
                else:
                    skipped += 1
                    continue
            elif "image_url" in row and row["image_url"]:
                try:
                    urlretrieve(row["image_url"], image_path)
                except Exception as e:
                    print(f"  Failed to download {row['image_url']}: {e}")
                    skipped += 1
                    continue
            else:
                skipped += 1
                continue

        samples.append({
            "image": image_filename,
            "question": question_text,
            "answer": answer_letter,
            "split": "test",
            "category": row.get("category", ""),
            "benchmark": "3dsrbench",
        })

    _save_json(samples, os.path.join(bench_dir, "test.json"))
    if skipped:
        print(f"  Skipped {skipped} samples (missing image/choices)")


# --------------------------------------------------------------------------- #
# MindCube                                                                     #
# --------------------------------------------------------------------------- #
def prepare_mindcube(output_dir: str) -> None:
    """Download MLL-Lab/MindCube from HuggingFace and convert.

    MindCube stores its data as a data.zip containing JSONL files and images.
    We use the tinybench split for evaluation.
    """
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    print("\n=== Preparing MindCube ===")
    bench_dir = _ensure_dir(os.path.join(output_dir, "mindcube"))
    img_dir = _ensure_dir(os.path.join(bench_dir, "images"))

    # Download the data.zip from HuggingFace
    zip_path = hf_hub_download(
        repo_id="MLL-Lab/MindCube",
        filename="data.zip",
        repo_type="dataset",
    )

    print(f"  Extracting {zip_path}...")
    extract_dir = os.path.join(bench_dir, "_extracted")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    # Find JSONL files - look for tinybench first, then any test/eval data
    jsonl_files = sorted(Path(extract_dir).rglob("*.jsonl"))
    print(f"  Found JSONL files: {[str(f.name) for f in jsonl_files]}")

    # Prefer tinybench for evaluation
    target_jsonl = None
    for jf in jsonl_files:
        if "tinybench" in jf.name.lower():
            target_jsonl = jf
            break
    if target_jsonl is None and jsonl_files:
        target_jsonl = jsonl_files[0]

    if target_jsonl is None:
        print("  ERROR: No JSONL files found in MindCube data.zip")
        # Try loading via HF datasets API as fallback
        try:
            ds = load_dataset("MLL-Lab/MindCube", split="train")
            _convert_mindcube_hf(ds, bench_dir, img_dir)
        except Exception as e:
            print(f"  Fallback also failed: {e}")
        return

    print(f"  Using: {target_jsonl.name}")
    samples = []
    img_base_dirs = list(Path(extract_dir).rglob("images")) + [target_jsonl.parent]

    with open(target_jsonl, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            question = row.get("question", row.get("prompt", ""))
            answer = row.get("answer", row.get("label", ""))
            if not question or answer == "":
                continue

            # Resolve image path
            image_ref = row.get("image", row.get("image_path", ""))
            if not image_ref:
                # Some entries may have images as list
                images = row.get("images", [])
                if images:
                    image_ref = images[0] if isinstance(images[0], str) else ""

            if not image_ref:
                continue

            # Try to find the image file
            image_filename = os.path.basename(image_ref)
            dest_path = os.path.join(img_dir, image_filename)

            if not os.path.exists(dest_path):
                # Search for the image in extracted dirs
                found = False
                for base in img_base_dirs:
                    candidate = base / image_ref if not os.path.isabs(image_ref) else Path(image_ref)
                    if candidate.exists():
                        shutil.copy2(str(candidate), dest_path)
                        found = True
                        break
                    # Also try just the filename
                    for match in Path(extract_dir).rglob(image_filename):
                        shutil.copy2(str(match), dest_path)
                        found = True
                        break
                    if found:
                        break
                if not found:
                    continue

            # Handle multiple-choice if present
            choices = row.get("choices", row.get("options", None))
            if choices and isinstance(choices, list):
                choice_dict = {}
                for i, c in enumerate(choices):
                    choice_dict[chr(ord("A") + i)] = str(c)
                question = _format_mcq(question, choice_dict)
                # If answer is an index, convert to letter
                if isinstance(answer, int):
                    answer = chr(ord("A") + answer)

            samples.append({
                "image": image_filename,
                "question": question,
                "answer": str(answer),
                "split": "test",
                "category": row.get("category", row.get("task_type", "")),
                "benchmark": "mindcube",
            })

    _save_json(samples, os.path.join(bench_dir, "test.json"))

    # Cleanup extracted temp dir
    shutil.rmtree(extract_dir, ignore_errors=True)


def _convert_mindcube_hf(ds, bench_dir: str, img_dir: str) -> None:
    """Fallback: convert MindCube from HF datasets API (imagefolder format)."""
    samples = []
    for i, row in enumerate(tqdm(ds, desc="MindCube (HF fallback)")):
        img = row.get("image")
        if img is None or not isinstance(img, Image.Image):
            continue

        image_filename = f"mindcube_{i:05d}.jpg"
        image_path = os.path.join(img_dir, image_filename)
        img.convert("RGB").save(image_path)

        # The HF imagefolder conversion may only have image+label
        question = row.get("question", row.get("text", ""))
        answer = row.get("answer", str(row.get("label", "")))

        if not question:
            continue

        samples.append({
            "image": image_filename,
            "question": question,
            "answer": str(answer),
            "split": "test",
            "benchmark": "mindcube",
        })

    _save_json(samples, os.path.join(bench_dir, "test.json"))


# --------------------------------------------------------------------------- #
# BLINK                                                                        #
# --------------------------------------------------------------------------- #

# Spatial-reasoning-relevant subtasks from BLINK
BLINK_SUBTASKS = [
    "Spatial_Relation",
    "Relative_Depth",
    "Object_Localization",
    "Multi-view_Reasoning",
    "Counting",
]


def _compose_blink_images(row: dict, img_dir: str, idx: str) -> Optional[str]:
    """Compose multiple BLINK images into a single grid image.

    BLINK tasks can have 1-4 images. We tile them into a 2x2 grid
    with labels so the model can reference "image 1", "image 2", etc.
    """
    images = []
    for k in ["image_1", "image_2", "image_3", "image_4"]:
        img = row.get(k)
        if img is not None and isinstance(img, Image.Image):
            images.append(img.convert("RGB"))

    if not images:
        return None

    if len(images) == 1:
        filename = f"{idx}.jpg"
        images[0].save(os.path.join(img_dir, filename))
        return filename

    # Compose into grid
    max_w = max(img.width for img in images)
    max_h = max(img.height for img in images)
    # Resize all to same size for clean grid
    target_w, target_h = min(max_w, 512), min(max_h, 512)
    resized = [img.resize((target_w, target_h)) for img in images]

    cols = 2 if len(resized) > 1 else 1
    rows = (len(resized) + cols - 1) // cols
    grid = Image.new("RGB", (cols * target_w, rows * target_h), (255, 255, 255))

    for j, img in enumerate(resized):
        r, c = divmod(j, cols)
        grid.paste(img, (c * target_w, r * target_h))

    filename = f"{idx}.jpg"
    grid.save(os.path.join(img_dir, filename))
    return filename


def prepare_blink(output_dir: str, subtasks: Optional[List[str]] = None) -> None:
    """Download BLINK-Benchmark/BLINK from HuggingFace and convert.

    Format: multiple-choice VQA with 1-4 images per question.
    We focus on spatial-reasoning-relevant subtasks by default.
    Multi-image questions get composed into a grid.
    """
    from datasets import load_dataset

    print("\n=== Preparing BLINK ===")
    bench_dir = _ensure_dir(os.path.join(output_dir, "blink"))
    img_dir = _ensure_dir(os.path.join(bench_dir, "images"))

    if subtasks is None:
        subtasks = BLINK_SUBTASKS

    all_samples = []

    for subtask in subtasks:
        print(f"  Loading subtask: {subtask}")
        try:
            ds = load_dataset("BLINK-Benchmark/BLINK", subtask, split="val")
        except Exception as e:
            print(f"    Failed to load {subtask}: {e}")
            continue

        for row in tqdm(ds, desc=f"BLINK/{subtask}"):
            idx = row.get("idx", f"{subtask}_{len(all_samples)}")

            # Compose images
            image_filename = _compose_blink_images(row, img_dir, idx)
            if image_filename is None:
                continue

            # Build question with choices
            prompt = row.get("prompt", row.get("question", ""))
            choices = row.get("choices", [])
            answer = row.get("answer", "").strip()

            if not prompt:
                continue

            # BLINK already includes choices in the prompt typically,
            # but if choices are separate, format them
            if choices and not any(f"({chr(65+i)})" in prompt for i in range(len(choices))):
                choice_dict = {}
                for i, c in enumerate(choices):
                    choice_dict[chr(ord("A") + i)] = str(c)
                question_text = _format_mcq(prompt, choice_dict)
            else:
                question_text = prompt
                if not question_text.rstrip().endswith("directly."):
                    question_text += "\nAnswer with the option's letter from the given choices directly."

            # Normalize answer: "(A)" -> "A"
            if answer.startswith("(") and answer.endswith(")"):
                answer = answer[1:-1]

            all_samples.append({
                "image": image_filename,
                "question": question_text,
                "answer": answer,
                "split": "test",
                "category": subtask,
                "benchmark": "blink",
            })

    _save_json(all_samples, os.path.join(bench_dir, "test.json"))


# --------------------------------------------------------------------------- #
# SRBench (Mind the Gap)                                                       #
# --------------------------------------------------------------------------- #
def prepare_srbench(output_dir: str) -> None:
    """Download stogian/srbenchv3 from HuggingFace and convert.

    Format: {image (PIL), question (str), answer (str)}
    This is already in our target format; we just need to save images to disk.
    """
    from datasets import load_dataset

    print("\n=== Preparing SRBench (v3) ===")
    bench_dir = _ensure_dir(os.path.join(output_dir, "srbench"))
    img_dir = _ensure_dir(os.path.join(bench_dir, "images"))

    try:
        ds = load_dataset("stogian/srbenchv3", split="test")
    except Exception as e:
        print(f"  Failed to load stogian/srbenchv3: {e}")
        print("  Trying all available splits...")
        try:
            ds_dict = load_dataset("stogian/srbenchv3")
            split_name = list(ds_dict.keys())[0]
            ds = ds_dict[split_name]
            print(f"  Using split: {split_name}")
        except Exception as e2:
            print(f"  Could not load SRBench v3 from HuggingFace: {e2}")
            return

    samples = []
    for i, row in enumerate(tqdm(ds, desc="SRBench")):
        img = row.get("image")
        question = row.get("question", "")
        answer = row.get("answer", "")

        if not question or answer == "":
            continue

        image_filename = f"srbench_{i:05d}.jpg"
        image_path = os.path.join(img_dir, image_filename)

        if not os.path.exists(image_path):
            if isinstance(img, Image.Image):
                img.convert("RGB").save(image_path)
            else:
                continue

        # SRBench may already include MCQ formatting in the question.
        # If it has choices, ensure consistent formatting.
        samples.append({
            "image": image_filename,
            "question": question,
            "answer": str(answer),
            "split": "test",
            "category": row.get("category", row.get("task_type", "")),
            "benchmark": "srbench",
        })

    _save_json(samples, os.path.join(bench_dir, "test.json"))


# --------------------------------------------------------------------------- #
# Q-Spatial-Bench                                                              #
# --------------------------------------------------------------------------- #
def prepare_qspatial(output_dir: str) -> None:
    """Download andrewliao11/Q-Spatial-Bench-v1 and convert.

    Free-form quantitative spatial reasoning (distance/height/width with a
    reference object). Produces {image, question, answer} with the ground-truth
    numeric value + unit joined as the answer string.
    """
    from datasets import load_dataset

    print("\n=== Preparing Q-Spatial-Bench ===")
    bench_dir = _ensure_dir(os.path.join(output_dir, "qspatial"))
    img_dir = _ensure_dir(os.path.join(bench_dir, "images"))

    repo_candidates = [
        "andrewliao11/Q-Spatial-Bench-v1",
        "andrewliao11/Q-Spatial-Bench",
    ]
    ds = None
    for repo in repo_candidates:
        try:
            ds_dict = load_dataset(repo)
            split_name = "test" if "test" in ds_dict else list(ds_dict.keys())[0]
            ds = ds_dict[split_name]
            print(f"  Loaded {repo} split={split_name} ({len(ds)} rows)")
            break
        except Exception as e:
            print(f"  Could not load {repo}: {e}")
    if ds is None:
        print("  Skipping Q-Spatial-Bench.")
        return

    samples = []
    skipped = 0
    for i, row in enumerate(tqdm(ds, desc="Q-Spatial-Bench")):
        img = row.get("image") or row.get("image_1") or row.get("img")
        question = row.get("question") or row.get("question_raw") or ""
        answer_value = row.get("answer_value", row.get("answer"))
        answer_unit = row.get("answer_unit", "")

        if img is None or not question or answer_value is None:
            skipped += 1
            continue

        image_filename = f"qspatial_{i:05d}.jpg"
        image_path = os.path.join(img_dir, image_filename)
        if not os.path.exists(image_path):
            if isinstance(img, Image.Image):
                img.convert("RGB").save(image_path)
            elif isinstance(img, str) and os.path.exists(img):
                Image.open(img).convert("RGB").save(image_path)
            else:
                skipped += 1
                continue

        answer_str = str(answer_value)
        if answer_unit:
            answer_str = f"{answer_str} {answer_unit}".strip()

        samples.append({
            "image": image_filename,
            "question": question,
            "answer": answer_str,
            "split": "test",
            "category": row.get("question_type", row.get("category", "quantitative")),
            "benchmark": "qspatial",
        })

    print(f"  Skipped {skipped} rows (missing image/question/answer)")
    _save_json(samples, os.path.join(bench_dir, "test.json"))


# --------------------------------------------------------------------------- #
# EmbSpatial-Bench                                                             #
# --------------------------------------------------------------------------- #
def prepare_embspatial(output_dir: str) -> None:
    """Download mengfeidu/EmbSpatial-Bench and convert.

    Embodied egocentric spatial MCQ (above/below/left/right/close/far)
    over indoor scenes. Produces {image, question, answer=letter}.
    """
    from datasets import load_dataset

    print("\n=== Preparing EmbSpatial-Bench ===")
    bench_dir = _ensure_dir(os.path.join(output_dir, "embspatial"))
    img_dir = _ensure_dir(os.path.join(bench_dir, "images"))

    repo_candidates = [
        "mengfeidu/EmbSpatial-Bench",
        "MengfeiDu/EmbSpatial-Bench",
    ]
    ds = None
    for repo in repo_candidates:
        try:
            ds_dict = load_dataset(repo)
            split_name = "test" if "test" in ds_dict else list(ds_dict.keys())[0]
            ds = ds_dict[split_name]
            print(f"  Loaded {repo} split={split_name} ({len(ds)} rows)")
            break
        except Exception as e:
            print(f"  Could not load {repo}: {e}")
    if ds is None:
        print("  Skipping EmbSpatial-Bench.")
        return

    samples = []
    skipped = 0
    for i, row in enumerate(tqdm(ds, desc="EmbSpatial-Bench")):
        img = row.get("image") or row.get("img")
        question_text = row.get("question", "")
        answer = row.get("answer", "")

        # Options may be under A/B/C/D columns, or under a single "options"/"choices" list.
        choices: dict[str, str] = {}
        if any(k in row for k in ("A", "B", "C", "D")):
            for letter in ["A", "B", "C", "D"]:
                val = row.get(letter)
                if val is not None and str(val).strip():
                    choices[letter] = str(val).strip()
        else:
            opts = row.get("options") or row.get("choices")
            if isinstance(opts, (list, tuple)):
                for letter, val in zip(["A", "B", "C", "D"], opts):
                    if val is not None and str(val).strip():
                        choices[letter] = str(val).strip()

        if img is None or not question_text or not choices or answer == "":
            skipped += 1
            continue

        # Normalize answer to a letter.
        ans_str = str(answer).strip()
        if ans_str not in choices:
            # Might be the answer *text* — map it back to a letter.
            match = next((L for L, v in choices.items() if v == ans_str), None)
            if match is None:
                skipped += 1
                continue
            ans_str = match

        image_filename = f"embspatial_{i:05d}.jpg"
        image_path = os.path.join(img_dir, image_filename)
        if not os.path.exists(image_path):
            if isinstance(img, Image.Image):
                img.convert("RGB").save(image_path)
            elif isinstance(img, str) and os.path.exists(img):
                Image.open(img).convert("RGB").save(image_path)
            else:
                skipped += 1
                continue

        samples.append({
            "image": image_filename,
            "question": _format_mcq(question_text, choices),
            "answer": ans_str,
            "split": "test",
            "category": row.get("category", row.get("relation", "embodied")),
            "benchmark": "embspatial",
        })

    print(f"  Skipped {skipped} rows (missing image/question/options/answer)")
    _save_json(samples, os.path.join(bench_dir, "test.json"))


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
PREPARE_FNS = {
    "3dsrbench": prepare_3dsrbench,
    "mindcube": prepare_mindcube,
    "blink": prepare_blink,
    "srbench": prepare_srbench,
    "qspatial": prepare_qspatial,
    "embspatial": prepare_embspatial,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare external benchmarks for evaluation")
    parser.add_argument(
        "--output_dir", type=str, default="/data/datasets",
        help="Root directory to save converted benchmark data",
    )
    parser.add_argument(
        "--benchmarks", nargs="+", default=["all"],
        help=f"Benchmarks to prepare: {ALL_BENCHMARKS} or 'all'",
    )
    parser.add_argument(
        "--blink_subtasks", nargs="+", default=None,
        help=f"BLINK subtasks to include (default: {BLINK_SUBTASKS})",
    )
    args = parser.parse_args()

    benchmarks = ALL_BENCHMARKS if "all" in args.benchmarks else args.benchmarks

    for bm in benchmarks:
        if bm not in PREPARE_FNS:
            print(f"Unknown benchmark: {bm}. Available: {ALL_BENCHMARKS}")
            continue
        if bm == "blink":
            prepare_blink(args.output_dir, subtasks=args.blink_subtasks)
        else:
            PREPARE_FNS[bm](args.output_dir)

    print("\nDone! Run evaluation with:")
    print(f"  python -m src.evaluate stage=eval benchmarks='[{','.join(benchmarks)}]'")


if __name__ == "__main__":
    main()
