"""Download and preprocess RefCOCO/RefCOCO+/RefCOCOg into the format expected by RefCOCODataset.

Creates:
  {output_root}/refcoco/train.json   (+ val.json, test.json)
  {output_root}/refcoco+/train.json  (+ val.json, test.json)
  {output_root}/refcocog/train.json  (+ val.json, test.json)
  {output_root}/{name}/images/       (symlink to COCO images)

Each JSON entry:
  {
    "image": "000000581857.jpg",
    "expression": "the red car on the left",
    "bbox": [x, y, w, h],       // COCO format (absolute pixels)
    "image_w": 427,
    "image_h": 640,
    "split": "train"
  }

Usage:
  python -m reinspection_qwen3vl.data.download_refcoco \
      --coco_images /data/datasets/coco/train2017 \
      --output_root /data/datasets
"""
import os
import json
import argparse


def convert_dataset(dataset_name, hf_name, coco_images_dir, output_root):
    """Download one RefCOCO variant from HuggingFace and convert."""
    from datasets import load_dataset

    print(f"Downloading {hf_name}...")
    ds = load_dataset(hf_name)

    out_dir = os.path.join(output_root, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    # Symlink images directory to COCO images
    images_link = os.path.join(out_dir, "images")
    if not os.path.exists(images_link):
        os.symlink(os.path.abspath(coco_images_dir), images_link)
        print(f"  Symlinked {images_link} -> {coco_images_dir}")

    # Map HF split names to our split names
    split_mapping = {}
    for split_name in ds:
        if "train" in split_name:
            split_mapping[split_name] = "train"
        elif "val" in split_name:
            split_mapping[split_name] = "val"
        else:
            # testA, testB, test — keep as-is
            split_mapping[split_name] = split_name

    # Collect samples per output split
    by_split = {}
    skipped = 0
    for hf_split, our_split in split_mapping.items():
        split_ds = ds[hf_split]
        samples = []

        for row in split_ds:
            extracted = _extract_samples(row, our_split, coco_images_dir)
            if extracted:
                samples.extend(extracted)
            else:
                skipped += 1

        if our_split not in by_split:
            by_split[our_split] = []
        by_split[our_split].extend(samples)
        print(f"  {hf_split} -> {our_split}: {len(samples)} expressions")

    # Merge testA + testB into combined "test" as well
    test_splits = [k for k in by_split if k.startswith("test")]
    if len(test_splits) > 1:
        combined = []
        for k in test_splits:
            combined.extend(by_split[k])
        by_split["test"] = combined

    # Write JSON files
    for split_name, samples in by_split.items():
        out_file = os.path.join(out_dir, f"{split_name}.json")
        with open(out_file, "w") as f:
            json.dump(samples, f)
        print(f"  Wrote {out_file}: {len(samples)} samples")

    if skipped:
        print(f"  Skipped {skipped} rows (missing image or bbox)")

    return by_split


def _extract_samples(row, split, coco_images_dir):
    """Extract sample(s) from a jxu124/refcoco HuggingFace row.

    HF schema:
      image_id: int, sentences: list[{raw, sent}], bbox: [x1, y1, x2, y2],
      raw_anns: JSON str with COCO bbox [x, y, w, h],
      raw_image_info: JSON str with width/height
    """
    # Construct COCO filename from image_id
    image_id = row.get("image_id")
    if image_id is None:
        return None
    file_name = f"{int(image_id):012d}.jpg"

    # Check image exists
    if not os.path.exists(os.path.join(coco_images_dir, file_name)):
        return None

    # Get COCO-format bbox [x, y, w, h] from raw_anns
    bbox = None
    raw_anns = row.get("raw_anns")
    if raw_anns:
        try:
            anns = json.loads(raw_anns) if isinstance(raw_anns, str) else raw_anns
            bbox = anns.get("bbox")
        except (json.JSONDecodeError, AttributeError):
            pass

    if bbox is None:
        # Fallback: convert [x1, y1, x2, y2] from HF bbox to [x, y, w, h]
        hf_bbox = row.get("bbox")
        if hf_bbox and len(hf_bbox) >= 4:
            x1, y1, x2, y2 = [float(v) for v in hf_bbox[:4]]
            bbox = [x1, y1, x2 - x1, y2 - y1]

    if bbox is None:
        return None

    bbox = [float(v) for v in bbox[:4]]

    # Get image dimensions from raw_image_info
    img_w, img_h = None, None
    raw_info = row.get("raw_image_info")
    if raw_info:
        try:
            info = json.loads(raw_info) if isinstance(raw_info, str) else raw_info
            img_w = info.get("width")
            img_h = info.get("height")
        except (json.JSONDecodeError, AttributeError):
            pass

    if img_w is None or img_h is None:
        # Fallback: read from image file
        try:
            from PIL import Image
            img = Image.open(os.path.join(coco_images_dir, file_name))
            img_w, img_h = img.size
        except Exception:
            return None

    # Extract expressions from sentences
    sentences = row.get("sentences", [])
    expressions = []
    if isinstance(sentences, list):
        for s in sentences:
            if isinstance(s, dict):
                text = s.get("raw", s.get("sent", ""))
            elif isinstance(s, str):
                text = s
            else:
                continue
            if text and text.strip():
                expressions.append(text.strip())

    # Fallback to captions field
    if not expressions:
        captions = row.get("captions", [])
        if isinstance(captions, list):
            expressions = [c.strip() for c in captions if isinstance(c, str) and c.strip()]

    if not expressions:
        return None

    samples = []
    for expr in expressions:
        samples.append({
            "image": file_name,
            "expression": expr,
            "bbox": bbox,
            "image_w": int(img_w),
            "image_h": int(img_h),
            "split": split,
        })

    return samples


def main():
    parser = argparse.ArgumentParser(description="Download and preprocess RefCOCO datasets")
    parser.add_argument("--coco_images", type=str, default="/data/datasets/coco/train2017",
                        help="Path to COCO train2017 images")
    parser.add_argument("--output_root", type=str, default="/data/datasets",
                        help="Output directory for processed datasets")
    parser.add_argument("--datasets", nargs="+",
                        default=["refcoco", "refcoco+", "refcocog"],
                        choices=["refcoco", "refcoco+", "refcocog"],
                        help="Which datasets to download")
    args = parser.parse_args()

    if not os.path.isdir(args.coco_images):
        raise FileNotFoundError(f"COCO images not found at {args.coco_images}")

    # HuggingFace dataset names
    hf_datasets = {
        "refcoco": "jxu124/refcoco",
        "refcoco+": "jxu124/refcocoplus",
        "refcocog": "jxu124/refcocog",
    }

    total = 0
    for name in args.datasets:
        hf_name = hf_datasets[name]
        print(f"\n{'='*60}")
        print(f"Processing {name} ({hf_name})")
        print(f"{'='*60}")
        by_split = convert_dataset(name, hf_name, args.coco_images, args.output_root)
        n = sum(len(v) for v in by_split.values())
        total += n
        print(f"  Total for {name}: {n}")

    print(f"\nDone. Total expressions across all datasets: {total}")
    print(f"Output directory: {args.output_root}")


if __name__ == "__main__":
    main()
