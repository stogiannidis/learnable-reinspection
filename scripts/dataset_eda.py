"""Exploratory data analysis for project datasets.

The script is intentionally read-only for dataset roots. It summarizes the
training corpora registered in ``src.data.registry``, the evaluation benchmarks
registered in ``src.evaluate``, and any explicitly listed external datasets.
Large JSON arrays are sampled without loading the whole file into memory.

Usage:
    python scripts/dataset_eda.py --data-root /data/datasets
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

from PIL import Image, UnidentifiedImageError

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.registry import REGISTRY


BENCHMARKS = {
    "vsr": {"data_file": "vsr/test.jsonl", "image_root": "vsr/images"},
    "gqa_spatial": {"data_file": "gqa_spatial/test.json", "image_root": "gqa_spatial/images"},
    "whatsup": {"data_file": "whatsup/test.json", "image_root": "whatsup/images"},
    "3dsrbench": {"data_file": "3dsrbench/test.json", "image_root": "3dsrbench/images"},
    "mindcube": {"data_file": "mindcube/test.json", "image_root": "mindcube/images"},
    "blink": {"data_file": "blink/test.json", "image_root": "blink/images"},
    "srbench": {"data_file": "srbench/test.json", "image_root": "srbench/images"},
    "qspatial": {"data_file": "qspatial/test.json", "image_root": "qspatial/images"},
    "embspatial": {"data_file": "embspatial/test.json", "image_root": "embspatial/images"},
}

EXTERNAL_DATASETS = {
    "OpenSpatialDataset": {
        "data_file": "OpenSpatialDataset/result_10_depth_convs.json",
        "image_root": None,
        "kind": "external_hf_snapshot",
    }
}

LARGE_JSON_BYTES = 2 * 1024**3
TEXT_FIELDS = ("expression", "question", "answer", "image", "filename")
COUNTER_FIELDS = ("split", "category", "benchmark", "source", "dataset", "answer")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
LARGE_RECORD_CACHE: dict[tuple[str, int], int] = {}


@dataclass
class FileSummary:
    path: str
    exists: bool
    size_bytes: int = 0
    records: int | None = None
    record_count_method: str = ""
    sample_count: int = 0
    format: str = ""
    load_error: str = ""


@dataclass
class DatasetSummary:
    name: str
    group: str
    stage: int | None = None
    default: bool | None = None
    data_files: list[FileSummary] = field(default_factory=list)
    image_root: str | None = None
    image_root_exists: bool = False
    image_file_count: int | None = None
    image_dir_count: int | None = None
    schema_keys: dict[str, int] = field(default_factory=dict)
    text_lengths: dict[str, dict[str, float]] = field(default_factory=dict)
    counters: dict[str, dict[str, int]] = field(default_factory=dict)
    bbox: dict[str, float] = field(default_factory=dict)
    image_reference_sample: dict[str, int] = field(default_factory=dict)
    image_dimensions_sample: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    sample_preview: dict[str, Any] = field(default_factory=dict)


def human_size(n: int | None) -> str:
    if n is None:
        return "n/a"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(n)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{n} B"


def describe_numbers(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def q(p: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        idx = p * (len(ordered) - 1)
        lo = math.floor(idx)
        hi = math.ceil(idx)
        if lo == hi:
            return ordered[lo]
        return ordered[lo] * (hi - idx) + ordered[hi] * (idx - lo)

    return {
        "count": float(len(values)),
        "min": float(ordered[0]),
        "p25": float(q(0.25)),
        "median": float(median(ordered)),
        "mean": float(mean(ordered)),
        "p75": float(q(0.75)),
        "max": float(ordered[-1]),
    }


def safe_stat(path: Path) -> tuple[bool, int]:
    try:
        st = path.stat()
        return True, st.st_size
    except OSError:
        return False, 0


def compact_value(value: Any, max_len: int = 120) -> Any:
    if isinstance(value, str):
        return value if len(value) <= max_len else value[: max_len - 3] + "..."
    if isinstance(value, list):
        return [compact_value(v, max_len) for v in value[:3]]
    if isinstance(value, dict):
        return {str(k): compact_value(v, max_len) for k, v in list(value.items())[:6]}
    return value


def load_json_file(path: Path) -> tuple[int | None, list[Any], str, str]:
    exists, size = safe_stat(path)
    if not exists:
        return None, [], "", "missing"
    if size >= LARGE_JSON_BYTES:
        sample, error = sample_large_json_array(path, limit=1000)
        cache_key = (str(path), size)
        if cache_key in LARGE_RECORD_CACHE:
            records = LARGE_RECORD_CACHE[cache_key]
            method = "cached streamed filename-key count"
        else:
            records = count_large_open_spatial_records(path) if path.name == "result_10_depth_convs.json" else None
            method = "streamed filename-key count" if records is not None else "sampled only"
        return records, sample, method, error
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:  # noqa: BLE001 - report parse/read failures.
        return None, [], "", str(exc)
    if isinstance(data, list):
        return len(data), data[:1000], "json.load exact", ""
    if isinstance(data, dict):
        return len(data), [data], "json.load dict keys", ""
    return None, [data], "json.load scalar", ""


def load_jsonl_file(path: Path) -> tuple[int | None, list[Any], str]:
    exists, _ = safe_stat(path)
    if not exists:
        return None, [], "missing"
    sample: list[Any] = []
    count = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                count += 1
                if len(sample) < 1000:
                    sample.append(json.loads(line))
    except Exception as exc:  # noqa: BLE001
        return None, sample, str(exc)
    return count, sample, ""


def sample_large_json_array(path: Path, limit: int) -> tuple[list[Any], str]:
    decoder = json.JSONDecoder()
    sample: list[Any] = []
    buffer = ""
    started = False
    try:
        with path.open("r", encoding="utf-8") as handle:
            while len(sample) < limit:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                buffer += chunk
                pos = 0
                if not started:
                    while pos < len(buffer) and buffer[pos].isspace():
                        pos += 1
                    if pos < len(buffer) and buffer[pos] == "[":
                        pos += 1
                        started = True
                    buffer = buffer[pos:]
                    pos = 0
                while len(sample) < limit:
                    while pos < len(buffer) and buffer[pos] in " \r\n\t,":
                        pos += 1
                    if pos >= len(buffer) or buffer[pos] == "]":
                        break
                    try:
                        item, end = decoder.raw_decode(buffer, pos)
                    except json.JSONDecodeError:
                        break
                    sample.append(item)
                    pos = end
                buffer = buffer[pos:]
                if len(buffer) > 16 * 1024 * 1024:
                    buffer = buffer[-1024 * 1024 :]
    except Exception as exc:  # noqa: BLE001
        return sample, str(exc)
    return sample, ""


def count_large_open_spatial_records(path: Path) -> int:
    marker = b'"filename"'
    overlap = len(marker) - 1
    count = 0
    tail = b""
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(64 * 1024 * 1024)
            if not chunk:
                break
            data = tail + chunk
            count += data.count(marker)
            tail = data[-overlap:]
    return count


def count_image_root(path: Path | None) -> tuple[bool, int | None, int | None]:
    if path is None or not path.exists():
        return False, None, None
    files = 0
    dirs = 0
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                try:
                    if entry.is_dir(follow_symlinks=False):
                        dirs += 1
                    elif entry.is_file(follow_symlinks=False) or entry.is_symlink():
                        if Path(entry.name).suffix.lower() in IMAGE_EXTS:
                            files += 1
                except OSError:
                    continue
    except OSError:
        return True, None, None
    return True, files, dirs


def summarize_dataset(
    name: str,
    group: str,
    data_root: Path,
    files: list[str],
    image_root: str | None,
    stage: int | None = None,
    default: bool | None = None,
) -> DatasetSummary:
    summary = DatasetSummary(name=name, group=group, stage=stage, default=default)
    image_root_path = data_root / image_root if image_root else None
    summary.image_root = str(image_root_path) if image_root_path else None
    exists, image_files, image_dirs = count_image_root(image_root_path)
    summary.image_root_exists = exists
    summary.image_file_count = image_files
    summary.image_dir_count = image_dirs

    all_samples: list[dict[str, Any]] = []
    for rel in files:
        path = data_root / rel
        exists_file, size = safe_stat(path)
        fmt = "jsonl" if path.suffix == ".jsonl" else "json"
        if fmt == "jsonl":
            records, sample, error = load_jsonl_file(path)
            method = "line count exact" if records is not None else ""
        else:
            records, sample, method, error = load_json_file(path)
        summary.data_files.append(
            FileSummary(
                path=str(path),
                exists=exists_file,
                size_bytes=size,
                records=records,
                record_count_method=method,
                sample_count=len(sample),
                format=fmt,
                load_error=error,
            )
        )
        all_samples.extend([s for s in sample if isinstance(s, dict)])

    analyze_samples(summary, all_samples, image_root_path)
    add_warnings(summary)
    return summary


def analyze_samples(summary: DatasetSummary, samples: list[dict[str, Any]], image_root: Path | None) -> None:
    if not samples:
        return

    key_counts: Counter[str] = Counter()
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    text_values: dict[str, list[float]] = defaultdict(list)
    bbox_widths: list[float] = []
    bbox_heights: list[float] = []
    bbox_areas: list[float] = []
    image_refs: list[str] = []

    for item in samples:
        key_counts.update(item.keys())
        for field in TEXT_FIELDS:
            value = item.get(field)
            if isinstance(value, str):
                text_values[field].append(float(len(value)))
        for field in COUNTER_FIELDS:
            value = item.get(field)
            if isinstance(value, (str, int, float, bool)):
                counters[field][str(value)] += 1
        if "image" in item and isinstance(item["image"], str):
            image_refs.append(item["image"])
        elif "filename" in item and isinstance(item["filename"], str):
            image_refs.append(item["filename"])
        bbox = item.get("bbox")
        if isinstance(bbox, list) and len(bbox) >= 4:
            try:
                x, y, w, h = [float(v) for v in bbox[:4]]
                bbox_widths.append(w)
                bbox_heights.append(h)
                bbox_areas.append(max(0.0, w) * max(0.0, h))
            except (TypeError, ValueError):
                pass

    summary.schema_keys = dict(sorted(key_counts.items(), key=lambda kv: (-kv[1], kv[0])))
    summary.text_lengths = {field: describe_numbers(vals) for field, vals in text_values.items()}
    summary.counters = {
        field: dict(counter.most_common(20))
        for field, counter in counters.items()
        if counter
    }
    if bbox_widths:
        summary.bbox = {
            "width_mean": mean(bbox_widths),
            "height_mean": mean(bbox_heights),
            "area_median": median(bbox_areas),
            "area_mean": mean(bbox_areas),
            "area_max": max(bbox_areas),
        }

    if image_refs:
        checked = image_refs[:200]
        existing = 0
        if image_root is not None:
            for ref in checked:
                if (image_root / ref).exists():
                    existing += 1
        summary.image_reference_sample = {
            "sample_checked": len(checked),
            "sample_existing": existing if image_root is not None else 0,
            "sample_unique_refs": len(set(checked)),
        }
        if image_root is not None and checked:
            dims = sample_image_dimensions(image_root, checked[:40])
            summary.image_dimensions_sample = dims

    summary.sample_preview = compact_value(samples[0])


def sample_image_dimensions(image_root: Path, refs: Iterable[str]) -> dict[str, Any]:
    widths: list[float] = []
    heights: list[float] = []
    errors = 0
    opened = 0
    for ref in refs:
        path = image_root / ref
        if not path.exists():
            continue
        try:
            with Image.open(path) as img:
                widths.append(float(img.width))
                heights.append(float(img.height))
                opened += 1
        except (OSError, UnidentifiedImageError):
            errors += 1
    result: dict[str, Any] = {"opened": opened, "errors": errors}
    if widths:
        result["width"] = describe_numbers(widths)
        result["height"] = describe_numbers(heights)
    return result


def add_warnings(summary: DatasetSummary) -> None:
    present_files = [f for f in summary.data_files if f.exists]
    if not present_files:
        summary.warnings.append("No annotation file found.")
    for f in summary.data_files:
        if f.exists and f.load_error:
            summary.warnings.append(f"{Path(f.path).name}: {f.load_error}")
        if f.exists and f.records == 0:
            summary.warnings.append(f"{Path(f.path).name}: zero records.")
    if summary.image_root and not summary.image_root_exists:
        summary.warnings.append("Image root is missing.")
    if (
        summary.image_reference_sample
        and summary.image_root
        and summary.image_reference_sample.get("sample_existing", 0) == 0
        and summary.image_reference_sample.get("sample_checked", 0) > 0
    ):
        summary.warnings.append("No sampled image references resolved under image root.")


def build_specs(data_root: Path) -> list[tuple[str, str, list[str], str | None, int | None, bool | None]]:
    specs: list[tuple[str, str, list[str], str | None, int | None, bool | None]] = []
    for spec in REGISTRY:
        splits = []
        for split in ("train", "val", "test", "testB"):
            ext = ".jsonl" if spec.name == "vsr" else ".json"
            rel = f"{spec.subdir}/{split}{ext}"
            if (data_root / rel).exists() or split in ("train", "val"):
                splits.append(rel)
        specs.append((spec.name, f"stage{spec.stage}_training", splits, f"{spec.subdir}/images", spec.stage, spec.default))

    for name, cfg in BENCHMARKS.items():
        specs.append((name, "eval_benchmark", [cfg["data_file"]], cfg["image_root"], None, None))

    for name, cfg in EXTERNAL_DATASETS.items():
        specs.append((name, cfg["kind"], [cfg["data_file"]], cfg["image_root"], None, None))
    return specs


def write_report(path: Path, summaries: list[DatasetSummary], data_root: Path) -> None:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# Dataset EDA Report",
        "",
        f"- Generated: {generated}",
        f"- Data root: `{data_root}`",
        "- Scope: project training registry, evaluation benchmark registry, and OpenSpatialDataset snapshot.",
        "- Large-file policy: JSON files over 2 GiB are sampled; OpenSpatialDataset records are counted by streaming `\"filename\"` key occurrences.",
        "",
        "## Executive Summary",
        "",
    ]

    total_train = 0
    total_eval = 0
    for summary in summaries:
        count = sum(f.records or 0 for f in summary.data_files if f.exists and f.records is not None)
        if summary.group.startswith("stage"):
            total_train += count
        elif summary.group == "eval_benchmark":
            total_eval += count
    missing = [s.name for s in summaries if not any(f.exists for f in s.data_files)]
    warnings = [(s.name, w) for s in summaries for w in s.warnings]
    lines.extend(
        [
            f"- Training/eval annotation rows counted: {total_train:,} training rows and {total_eval:,} benchmark rows.",
            f"- Dataset entries analyzed: {len(summaries)}.",
            f"- Missing dataset entries: {', '.join(missing) if missing else 'none'}.",
            f"- Warning count: {len(warnings)}.",
            "",
            "## Dataset Inventory",
            "",
            "| Dataset | Group | Files | Records | Disk | Images | Warnings |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for summary in summaries:
        files = [f for f in summary.data_files if f.exists]
        records = sum(f.records or 0 for f in files if f.records is not None)
        disk = sum(f.size_bytes for f in files)
        images = "missing"
        if summary.image_root is None:
            images = "n/a"
        elif summary.image_file_count is not None:
            images = f"{summary.image_file_count:,}"
        elif summary.image_root_exists:
            images = "unknown"
        warning_text = "<br>".join(summary.warnings[:3])
        lines.append(
            f"| `{summary.name}` | {summary.group} | {len(files)}/{len(summary.data_files)} | "
            f"{records:,} | {human_size(disk)} | {images} | {warning_text} |"
        )

    lines.extend(["", "## Per-Dataset Notes", ""])
    for summary in summaries:
        lines.extend(render_dataset_section(summary))

    lines.extend(["", "## Recommendations", ""])
    lines.extend(
        [
            "- Fix or remove empty benchmark entries before full evaluation: `mindcube` is present but has zero rows; `qspatial` and `embspatial` are missing on this machine.",
            "- Treat `OpenSpatialDataset` as an external ShareGPT-style conversation corpus until an adapter maps it into this project's `{image, question, answer}` or grounding schema.",
            "- Do image-level split checks across shared pools called out in `docs/datasets.md`: COCO, VG, and GQA overlap at the photo-source level.",
            "- Add a CI smoke test that opens a small random sample of image references per dataset, because annotation files can exist even when shared image roots are absent or stale.",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_dataset_section(summary: DatasetSummary) -> list[str]:
    lines = [f"### {summary.name}", ""]
    lines.append(f"- Group: `{summary.group}`")
    if summary.stage is not None:
        lines.append(f"- Stage: {summary.stage}; default mix: {summary.default}")
    if summary.image_root:
        lines.append(
            f"- Image root: `{summary.image_root}`; exists: {summary.image_root_exists}; "
            f"top-level image files: {summary.image_file_count if summary.image_file_count is not None else 'unknown'}"
        )
    else:
        lines.append("- Image root: n/a")
    for f in summary.data_files:
        status = "present" if f.exists else "missing"
        records = "unknown" if f.records is None else f"{f.records:,}"
        lines.append(
            f"- `{Path(f.path).name}`: {status}, {human_size(f.size_bytes)}, "
            f"records: {records}, method: {f.record_count_method or 'n/a'}"
        )
    if summary.schema_keys:
        keys = ", ".join(f"`{k}` ({v})" for k, v in list(summary.schema_keys.items())[:12])
        lines.append(f"- Dominant keys in sample: {keys}")
    if summary.text_lengths:
        text_bits = []
        for field, stats in summary.text_lengths.items():
            text_bits.append(
                f"`{field}` median {stats.get('median', 0):.0f}, mean {stats.get('mean', 0):.1f}, max {stats.get('max', 0):.0f}"
            )
        lines.append("- Text lengths: " + "; ".join(text_bits))
    if summary.bbox:
        lines.append(
            "- BBox sample: "
            f"mean width {summary.bbox['width_mean']:.1f}, mean height {summary.bbox['height_mean']:.1f}, "
            f"median area {summary.bbox['area_median']:.1f}"
        )
    if summary.counters:
        for field, values in summary.counters.items():
            top = ", ".join(f"{k}: {v}" for k, v in list(values.items())[:8])
            lines.append(f"- Top `{field}` values: {top}")
    if summary.image_reference_sample:
        refs = summary.image_reference_sample
        lines.append(
            f"- Image reference sample: {refs.get('sample_existing', 0)}/{refs.get('sample_checked', 0)} resolved; "
            f"{refs.get('sample_unique_refs', 0)} unique in checked sample."
        )
    if summary.image_dimensions_sample.get("opened"):
        dims = summary.image_dimensions_sample
        lines.append(
            "- Image dimensions sample: "
            f"{dims['opened']} opened; width median {dims['width']['median']:.0f}; "
            f"height median {dims['height']['median']:.0f}"
        )
    if summary.warnings:
        lines.append("- Warnings: " + "; ".join(summary.warnings))
    if summary.sample_preview:
        lines.append("")
        lines.append("Sample preview:")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(summary.sample_preview, ensure_ascii=False, indent=2))
        lines.append("```")
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA for project datasets.")
    parser.add_argument("--data-root", type=Path, default=Path("/data/datasets"))
    parser.add_argument("--report", type=Path, default=Path("docs/dataset_eda.md"))
    parser.add_argument("--json", type=Path, default=Path("outputs/dataset_eda_summary.json"))
    args = parser.parse_args()

    if args.json.exists():
        try:
            previous = json.loads(args.json.read_text(encoding="utf-8"))
            for dataset in previous:
                for file_summary in dataset.get("data_files", []):
                    path = file_summary.get("path")
                    size = file_summary.get("size_bytes")
                    records = file_summary.get("records")
                    if isinstance(path, str) and isinstance(size, int) and isinstance(records, int):
                        LARGE_RECORD_CACHE[(path, size)] = records
        except Exception:
            LARGE_RECORD_CACHE.clear()

    summaries: list[DatasetSummary] = []
    for name, group, files, image_root, stage, default in build_specs(args.data_root):
        print(f"Analyzing {name} ({group})")
        summaries.append(
            summarize_dataset(
                name=name,
                group=group,
                data_root=args.data_root,
                files=files,
                image_root=image_root,
                stage=stage,
                default=default,
            )
        )

    write_report(args.report, summaries, args.data_root)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps([asdict(s) for s in summaries], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {args.report}")
    print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
