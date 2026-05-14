"""Spatial VQA datasets for stage-2 instruction tuning (multi-benchmark union).

``SpatialVQADataset`` loads JSON/JSONL files, applies backend-specific chat
templates and processors, and constructs causal LM labels by masking prompt
tokens.  ``build_spatial_dataset`` concatenates available corpora under a data
root into a single :class:`torch.utils.data.ConcatDataset`.
"""

import json
import logging
import os
import random
from typing import Dict, List, Optional

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import ConcatDataset, Dataset

from .utils import build_chat_messages as qwen_build_chat
from .chat_template import build_chat_messages as intern_build_chat
from .gemma4_chat import build_chat_messages as gemma4_build_chat
from .registry import dataset_path_configs, stage2_defaults

_QWEN_BACKENDS = ("qwen25vl",)
_VALID_BACKENDS = ("qwen25vl", "internvl3", "gemma4")

logger = logging.getLogger(__name__)
_MAX_RETRIES = 10


class _EmptyDataset(Dataset):
    """Placeholder dataset when no benchmark files exist on disk."""

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise IndexError(idx)


def _squeeze_intern(batch: Dict) -> Dict:
    """Remove batch dimension ``1`` from InternVL/Gemma processor outputs except images."""
    result = {}
    for key, value in batch.items():
        if not isinstance(value, torch.Tensor):
            result[key] = value
        elif key == "pixel_values":
            result[key] = value
        else:
            result[key] = value.squeeze(0)
    return result


class SpatialVQADataset(Dataset):
    """Single-file spatial VQA with per-backend tokenization and label masking."""

    def __init__(
        self,
        data_file: str,
        image_root: str,
        processor,
        backend: str,
        split: str = "train",
        max_pixels: int = 1280 * 28 * 28,
        min_pixels: int = 4 * 28 * 28,
        crop_to_patches: bool = True,
        system_prompt: str = "You are a helpful assistant.",
        answer_ignore_index: int = -100,
    ):
        """Load samples filtered by ``split`` and store preprocessing options.

        Args:
            data_file: Path to ``.json`` (list) or ``.jsonl`` (line-delimited) data.
            image_root: Base directory for relative image paths in items.
            processor: HF processor providing ``apply_chat_template`` and tensorization.
            backend: ``internvl3``, ``qwen25vl``, or ``gemma4``.
            split: Value of the optional ``split`` field to keep when present.
            max_pixels: Upper bound on resized pixels (Qwen/InternVL).
            min_pixels: Lower bound on resized pixels (Qwen/InternVL).
            crop_to_patches: Whether to enable dynamic cropping (InternVL stage 2).
            system_prompt: System message for templated backends.
            answer_ignore_index: Label mask value for non-target tokens.

        Raises:
            ValueError: If ``backend`` is not supported.
        """
        if backend not in _VALID_BACKENDS:
            raise ValueError(f"backend must be one of {_VALID_BACKENDS}, got {backend}")
        self.processor = processor
        self.backend = backend
        self.image_root = image_root
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.crop_to_patches = crop_to_patches
        self.system_prompt = system_prompt
        self.answer_ignore_index = answer_ignore_index
        self.samples = []
        if data_file.endswith(".jsonl"):
            with open(data_file, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line.strip())
                    if item.get("split", split) == split:
                        self.samples.append(item)
        else:
            with open(data_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.samples = [s for s in data if s.get("split", split) == split]

    def __len__(self):
        """Number of samples after split filtering."""
        return len(self.samples)

    def _intern_proc_kwargs(self) -> Dict:
        """Keyword arguments shared by InternVL-style processor calls."""
        return {"return_tensors": "pt"}

    def __getitem__(self, idx) -> Dict:
        """Return a model input dict with ``labels`` for the sample at ``idx``.

        InternVL/Gemma paths retry on corrupt images by resampling indices; Qwen
        advances sequentially on read errors before failing hard.
        """
        if self.backend in ("internvl3", "gemma4"):
            getter = self._getitem_gemma4 if self.backend == "gemma4" else self._getitem_intern
            for _ in range(_MAX_RETRIES):
                try:
                    return getter(idx)
                except (OSError, SyntaxError, Image.UnidentifiedImageError) as exc:
                    logger.warning("Skipping bad sample %d: %s", idx, exc)
                    idx = random.randint(0, len(self.samples) - 1)
            return getter(idx)

        last_error = None
        for _ in range(3):
            try:
                return self._getitem_qwen(idx)
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                last_error = exc
                idx = (idx + 1) % len(self.samples)
        raise RuntimeError("Too many unreadable images in a row.") from last_error

    def _getitem_qwen(self, idx: int) -> Dict:
        """Build Qwen2.5-VL tensors with masked labels after the prompt span."""
        sample = self.samples[idx]
        question = sample["question"]
        answer = sample["answer"]
        image_path = os.path.join(self.image_root, sample["image"])

        prompt_messages = qwen_build_chat(question, image_path=image_path)
        prompt_text = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
        )
        full_messages = qwen_build_chat(question, answer, image_path=image_path)
        full_text = self.processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False,
        )
        proc_kwargs = dict(
            images=[image_path],
            return_tensors="pt",
            max_pixels=self.max_pixels,
            min_pixels=self.min_pixels,
        )
        prompt_inputs = self.processor(text=[prompt_text], **proc_kwargs)
        full_inputs = self.processor(text=[full_text], **proc_kwargs)
        prompt_len = prompt_inputs["input_ids"].shape[-1]
        labels = full_inputs["input_ids"].clone()
        labels[:, :prompt_len] = -100
        full_inputs["labels"] = labels
        _no_squeeze = {"image_grid_thw", "video_grid_thw"}
        return {
            k: (v.squeeze(0) if isinstance(v, torch.Tensor) and k not in _no_squeeze else v)
            for k, v in full_inputs.items()
        }

    def _getitem_intern(self, idx: int) -> Dict:
        """Tokenize an InternVL sample from an on-disk RGB image."""
        sample = self.samples[idx]
        image_path = os.path.join(self.image_root, sample["image"])
        image = Image.open(image_path).convert("RGB")

        prompt_messages = intern_build_chat(
            question=sample["question"],
            image_path=image_path,
            system_prompt=self.system_prompt,
        )
        full_messages = intern_build_chat(
            question=sample["question"],
            answer=sample["answer"],
            image_path=image_path,
            system_prompt=self.system_prompt,
        )
        prompt_text = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
        )
        full_text = self.processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False,
        )
        pk = self._intern_proc_kwargs()
        prompt_inputs = self.processor(text=[prompt_text], images=[image], **pk)
        full_inputs = self.processor(text=[full_text], images=[image], **pk)
        prompt_len = prompt_inputs["input_ids"].shape[-1]
        labels = full_inputs["input_ids"].clone()
        labels[:, :prompt_len] = self.answer_ignore_index
        full_inputs["labels"] = labels
        result = _squeeze_intern(full_inputs)
        result["image_path"] = image_path
        return result

    def _getitem_gemma4(self, idx: int) -> Dict:
        """Tokenize a Gemma 4 multimodal sample (same masking contract as InternVL)."""
        sample = self.samples[idx]
        image_path = os.path.join(self.image_root, sample["image"])
        image = Image.open(image_path).convert("RGB")

        prompt_messages = gemma4_build_chat(
            question=sample["question"],
            image_path=image_path,
            system_prompt=self.system_prompt,
        )
        full_messages = gemma4_build_chat(
            question=sample["question"],
            answer=sample["answer"],
            image_path=image_path,
            system_prompt=self.system_prompt,
        )
        prompt_text = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
        )
        full_text = self.processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False,
        )
        pk = {"return_tensors": "pt", "images": [image]}
        prompt_inputs = self.processor(text=[prompt_text], **pk)
        full_inputs = self.processor(text=[full_text], **pk)
        prompt_len = prompt_inputs["input_ids"].shape[-1]
        labels = full_inputs["input_ids"].clone()
        labels[:, :prompt_len] = self.answer_ignore_index
        full_inputs["labels"] = labels
        result = _squeeze_intern(full_inputs)
        result["image_path"] = image_path
        return result


def build_spatial_dataset(
    data_root: str,
    processor,
    backend: str,
    split: str = "train",
    datasets: Optional[List[str]] = None,
    max_pixels: int = 1280 * 28 * 28,
    min_pixels: int = 4 * 28 * 28,
    crop_to_patches: bool = True,
    system_prompt: str = "You are a helpful assistant.",
    answer_ignore_index: int = -100,
) -> Dataset:
    """Concatenate all existing configured spatial datasets under ``data_root``.

    Args:
        data_root: Root folder containing per-benchmark subdirectories.
        processor: HF processor passed through to each child dataset.
        backend: VLM backend id forwarded to :class:`SpatialVQADataset`.
        split: Train/test filter for samples that expose a ``split`` field.
        datasets: Optional subset of benchmark names; defaults to a standard list.
        max_pixels: Passed to each child (Qwen/InternVL resizing budget).
        min_pixels: Minimum pixel budget for dynamic resizing.
        crop_to_patches: InternVL dynamic patch cropping toggle for stage 2.
        system_prompt: System string for templated backends.
        answer_ignore_index: Mask value for prompt tokens in ``labels``.

    Returns:
        A :class:`torch.utils.data.ConcatDataset` over available splits, or an
        :class:`_EmptyDataset` when nothing on disk matches the configuration.
    """
    if datasets is None:
        datasets = stage2_defaults()

    path_configs = dataset_path_configs(data_root, split)

    # VSR uses .jsonl; override extension for that dataset only
    if "vsr" in path_configs:
        path_configs["vsr"]["data_file"] = os.path.join(data_root, "vsr", f"{split}.jsonl")

    all_datasets = []
    for name in datasets:
        cfg = path_configs.get(name)
        if cfg is None or not os.path.exists(cfg["data_file"]):
            continue
        all_datasets.append(
            SpatialVQADataset(
                data_file=cfg["data_file"],
                image_root=cfg["image_root"],
                processor=processor,
                backend=backend,
                split=split,
                max_pixels=max_pixels,
                min_pixels=min_pixels,
                crop_to_patches=crop_to_patches,
                system_prompt=system_prompt,
                answer_ignore_index=answer_ignore_index,
            )
        )

    if not all_datasets:
        return _EmptyDataset()
    return ConcatDataset(all_datasets)
