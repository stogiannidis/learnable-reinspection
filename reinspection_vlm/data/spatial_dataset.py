"""Spatial VQA datasets for Stage 2 (unified, backend-specific)."""

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

logger = logging.getLogger(__name__)
_MAX_RETRIES = 10


class _EmptyDataset(Dataset):
    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise IndexError(idx)


def _squeeze_intern(batch: Dict) -> Dict:
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
    """JSON/JSONL spatial VQA; set ``backend`` to ``qwen3vl`` or ``internvl3``."""

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
        if backend not in ("qwen3vl", "internvl3"):
            raise ValueError(f"backend must be 'qwen3vl' or 'internvl3', got {backend}")
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
        return len(self.samples)

    def _intern_proc_kwargs(self) -> Dict:
        return {"return_tensors": "pt"}

    def __getitem__(self, idx) -> Dict:
        if self.backend == "internvl3":
            for _ in range(_MAX_RETRIES):
                try:
                    return self._getitem_intern(idx)
                except (OSError, SyntaxError, Image.UnidentifiedImageError) as exc:
                    logger.warning("Skipping bad sample %d: %s", idx, exc)
                    idx = random.randint(0, len(self.samples) - 1)
            return self._getitem_intern(idx)

        last_error = None
        for _ in range(3):
            try:
                return self._getitem_qwen(idx)
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                last_error = exc
                idx = (idx + 1) % len(self.samples)
        raise RuntimeError("Too many unreadable images in a row.") from last_error

    def _getitem_qwen(self, idx: int) -> Dict:
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
    if datasets is None:
        datasets = ["vsr", "whatsup", "gqa_spatial", "spatialbench"]

    dataset_configs = {
        "vsr": {
            "data_file": os.path.join(data_root, "vsr", f"{split}.jsonl"),
            "image_root": os.path.join(data_root, "vsr", "images"),
        },
        "whatsup": {
            "data_file": os.path.join(data_root, "whatsup", f"{split}.json"),
            "image_root": os.path.join(data_root, "whatsup", "images"),
        },
        "gqa_spatial": {
            "data_file": os.path.join(data_root, "gqa_spatial", f"{split}.json"),
            "image_root": os.path.join(data_root, "gqa_spatial", "images"),
        },
        "spatialbench": {
            "data_file": os.path.join(data_root, "spatialbench", f"{split}.json"),
            "image_root": os.path.join(data_root, "spatialbench", "images"),
        },
    }

    all_datasets = []
    for name in datasets:
        cfg = dataset_configs.get(name)
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
