"""RefCOCO/RefCOCO+/RefCOCOg for Stage 1 (backend-specific processing)."""

import json
import math
import os
from typing import Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

from .utils import bbox_to_patch_mask as qwen_bbox_to_patch_mask, build_chat_messages as qwen_build_chat
from .chat_template import build_chat_messages as intern_build_chat
from .gemma4_chat import build_chat_messages as gemma4_build_chat

_QWEN_BACKENDS = ("qwen25vl",)
_VALID_BACKENDS = ("qwen25vl", "internvl3", "gemma4")


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


def intern_bbox_to_patch_mask(
    bbox: List[float],
    num_image_patches: int,
    image_seq_length: int = 256,
) -> torch.Tensor:
    """Map a normalized bbox to InternVL image-token supervision."""
    side = int(math.isqrt(image_seq_length))
    if side * side != image_seq_length:
        raise ValueError(f"image_seq_length={image_seq_length} is not a square grid")

    x1, y1, x2, y2 = bbox
    col_start = max(0, min(int(math.floor(x1 * side)), side - 1))
    col_end = max(col_start + 1, min(int(math.ceil(x2 * side)), side))
    row_start = max(0, min(int(math.floor(y1 * side)), side - 1))
    row_end = max(row_start + 1, min(int(math.ceil(y2 * side)), side))

    base_mask = torch.zeros(side * side, dtype=torch.float32)
    for row in range(row_start, row_end):
        for col in range(col_start, col_end):
            base_mask[row * side + col] = 1.0

    if num_image_patches > 1:
        mask = base_mask.repeat(num_image_patches)
    else:
        mask = base_mask

    total = mask.sum()
    if total > 0:
        mask = mask / total
    return mask


class RefCOCODataset(Dataset):
    """RefCOCO family loader for all supported backends."""

    def __init__(
        self,
        data_root: str,
        processor,
        backend: str,
        split: str = "train",
        dataset_names: Optional[List[str]] = None,
        max_pixels: int = 1280 * 28 * 28,
        min_pixels: int = 4 * 28 * 28,
        crop_to_patches: bool = False,
        system_prompt: str = "You are a helpful assistant.",
        answer_ignore_index: int = -100,
    ):
        if backend not in _VALID_BACKENDS:
            raise ValueError(f"backend must be one of {_VALID_BACKENDS}, got {backend}")
        self.processor = processor
        self.backend = backend
        self.split = split
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.crop_to_patches = crop_to_patches
        self.system_prompt = system_prompt
        self.answer_ignore_index = answer_ignore_index
        self.image_seq_length = getattr(processor, "image_seq_length", 256)

        if dataset_names is None:
            dataset_names = ["refcoco", "refcoco+", "refcocog"]

        self.samples = []
        for name in dataset_names:
            ann_file = os.path.join(data_root, name, f"{split}.json")
            if not os.path.exists(ann_file):
                continue
            with open(ann_file, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            for item in data:
                item = dict(item)
                item["dataset"] = name
                if not os.path.isabs(item["image"]):
                    item["image"] = os.path.join(data_root, name, "images", item["image"])
                self.samples.append(item)

    def __len__(self) -> int:
        return len(self.samples)

    def _intern_num_patches(self, image: Image.Image) -> int:
        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is None or not hasattr(image_processor, "get_number_of_image_patches"):
            return 1
        kwargs = {
            "crop_to_patches": self.crop_to_patches,
            "max_pixels": self.max_pixels,
            "min_pixels": self.min_pixels,
        }
        try:
            return int(image_processor.get_number_of_image_patches(image.height, image.width, kwargs))
        except TypeError:
            return int(image_processor.get_number_of_image_patches(image.height, image.width))

    @staticmethod
    def _coco_to_norm(bbox, img_w, img_h):
        x, y, w, h = bbox
        return [x / img_w, y / img_h, (x + w) / img_w, (y + h) / img_h]

    def __getitem__(self, idx) -> Dict:
        sample = self.samples[idx]
        expression = sample["expression"]
        bbox = sample["bbox"]
        img_w = sample["image_w"]
        img_h = sample["image_h"]
        image_path = sample["image"]

        bbox_norm = self._coco_to_norm(bbox, img_w, img_h)
        answer = f"[{bbox_norm[0]:.3f}, {bbox_norm[1]:.3f}, {bbox_norm[2]:.3f}, {bbox_norm[3]:.3f}]"

        if self.backend in _QWEN_BACKENDS:
            question = f"Locate the following object in the image: {expression}"
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
            result = {
                k: (v.squeeze(0) if isinstance(v, torch.Tensor) and k not in _no_squeeze else v)
                for k, v in full_inputs.items()
            }
            if "image_grid_thw" in result:
                attn_target = qwen_bbox_to_patch_mask(
                    bbox_norm,
                    result["image_grid_thw"][0],
                    spatial_merge_size=2,
                )
                result["attn_target_mask"] = attn_target
            else:
                result["attn_target_mask"] = torch.tensor([])
            result["bbox_norm"] = torch.tensor(bbox_norm, dtype=torch.float32)
            return result

        if self.backend == "gemma4":
            question = f"Locate the following object in the image: {expression}"
            image = Image.open(image_path).convert("RGB")
            prompt_messages = gemma4_build_chat(
                question=question, image_path=image_path, system_prompt=self.system_prompt,
            )
            full_messages = gemma4_build_chat(
                question=question, answer=answer, image_path=image_path, system_prompt=self.system_prompt,
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
            result["attn_target_mask"] = torch.tensor([])
            result["bbox_norm"] = torch.tensor(bbox_norm, dtype=torch.float32)
            return result

        # internvl3
        question = f"Describe the location of: {expression}"
        image = Image.open(image_path).convert("RGB")
        prompt_messages = intern_build_chat(
            question=question, image_path=image_path, system_prompt=self.system_prompt,
        )
        full_messages = intern_build_chat(
            question=question, answer=answer, image_path=image_path, system_prompt=self.system_prompt,
        )
        prompt_text = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
        )
        full_text = self.processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False,
        )
        pk = {"return_tensors": "pt"}
        prompt_inputs = self.processor(text=[prompt_text], images=[image], **pk)
        full_inputs = self.processor(text=[full_text], images=[image], **pk)
        prompt_len = prompt_inputs["input_ids"].shape[-1]
        labels = full_inputs["input_ids"].clone()
        labels[:, :prompt_len] = self.answer_ignore_index
        full_inputs["labels"] = labels
        result = _squeeze_intern(full_inputs)
        result["attn_target_mask"] = intern_bbox_to_patch_mask(
            bbox=bbox_norm,
            num_image_patches=self._intern_num_patches(image),
            image_seq_length=self.image_seq_length,
        )
        result["bbox_norm"] = torch.tensor(bbox_norm, dtype=torch.float32)
        return result
