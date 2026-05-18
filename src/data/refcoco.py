"""RefCOCO family loaders for stage-1 referring-expression grounding.

Produces model inputs with masked labels, normalized bounding boxes, and
backend-specific attention targets over vision tokens (patch grids).
"""

import json
import math
import os
from typing import Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

from .utils import (
    _overlap_area_grid,
    bbox_to_patch_mask as qwen_bbox_to_patch_mask,
    build_chat_messages as qwen_build_chat,
)
from .chat_template import build_chat_messages as intern_build_chat
from .gemma4_chat import build_chat_messages as gemma4_build_chat
from .llava_next_chat import build_chat_messages as llava_next_build_chat
from .registry import stage1_defaults

_QWEN_BACKENDS = ("qwen25vl",)
_VALID_BACKENDS = ("qwen25vl", "internvl3", "gemma4", "llava_next")
_COCO_IMAGE_DATASETS = {"refcoco", "refcoco+", "refcocog"}


def _resolve_grounding_image_path(
    data_root: str,
    dataset_name: str,
    image_name: str,
    coco_images_dir: Optional[str] = None,
) -> str:
    """Resolve prepared grounding image paths.

    Classic RefCOCO annotations store bare COCO ids and can use the canonical
    COCO image directory. Other prepared grounding datasets use dataset-local
    filenames such as ``grefcoco_*.jpg``, ``vg_*.jpg``, and ``grit_*.jpg``.
    """
    if os.path.isabs(image_name):
        return image_name

    dataset_image_path = os.path.join(data_root, dataset_name, "images", image_name)
    candidates = []
    if coco_images_dir is not None and dataset_name in _COCO_IMAGE_DATASETS:
        stem = os.path.splitext(image_name)[0]
        candidates.extend(
            [
                os.path.join(coco_images_dir, f"COCO_train2014_{stem}.jpg"),
                os.path.join(coco_images_dir, image_name),
            ]
        )
    candidates.append(dataset_image_path)

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def _squeeze_intern(batch: Dict) -> Dict:
    """Drop singleton batch dimensions from InternVL processor tensors except pixels."""
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
    """Map a normalized box to a probability mask over InternVL patch tokens.

    Args:
        bbox: ``[x1, y1, x2, y2]`` in normalized image coordinates.
        num_image_patches: Number of visual crops (mask is tiled when ``> 1``).
        image_seq_length: Square grid length ``side**2`` for patch indexing.

    Returns:
        Float tensor of length ``num_image_patches * image_seq_length`` summing
        to 1. Each entry is the fractional bbox-overlap area for that patch
        (tiled uniformly across crops when ``num_image_patches > 1``); tiny
        boxes that miss every patch boundary collapse to a single delta on the
        patch containing the bbox center.

    Raises:
        ValueError: If ``image_seq_length`` is not a perfect square.
    """
    side = int(math.isqrt(image_seq_length))
    if side * side != image_seq_length:
        raise ValueError(f"image_seq_length={image_seq_length} is not a square grid")

    base_mask = _overlap_area_grid(bbox, side, side).flatten()
    if num_image_patches > 1:
        # Replicate across crops; renormalize so the full vector sums to 1.
        mask = base_mask.unsqueeze(0).expand(num_image_patches, -1).contiguous().flatten()
        mask = mask / float(num_image_patches)
    else:
        mask = base_mask
    return mask


class RefCOCODataset(Dataset):
    """Merged RefCOCO/+ /g splits with referring strings and bounding boxes."""

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
        coco_images_dir: Optional[str] = None,
    ):
        """Scan annotation JSON files and build the in-memory sample list.

        Args:
            data_root: Directory containing ``refcoco`` (etc.) subfolders.
            processor: HF processor (provides ``image_seq_length`` when present).
            backend: ``internvl3``, ``qwen25vl``, or ``gemma4``.
            split: Which JSON split filename to read (``train`` / ``val`` / ``test``).
            dataset_names: Subset of dataset folder names; defaults to all three.
            max_pixels: InternVL dynamic resize upper bound.
            min_pixels: InternVL dynamic resize lower bound.
            crop_to_patches: Whether to request patch cropping from the image processor.
            system_prompt: System message for chat-templated backends.
            answer_ignore_index: Label mask for prompt tokens.
            coco_images_dir: If set, resolve classic RefCOCO numeric filenames
                from this canonical COCO directory, e.g.
                ``/data/datasets/coco/images/train2014``. Dataset-local
                prepared images still resolve from ``{data_root}/{name}/images``.

        Raises:
            ValueError: If ``backend`` is not supported.
        """
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
            dataset_names = stage1_defaults()

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
                item["image"] = _resolve_grounding_image_path(
                    data_root=data_root,
                    dataset_name=name,
                    image_name=item["image"],
                    coco_images_dir=coco_images_dir,
                )
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

        if self.backend == "llava_next":
            question = f"Locate the following object in the image: {expression}"
            image = Image.open(image_path).convert("RGB")
            prompt_messages = llava_next_build_chat(
                question=question, image_path=image_path, system_prompt=self.system_prompt,
            )
            full_messages = llava_next_build_chat(
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
            _no_squeeze = {"pixel_values", "image_sizes"}
            result = {
                k: (v.squeeze(0) if isinstance(v, torch.Tensor) and k not in _no_squeeze else v)
                for k, v in full_inputs.items()
            }
            # L_attn target — not wired for LLaVA-Next yet (would need a custom
            # patch-mask aligned to AnyRes packing). Leave empty so the trainer's
            # nan-safe fallback degenerates to uniform supervision (same as
            # Gemma4's current Stage-1 contract).
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
        # Count actual image tokens emitted by the processor (dynamic tiling can
        # produce more crops than ``get_number_of_image_patches`` reports when
        # we don't thread ``crop_to_patches``/pixel bounds through the call).
        image_token_id = getattr(self.processor, "image_token_id", None)
        if image_token_id is not None:
            n_image_tokens = int((result["input_ids"] == image_token_id).sum().item())
            num_tiles = max(1, n_image_tokens // self.image_seq_length)
        else:
            num_tiles = self._intern_num_patches(image)
        result["attn_target_mask"] = intern_bbox_to_patch_mask(
            bbox=bbox_norm,
            num_image_patches=num_tiles,
            image_seq_length=self.image_seq_length,
        )
        result["bbox_norm"] = torch.tensor(bbox_norm, dtype=torch.float32)
        return result
