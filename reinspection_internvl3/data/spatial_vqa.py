"""Compatibility shim — InternVL spatial VQA."""

from typing import List, Optional

from torch.utils.data import Dataset

from reinspection_vlm.data.spatial_dataset import SpatialVQADataset as _SpatialVQADataset
from reinspection_vlm.data.spatial_dataset import build_spatial_dataset as _build_spatial_dataset


class SpatialVQADataset(_SpatialVQADataset):
    def __init__(
        self,
        data_file: str,
        image_root: str,
        processor,
        split: str = "train",
        max_pixels: int = 1280 * 28 * 28,
        min_pixels: int = 4 * 28 * 28,
        crop_to_patches: bool = True,
        system_prompt: str = "You are a helpful assistant.",
        answer_ignore_index: int = -100,
    ):
        super().__init__(
            data_file=data_file,
            image_root=image_root,
            processor=processor,
            backend="internvl3",
            split=split,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            crop_to_patches=crop_to_patches,
            system_prompt=system_prompt,
            answer_ignore_index=answer_ignore_index,
        )


def build_spatial_dataset(
    data_root: str,
    processor,
    split: str = "train",
    datasets: Optional[List[str]] = None,
    max_pixels: int = 1280 * 28 * 28,
    min_pixels: int = 4 * 28 * 28,
    crop_to_patches: bool = True,
    system_prompt: str = "You are a helpful assistant.",
    answer_ignore_index: int = -100,
) -> Dataset:
    return _build_spatial_dataset(
        data_root=data_root,
        processor=processor,
        backend="internvl3",
        split=split,
        datasets=datasets,
        max_pixels=max_pixels,
        min_pixels=min_pixels,
        crop_to_patches=crop_to_patches,
        system_prompt=system_prompt,
        answer_ignore_index=answer_ignore_index,
    )
