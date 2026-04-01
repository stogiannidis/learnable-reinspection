"""Compatibility shim — Qwen3-VL spatial datasets."""

from typing import List, Optional

from torch.utils.data import ConcatDataset, Dataset

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
    ):
        super().__init__(
            data_file=data_file,
            image_root=image_root,
            processor=processor,
            backend="qwen3vl",
            split=split,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
        )


def build_spatial_dataset(
    data_root: str,
    processor,
    split: str = "train",
    datasets: Optional[List[str]] = None,
    max_pixels: int = 1280 * 28 * 28,
    min_pixels: int = 4 * 28 * 28,
) -> Dataset:
    return _build_spatial_dataset(
        data_root=data_root,
        processor=processor,
        backend="qwen3vl",
        split=split,
        datasets=datasets,
        max_pixels=max_pixels,
        min_pixels=min_pixels,
    )
