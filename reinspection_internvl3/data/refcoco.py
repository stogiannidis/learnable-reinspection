"""Compatibility shim — InternVL RefCOCO."""

from typing import List, Optional

from reinspection_vlm.data.refcoco import RefCOCODataset as _RefCOCODataset


class RefCOCODataset(_RefCOCODataset):
    def __init__(
        self,
        data_root: str,
        processor,
        split: str = "train",
        dataset_names: Optional[List[str]] = None,
        max_pixels: int = 1280 * 28 * 28,
        min_pixels: int = 4 * 28 * 28,
        crop_to_patches: bool = False,
        system_prompt: str = "You are a helpful assistant.",
        answer_ignore_index: int = -100,
    ):
        super().__init__(
            data_root=data_root,
            processor=processor,
            backend="internvl3",
            split=split,
            dataset_names=dataset_names,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            crop_to_patches=crop_to_patches,
            system_prompt=system_prompt,
            answer_ignore_index=answer_ignore_index,
        )
