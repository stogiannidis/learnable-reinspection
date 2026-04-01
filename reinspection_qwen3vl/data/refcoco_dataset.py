"""Compatibility shim — Qwen3-VL RefCOCO with ``backend='qwen3vl'``."""

from reinspection_vlm.data.refcoco import RefCOCODataset as _RefCOCODataset


class RefCOCODataset(_RefCOCODataset):
    """Same as ``reinspection_vlm.data.refcoco.RefCOCODataset`` with Qwen defaults."""

    def __init__(self, data_root, processor, split="train", dataset_names=None, max_pixels=1280 * 28 * 28, min_pixels=4 * 28 * 28):
        super().__init__(
            data_root=data_root,
            processor=processor,
            backend="qwen3vl",
            split=split,
            dataset_names=dataset_names,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
        )
