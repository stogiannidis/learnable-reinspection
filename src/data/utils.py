"""Qwen3-VL-oriented data utilities."""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple


def bbox_to_patch_mask(
    bbox: List[float],
    image_grid_thw: torch.LongTensor,
    spatial_merge_size: int = 2,
) -> torch.Tensor:
    """Map a normalized bbox to Qwen3-VL merged patch supervision."""
    t, h_patches, w_patches = image_grid_thw.tolist()
    h_merged = h_patches // spatial_merge_size
    w_merged = w_patches // spatial_merge_size

    x1, y1, x2, y2 = bbox

    col_start = int(x1 * w_merged)
    col_end = int(np.ceil(x2 * w_merged))
    row_start = int(y1 * h_merged)
    row_end = int(np.ceil(y2 * h_merged))

    col_start = max(0, min(col_start, w_merged - 1))
    col_end = max(1, min(col_end, w_merged))
    row_start = max(0, min(row_start, h_merged - 1))
    row_end = max(1, min(row_end, h_merged))

    n_patches = t * h_merged * w_merged
    mask = torch.zeros(n_patches, dtype=torch.float32)

    frames = torch.arange(t)[:, None, None]
    rows = torch.arange(row_start, row_end)[None, :, None]
    cols = torch.arange(col_start, col_end)[None, None, :]
    indices = frames * (h_merged * w_merged) + rows * w_merged + cols
    mask[indices.flatten()] = 1.0

    total = mask.sum()
    if total > 0:
        mask = mask / total

    return mask


def build_chat_messages(
    question: str,
    answer: Optional[str] = None,
    image_path: Optional[str] = None,
    image_url: Optional[str] = None,
) -> List[Dict]:
    """Build Qwen3-VL chat messages."""
    content = []

    if image_path is not None:
        content.append({"type": "image", "image": image_path})
    elif image_url is not None:
        content.append({"type": "image", "image": image_url})

    content.append({"type": "text", "text": question})

    messages = [{"role": "user", "content": content}]

    if answer is not None:
        messages.append({"role": "assistant", "content": answer})

    return messages
