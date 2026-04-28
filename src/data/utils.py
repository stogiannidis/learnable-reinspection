"""Qwen2.5-VL data helpers: chat message layout and bbox-to-patch supervision."""

import torch
from typing import Dict, List, Optional, Tuple


def _overlap_area_grid(
    bbox: List[float],
    h_merged: int,
    w_merged: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-patch fractional overlap with a normalized bbox over a 2D grid.

    Returns a ``(h_merged, w_merged)`` simplex (sums to 1). For tiny boxes that
    don't overlap any patch by more than ``eps`` of grid area, places full mass
    on the patch containing the bbox center.
    """
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(1.0, float(x1)))
    x2 = max(0.0, min(1.0, float(x2)))
    y1 = max(0.0, min(1.0, float(y1)))
    y2 = max(0.0, min(1.0, float(y2)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    col_edges = torch.linspace(0.0, 1.0, w_merged + 1, dtype=torch.float32)
    row_edges = torch.linspace(0.0, 1.0, h_merged + 1, dtype=torch.float32)

    col_lo = col_edges[:-1].clamp(min=x1)
    col_hi = col_edges[1:].clamp(max=x2)
    col_overlap = (col_hi - col_lo).clamp(min=0.0)         # (W,)

    row_lo = row_edges[:-1].clamp(min=y1)
    row_hi = row_edges[1:].clamp(max=y2)
    row_overlap = (row_hi - row_lo).clamp(min=0.0)         # (H,)

    grid = row_overlap[:, None] * col_overlap[None, :]     # (H, W)
    total = grid.sum()
    if total > eps:
        return grid / total

    # Tiny-box fallback: all mass on the patch containing the bbox center.
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    c = min(int(cx * w_merged), w_merged - 1)
    r = min(int(cy * h_merged), h_merged - 1)
    grid = torch.zeros(h_merged, w_merged, dtype=torch.float32)
    grid[r, c] = 1.0
    return grid


def bbox_to_patch_mask(
    bbox: List[float],
    image_grid_thw: torch.LongTensor,
    spatial_merge_size: int = 2,
) -> torch.Tensor:
    """Convert a normalized box to a soft overlap-area mask over merged patches.

    Each merged-patch entry is the fractional area of the bbox covered by that
    patch, normalized so the result sums to 1. Tiny boxes that miss every patch
    boundary fall back to a single delta on the patch containing the bbox
    center.

    Args:
        bbox: ``[x1, y1, x2, y2]`` in ``[0, 1]`` image coordinates.
        image_grid_thw: Length-3 tensor ``(T, H, W)`` describing native patch grid.
        spatial_merge_size: Integer merge factor along height/width.

    Returns:
        Float vector of length ``T * (H//m) * (W//m)`` aligned with Qwen tokens.
    """
    t, h_patches, w_patches = image_grid_thw.tolist()
    h_merged = h_patches // spatial_merge_size
    w_merged = w_patches // spatial_merge_size

    grid = _overlap_area_grid(bbox, h_merged, w_merged)    # (H, W) sums to 1
    if t == 1:
        return grid.flatten()
    # Replicate across temporal frames, then renormalize so the full vector sums to 1.
    return (grid.flatten().unsqueeze(0).expand(t, -1).contiguous() / float(t)).flatten()


def build_chat_messages(
    question: str,
    answer: Optional[str] = None,
    image_path: Optional[str] = None,
    image_url: Optional[str] = None,
) -> List[Dict]:
    """Construct Qwen ``apply_chat_template`` message list with optional vision.

    Exactly one of ``image_path`` or ``image_url`` may be set to prepend an image
    content block; when ``answer`` is provided, an assistant turn is appended for
    supervised fine-tuning prompts.

    Args:
        question: User question text.
        answer: Optional assistant reply for SFT formatting.
        image_path: Local filesystem path for the image modality.
        image_url: Remote URL for the image modality.

    Returns:
        Messages in HF multimodal chat schema (list of role/content dicts).
    """
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
