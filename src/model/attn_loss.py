"""Stage-1 attention supervision: KL alignment of query attention to targets."""

from typing import Optional

import torch
import torch.nn.functional as F


def compute_attn_loss_kl(
    attn_vis: torch.Tensor,
    attn_target: torch.Tensor,
    bbox_areas: Optional[torch.Tensor] = None,
    small_box_weight: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """KL divergence between predicted and target attention over vision tokens.

    Normalizes both distributions over the vision-token axis (the caller is
    expected to pass selector-only attention rows, so non-image columns should
    not be present). Optionally upweights small-bbox samples to compensate for
    the sparser supervision signal, with weight
    ``w = clip((area + eps)^{-1/2}, 1, 4)``.

    Args:
        attn_vis: Predicted attention ``(B, Q, V)`` for the supervised queries.
        attn_target: Soft target mask ``(B, V)`` (sums need not be 1 before norm).
        bbox_areas: Per-sample normalized bbox area ``(B,)`` in ``[0, 1]``;
            required when ``small_box_weight=True``.
        small_box_weight: If True, scale per-sample loss by the small-box weight.

    Returns:
        Scalar KL loss averaged over the batch and queries.
    """
    pred = attn_vis.float().clamp(min=eps)
    pred = pred / pred.sum(dim=-1, keepdim=True)                  # (B, Q, V)
    target = attn_target.float().clamp(min=eps)
    target = target / target.sum(dim=-1, keepdim=True)            # (B, V)
    target_b = target.unsqueeze(1).expand_as(pred)                # (B, Q, V)

    # Per-sample KL averaged over selectors: KL(target || pred).
    per_query = (target_b * (target_b.log() - pred.log())).sum(dim=-1)  # (B, Q)
    per_sample = per_query.mean(dim=-1)                           # (B,)

    if small_box_weight:
        if bbox_areas is None:
            raise ValueError("small_box_weight=True requires bbox_areas")
        w = (bbox_areas.float().clamp(min=eps) + eps).rsqrt().clamp(min=1.0, max=4.0)
        per_sample = per_sample * w

    return per_sample.mean()
