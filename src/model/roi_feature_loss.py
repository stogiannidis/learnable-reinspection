"""ROI feature grounding loss for Stage 1.

Pulls the pooled content tokens of the re-inspection module toward the
overlap-area-weighted mean of the frozen vision patch features inside the
ground-truth bbox. The target is detached so gradients only flow through the
content tokens and a small learnable projection.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ROIProjection(nn.Module):
    """Learnable projection from bottleneck (``d_r``) to vision width (``d_model``)."""

    def __init__(self, d_bottleneck: int, d_model: int, dtype: torch.dtype = None):
        super().__init__()
        self.proj = nn.Linear(d_bottleneck, d_model, bias=False, dtype=dtype)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def compute_roi_feature_loss(
    V_frozen: torch.Tensor,
    R_content: torch.Tensor,
    attn_target_mask: torch.Tensor,
    projection: ROIProjection,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cosine loss between projected pooled content tokens and ROI-pooled features.

    Args:
        V_frozen: Frozen vision hidden states ``(B, S_v, d_model)``.
        R_content: Content-role re-inspection queries ``(B, K_content, d_r)``.
        attn_target_mask: Per-patch overlap-area weights summing to ~1
            ``(B, S_v)``; same tensor used for the attention KL loss.
        projection: Learnable ``d_r -> d_model`` projection module.

    Returns:
        ``(loss, cos_sim_mean)`` where ``loss = mean(1 - cos)`` and
        ``cos_sim_mean`` is reported for monitoring.
    """
    if attn_target_mask.dim() != 2:
        raise ValueError(
            f"attn_target_mask must be (B, S_v); got shape {tuple(attn_target_mask.shape)}"
        )

    target_dtype = V_frozen.dtype
    mask = attn_target_mask.to(device=V_frozen.device, dtype=target_dtype)
    # Renormalize per-sample so each ROI target is a proper weighted mean even
    # when the data-side mask was tiled across temporal frames or crops.
    norm = mask.sum(dim=1, keepdim=True).clamp(min=eps)
    weights = mask / norm                                          # (B, S_v)

    z_roi = (weights.unsqueeze(-1) * V_frozen).sum(dim=1)          # (B, d_model)
    z_roi = z_roi.detach()

    pooled = R_content.mean(dim=1)                                 # (B, d_r)
    pooled = pooled.to(projection.proj.weight.dtype)
    z_r = projection(pooled).to(target_dtype)                      # (B, d_model)

    cos = F.cosine_similarity(z_r, z_roi, dim=-1, eps=eps)         # (B,)
    loss = (1.0 - cos).mean()
    return loss, cos.detach().mean()
