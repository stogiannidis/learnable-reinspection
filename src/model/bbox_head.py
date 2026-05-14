"""Auxiliary bounding-box head and GIoU-based grounding loss for stage 1.

The head pools bottleneck query states and predicts a normalized axis-aligned
box; the composite loss combines L1 coordinate error with generalized IoU.
"""

from typing import Optional

import torch
import torch.nn as nn


class BboxHead(nn.Module):
    """Three-layer MLP mapping mean-pooled bottleneck states to ``[x1,y1,x2,y2]``."""

    def __init__(self, d_r: int, dtype: Optional[torch.dtype] = None):
        """Create the MLP and Sigmoid output squashing coordinates to ``[0, 1]``.

        Args:
            d_r: Bottleneck dimension ``R_r`` (matches re-inspection width).
            dtype: Optional module dtype for mixed precision.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_r, d_r, dtype=dtype),
            nn.GELU(),
            nn.Linear(d_r, d_r, dtype=dtype),
            nn.GELU(),
            nn.Linear(d_r, 4, dtype=dtype),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        """Xavier-initialize linear layers with zero bias."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, R_r: torch.Tensor) -> torch.Tensor:
        """Predict a single normalized box per batch item from query tokens.

        Args:
            R_r: Tensor of shape ``(B, K, d_r)`` — bottleneck query states for
                the supervised tokens. Stage 1 passes only the selector slice
                ``R_bottleneck[:, :n_selector_queries]``; mean-pooling reduces
                across the K selectors before the MLP.

        Returns:
            Tensor of shape ``(B, 4)`` with values in ``[0, 1]`` interpreted as
            ``[x1, y1, x2, y2]`` in normalized image coordinates.
        """
        w_dtype = next(self.parameters()).dtype
        r_bar = R_r.to(dtype=w_dtype).mean(dim=1)
        return self.mlp(r_bar)


def _giou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean ``1 - GIoU`` for axis-aligned boxes in normalized coordinates.

    Enforces valid ordering by sorting each predicted corner pair before IoU.
    """
    px1, py1, px2, py2 = pred.unbind(-1)
    gx1, gy1, gx2, gy2 = target.unbind(-1)

    px1, px2 = torch.min(px1, px2), torch.max(px1, px2)
    py1, py2 = torch.min(py1, py2), torch.max(py1, py2)

    inter_x1 = torch.max(px1, gx1)
    inter_y1 = torch.max(py1, gy1)
    inter_x2 = torch.min(px2, gx2)
    inter_y2 = torch.min(py2, gy2)
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    area_pred = (px2 - px1) * (py2 - py1)
    area_gt = (gx2 - gx1) * (gy2 - gy1)
    union = area_pred + area_gt - inter_area + 1e-7

    iou = inter_area / union

    enc_x1 = torch.min(px1, gx1)
    enc_y1 = torch.min(py1, gy1)
    enc_x2 = torch.max(px2, gx2)
    enc_y2 = torch.max(py2, gy2)
    enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1) + 1e-7

    giou = iou - (enc_area - union) / enc_area
    return (1.0 - giou).mean()


def compute_grounding_loss(
    bbox_pred: torch.Tensor,
    bbox_gt: torch.Tensor,
    l1_weight: float = 5.0,
    giou_weight: float = 2.0,
) -> torch.Tensor:
    """Weighted sum of L1 and GIoU between predicted and ground-truth boxes.

    Args:
        bbox_pred: Predicted boxes ``(B, 4)``.
        bbox_gt: Ground-truth boxes ``(B, 4)``, same layout as predictions.
        l1_weight: Multiplier on mean L1 coordinate error.
        giou_weight: Multiplier on :func:`_giou_loss`.

    Returns:
        Scalar combined loss.
    """
    pred = bbox_pred.float()
    gt = bbox_gt.float()
    l1 = nn.functional.l1_loss(pred, gt)
    giou = _giou_loss(pred, gt)
    return l1_weight * l1 + giou_weight * giou
