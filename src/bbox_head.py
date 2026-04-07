"""Auxiliary bounding-box regression head for Stage 1 grounding supervision."""

from typing import Optional

import torch
import torch.nn as nn


class BboxHead(nn.Module):
    """3-layer MLP that predicts a normalized bounding box from pooled R_r."""

    def __init__(self, d_r: int, dtype: Optional[torch.dtype] = None):
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
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, R_r: torch.Tensor) -> torch.Tensor:
        """
        Args:
            R_r: (B, N_q, d_r) bottleneck-space query representations.
        Returns:
            bbox_pred: (B, 4) predicted normalized [x1, y1, x2, y2].
        """
        r_bar = R_r.float().mean(dim=1)
        return self.mlp(r_bar)


def _giou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Generalized IoU loss for [x1, y1, x2, y2] boxes in [0, 1]."""
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
    """L1 + GIoU grounding loss on normalized [x1, y1, x2, y2] boxes."""
    pred = bbox_pred.float()
    gt = bbox_gt.float()
    l1 = nn.functional.l1_loss(pred, gt)
    giou = _giou_loss(pred, gt)
    return l1_weight * l1 + giou_weight * giou
