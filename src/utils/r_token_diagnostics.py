"""Small tensor utilities for R-token geometry and grounding diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F


def canonicalize_boxes(boxes: torch.Tensor) -> torch.Tensor:
    """Clamp boxes to ``[0, 1]`` and ensure ``x1 <= x2`` / ``y1 <= y2``."""
    boxes = boxes.float().clamp(0.0, 1.0)
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack(
        [torch.minimum(x1, x2), torch.minimum(y1, y2), torch.maximum(x1, x2), torch.maximum(y1, y2)],
        dim=-1,
    )


def box_iou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Pairwise aligned IoU for ``(N, 4)`` normalized boxes."""
    pred = canonicalize_boxes(pred)
    target = canonicalize_boxes(target)
    px1, py1, px2, py2 = pred.unbind(-1)
    gx1, gy1, gx2, gy2 = target.unbind(-1)

    ix1 = torch.maximum(px1, gx1)
    iy1 = torch.maximum(py1, gy1)
    ix2 = torch.minimum(px2, gx2)
    iy2 = torch.minimum(py2, gy2)
    inter = (ix2 - ix1).clamp(min=0.0) * (iy2 - iy1).clamp(min=0.0)

    pred_area = (px2 - px1).clamp(min=0.0) * (py2 - py1).clamp(min=0.0)
    target_area = (gx2 - gx1).clamp(min=0.0) * (gy2 - gy1).clamp(min=0.0)
    return inter / (pred_area + target_area - inter + eps)


def bbox_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Return scalar localization metrics for aligned predicted/target boxes."""
    pred = canonicalize_boxes(pred.detach().float())
    target = canonicalize_boxes(target.detach().float())
    iou = box_iou(pred, target)
    center_pred = torch.stack([(pred[:, 0] + pred[:, 2]) * 0.5, (pred[:, 1] + pred[:, 3]) * 0.5], dim=-1)
    center_target = torch.stack(
        [(target[:, 0] + target[:, 2]) * 0.5, (target[:, 1] + target[:, 3]) * 0.5],
        dim=-1,
    )
    return {
        "n": int(pred.shape[0]),
        "l1": float((pred - target).abs().mean().item()),
        "center_l1": float((center_pred - center_target).abs().mean().item()),
        "iou_mean": float(iou.mean().item()),
        "iou_median": float(iou.median().item()),
        "iou_at_0_3": float((iou >= 0.3).float().mean().item()),
        "iou_at_0_5": float((iou >= 0.5).float().mean().item()),
    }


@dataclass
class LinearProbe:
    """A fitted linear probe ``x @ weight + bias``."""

    weight: torch.Tensor
    bias: torch.Tensor
    x_mean: torch.Tensor
    y_mean: torch.Tensor
    l2: float

    def predict(self, x: torch.Tensor, clamp: bool = True) -> torch.Tensor:
        y = x.float() @ self.weight + self.bias
        return y.clamp(0.0, 1.0) if clamp else y


def fit_ridge_probe(x: torch.Tensor, y: torch.Tensor, l2: float = 1e-3) -> LinearProbe:
    """Fit a multi-output ridge probe with an unregularized intercept.

    Uses the dual solve when samples are fewer than feature dimensions, which is
    the common case for LM-space probes.
    """
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be rank-2 tensors")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"x/y sample mismatch: {x.shape[0]} vs {y.shape[0]}")
    if x.shape[0] < 2:
        raise ValueError("at least two samples are required to fit a probe")

    x = x.float()
    y = y.float()
    x_mean = x.mean(dim=0, keepdim=True)
    y_mean = y.mean(dim=0, keepdim=True)
    xc = x - x_mean
    yc = y - y_mean
    n, d = xc.shape
    ridge = max(float(l2), 0.0)

    if n <= d:
        gram = xc @ xc.T
        gram = gram + ridge * torch.eye(n, device=gram.device, dtype=gram.dtype)
        try:
            alpha = torch.linalg.solve(gram, yc)
        except torch.linalg.LinAlgError:
            alpha = torch.linalg.pinv(gram) @ yc
        weight = xc.T @ alpha
    else:
        gram = xc.T @ xc
        gram = gram + ridge * torch.eye(d, device=gram.device, dtype=gram.dtype)
        try:
            weight = torch.linalg.solve(gram, xc.T @ yc)
        except torch.linalg.LinAlgError:
            weight = torch.linalg.pinv(gram) @ (xc.T @ yc)

    bias = y_mean.squeeze(0) - x_mean.squeeze(0) @ weight
    return LinearProbe(
        weight=weight.detach().cpu(),
        bias=bias.detach().cpu(),
        x_mean=x_mean.squeeze(0).detach().cpu(),
        y_mean=y_mean.squeeze(0).detach().cpu(),
        l2=ridge,
    )


def summarize(values: Iterable[float]) -> Dict[str, float]:
    """Mean/std/quantile summary for a numeric iterable."""
    vals = torch.as_tensor(list(values), dtype=torch.float32)
    if vals.numel() == 0:
        return {"n": 0}
    return {
        "n": int(vals.numel()),
        "mean": float(vals.mean().item()),
        "std": float(vals.std(unbiased=False).item()),
        "min": float(vals.min().item()),
        "p10": float(torch.quantile(vals, 0.10).item()),
        "median": float(torch.quantile(vals, 0.50).item()),
        "p90": float(torch.quantile(vals, 0.90).item()),
        "max": float(vals.max().item()),
    }


def r_text_cosine_stats(
    r_tokens: torch.Tensor,
    text_tokens: torch.Tensor,
    text_mask: torch.BoolTensor,
) -> Dict[str, List[float]]:
    """Per-sample cosine stats between R tokens and normal text embeddings."""
    if r_tokens.ndim != 3 or text_tokens.ndim != 3:
        raise ValueError("r_tokens and text_tokens must be rank-3 tensors")
    out = {"mean": [], "max": []}
    r_norm = F.normalize(r_tokens.float(), dim=-1, eps=1e-8)
    t_norm = F.normalize(text_tokens.float(), dim=-1, eps=1e-8)
    for b in range(r_norm.shape[0]):
        valid = text_mask[b]
        if valid.sum().item() == 0:
            continue
        cos = r_norm[b] @ t_norm[b, valid].T
        out["mean"].append(float(cos.mean().item()))
        out["max"].append(float(cos.max(dim=-1).values.mean().item()))
    return out


def shifted_answer_mask(
    labels: torch.LongTensor,
    insert_positions: torch.LongTensor,
    n_queries: int,
    ignore_index: int = -100,
) -> torch.BoolTensor:
    """Build answer-token mask after inserting R tokens into a labeled sequence."""
    bsz, seq_len = labels.shape
    out = torch.zeros(bsz, seq_len + n_queries, dtype=torch.bool, device=labels.device)
    for b in range(bsz):
        p = int(insert_positions[b].item())
        ans = (labels[b] != ignore_index).nonzero(as_tuple=False).squeeze(-1)
        if ans.numel() == 0:
            continue
        shifted = ans + (ans >= p).long() * n_queries
        out[b, shifted] = True
    return out


def r_token_mask(
    batch_size: int,
    seq_len: int,
    insert_positions: torch.LongTensor,
    n_queries: int,
) -> torch.BoolTensor:
    """Build R-token key mask for sequences after R insertion."""
    out = torch.zeros(batch_size, seq_len + n_queries, dtype=torch.bool, device=insert_positions.device)
    for b in range(batch_size):
        p = int(insert_positions[b].item())
        out[b, p : p + n_queries] = True
    return out


def attention_mass_to_key_mask(
    attentions: Optional[Sequence[torch.Tensor]],
    query_mask: torch.BoolTensor,
    key_mask: torch.BoolTensor,
) -> List[float]:
    """Mean attention mass from selected query rows to selected key columns.

    Args:
        attentions: Forward-pass attention tuple, one tensor per layer with shape
            ``(B, H, L, L)``.
        query_mask: Boolean ``(B, L)`` mask for rows to average.
        key_mask: Boolean ``(B, L)`` mask for key columns to sum over.

    Returns:
        One scalar per layer. Missing layers or samples with no selected query/key
        rows are skipped.
    """
    if attentions is None:
        return []
    masses: List[float] = []
    for layer_attn in attentions:
        if layer_attn is None:
            continue
        attn = layer_attn.float()
        if attn.ndim == 3:
            attn = attn.unsqueeze(1)
        layer_vals = []
        for b in range(attn.shape[0]):
            q_idx = query_mask[b].nonzero(as_tuple=False).squeeze(-1)
            k_idx = key_mask[b].nonzero(as_tuple=False).squeeze(-1)
            if q_idx.numel() == 0 or k_idx.numel() == 0:
                continue
            selected = attn[b, :, q_idx][:, :, k_idx]
            layer_vals.append(selected.sum(dim=-1).mean())
        if layer_vals:
            masses.append(float(torch.stack(layer_vals).mean().item()))
    return masses
