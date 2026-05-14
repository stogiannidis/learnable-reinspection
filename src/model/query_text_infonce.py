"""Symmetric query–text InfoNCE in the re-inspection bottleneck.

Pools learned queries (after the text cross-attention stage) and down-projected
text tokens, L2-normalizes, and applies a standard bidirectional in-batch
InfoNCE loss so each example's query aggregate aligns with its own text
sequence and is separated from other batch items' text.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_query_text_infonce_loss(
    Q_task: torch.Tensor,
    T_down: torch.Tensor,
    T_mask: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric InfoNCE between pooled queries and masked-mean text embeddings.

    Args:
        Q_task: Bottleneck query states after text path, shape ``(B, n_q, d_r)``.
        T_down: Down-projected text tokens, shape ``(B, S_t, d_r)``.
        T_mask: Boolean mask, shape ``(B, S_t)``, ``True`` for valid text positions.
        temperature: Softmax temperature (must be positive).

    Returns:
        Scalar mean of query-to-text and text-to-query cross-entropy losses.
    """
    if Q_task.dim() != 3 or T_down.dim() != 3:
        raise ValueError("Q_task and T_down must be 3D (B, seq, d_r)")
    B = Q_task.size(0)
    if T_down.shape[0] != B or T_mask.shape[0] != B:
        raise ValueError("Batch dimensions of Q_task, T_down, and T_mask must match")
    if B < 2:
        return (Q_task * 0.0).sum() + (T_down * 0.0).sum()

    # Aggregate queries: mean over learned query slots.
    q = Q_task.float().mean(dim=1)  # (B, d_r)
    # Masked mean over text tokens
    m = T_mask.to(dtype=T_down.dtype).unsqueeze(-1)
    denom = m.sum(dim=1).clamp(min=1.0)  # (B, 1)
    t = (T_down.float() * m).sum(dim=1) / denom  # (B, d_r)

    q = F.normalize(q, dim=-1, eps=1e-8)
    t = F.normalize(t, dim=-1, eps=1e-8)

    temp = max(float(temperature), 1e-8)
    logits = (q @ t.T) / temp  # (B, B)
    labels = torch.arange(B, device=logits.device, dtype=torch.long)
    loss_q2t = F.cross_entropy(logits, labels)
    loss_t2q = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_q2t + loss_t2q)
