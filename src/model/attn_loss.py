"""Stage-1 attention supervision: KL alignment of query attention to targets."""

import torch
import torch.nn.functional as F


def compute_attn_loss_kl(attn_vis: torch.Tensor, attn_target: torch.Tensor) -> torch.Tensor:
    """KL divergence between predicted and target attention over vision tokens.

    Normalizes both distributions per query position, expands the target across
    heads, and averages ``batchmean`` KL.  The loss is scaled by the number of
    queries so magnitude stays stable as ``n_queries`` changes.

    Args:
        attn_vis: Predicted attention ``(B, Q, V)`` from the re-inspection module.
        attn_target: Soft target mask ``(B, V)`` (sums need not be 1 before norm).

    Returns:
        Scalar KL loss averaged over the batch and queries.
    """
    pred = attn_vis.float().clamp(min=1e-8)
    pred = pred / pred.sum(dim=-1, keepdim=True)
    target = attn_target.float().clamp(min=1e-8)
    target = target / target.sum(dim=-1, keepdim=True)
    target = target.unsqueeze(1).expand_as(pred)
    loss = F.kl_div(pred.log(), target, reduction="batchmean")
    return loss / pred.shape[1]
