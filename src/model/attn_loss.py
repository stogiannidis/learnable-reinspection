"""Stage-1 attention supervision losses for aligning query attention to targets.

Provides KL-based matching (InternVL-style) and focal loss with optional
per-query spatial diversification for Qwen patch grids.
"""

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


def _diversify_targets(
    binary_target: torch.Tensor,
    n_queries: int,
    image_grid_thw: torch.Tensor,
    spatial_merge_size: int = 2,
) -> torch.Tensor:
    """Partition positive patch rows into disjoint vertical strips per query.

    Encourages different bottleneck queries to cover different spatial bands
    derived from the ground-truth bounding box on the merged Qwen patch grid.

    Args:
        binary_target: ``(B, N_v)`` binary mask of supervised vision tokens.
        n_queries: Number of re-inspection queries (strip count upper bound).
        image_grid_thw: ``(B, 3)`` tensor ``(T, H, W)`` describing patch layout.
        spatial_merge_size: Patch merge factor used by the vision tower.

    Returns:
        Tensor ``(B, n_queries, N_v)`` with per-query targets (zeros where no split).
    """
    B, N_v = binary_target.shape
    per_query = binary_target.unsqueeze(1).expand(B, n_queries, N_v).clone()

    for b in range(B):
        t = image_grid_thw[b, 0].item()
        h_merged = image_grid_thw[b, 1].item() // spatial_merge_size
        w_merged = image_grid_thw[b, 2].item() // spatial_merge_size
        n_patches_expected = t * h_merged * w_merged

        if n_patches_expected != N_v or h_merged == 0:
            continue

        row_idx = torch.arange(N_v, device=binary_target.device) // w_merged % h_merged
        positive_mask = binary_target[b] > 0
        if positive_mask.sum() == 0:
            continue

        pos_rows = row_idx[positive_mask]
        row_min, row_max = pos_rows.min().item(), pos_rows.max().item() + 1
        n_rows = row_max - row_min

        if n_rows <= 1:
            continue

        for q in range(n_queries):
            strip_start = row_min + (q * n_rows) // n_queries
            strip_end = row_min + ((q + 1) * n_rows) // n_queries
            strip_mask = (row_idx >= strip_start) & (row_idx < strip_end)
            query_target = binary_target[b] * strip_mask.float()
            if query_target.sum() > 0:
                per_query[b, q] = query_target

    return per_query


def compute_attn_loss_focal(
    A_vis: torch.Tensor,
    attn_target: torch.Tensor,
    image_grid_thw: torch.Tensor = None,
    n_queries: int = 32,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Binary focal loss between predicted vision attention and soft targets.

    When ``image_grid_thw`` is provided, targets are diversified per query via
    :func:`_diversify_targets`; otherwise the same binary mask is broadcast to
    all queries.  Probabilities are computed in float32 to avoid ``log(0)``
    under bfloat16 rounding.

    Args:
        A_vis: Predicted attention logits or probabilities ``(B, Q, V)``.
        attn_target: Non-negative supervision weights ``(B, V)``.
        image_grid_thw: Optional Qwen grid tensor for diversification.
        n_queries: Query count (must match ``A_vis``'s query dimension).
        alpha: Focal balancing factor ``alpha`` in ``[0, 1]``.
        gamma: Focusing exponent on ``(1 - p_t)``.

    Returns:
        Scalar mean focal loss over all entries.
    """
    binary = (attn_target > 0).float()

    if image_grid_thw is not None:
        target = _diversify_targets(binary, n_queries, image_grid_thw)
    else:
        target = binary.unsqueeze(1).expand_as(A_vis)

    # Compute the focal loss in FP32. In bf16, `1.0 - 1e-6` rounds back to 1.0,
    # which can leave exact 1s after clamping and trigger `0 * log(0) -> NaN`.
    pred = (A_vis.float() * A_vis.shape[-1]).clamp(min=1e-6, max=1.0 - 1e-6)
    target = target.float()
    bce = -(target * pred.log() + (1 - target) * (1 - pred).log())
    p_t = target * pred + (1 - target) * (1 - pred)
    focal_weight = (1 - p_t) ** gamma
    alpha_t = target * alpha + (1 - target) * (1 - alpha)
    loss = alpha_t * focal_weight * bce
    return loss.mean()
