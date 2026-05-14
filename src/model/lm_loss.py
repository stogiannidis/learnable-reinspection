"""Efficient language-model losses for short supervised answer spans."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_answer_cross_entropy(
    hidden_states: torch.Tensor,
    labels: torch.LongTensor,
    lm_head,
    ignore_index: int = -100,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Compute next-token CE only where labels are supervised.

    Stage-2 prompts contain many ignored image/prompt tokens and short answers.
    Projecting only supervised hidden states avoids a full-sequence vocabulary
    matmul while preserving the same shifted causal-LM objective.
    """
    shift_hidden = hidden_states[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    valid = shift_labels != ignore_index
    valid_indices = valid.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
    if valid_indices.numel() == 0:
        return hidden_states.sum() * 0.0

    flat_hidden = shift_hidden.reshape(-1, shift_hidden.shape[-1])
    flat_labels = shift_labels.reshape(-1)
    total_loss = torch.zeros((), device=hidden_states.device, dtype=torch.float32)

    for start in range(0, valid_indices.numel(), chunk_size):
        idx = valid_indices[start : start + chunk_size]
        logits = lm_head(flat_hidden.index_select(0, idx)).float()
        targets = flat_labels.index_select(0, idx)
        total_loss = total_loss + F.cross_entropy(logits, targets, reduction="sum")

    return total_loss / valid_indices.numel()
