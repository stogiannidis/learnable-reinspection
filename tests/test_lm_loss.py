"""Tests for efficient supervised-answer LM loss helpers."""

import torch
import torch.nn.functional as F

from src.model.lm_loss import masked_answer_cross_entropy


def test_masked_answer_cross_entropy_matches_full_shifted_ce():
    torch.manual_seed(0)
    hidden_states = torch.randn(2, 6, 4)
    lm_head = torch.nn.Linear(4, 7, bias=False)
    labels = torch.tensor(
        [
            [-100, -100, 1, 2, -100, 3],
            [-100, 4, -100, 5, 6, -100],
        ]
    )

    logits = lm_head(hidden_states)
    expected = F.cross_entropy(
        logits[..., :-1, :].reshape(-1, logits.shape[-1]),
        labels[..., 1:].reshape(-1),
        ignore_index=-100,
    )

    actual = masked_answer_cross_entropy(hidden_states, labels, lm_head, ignore_index=-100)

    assert torch.allclose(actual, expected)


def test_masked_answer_cross_entropy_handles_no_supervised_tokens():
    hidden_states = torch.randn(2, 4, 3, requires_grad=True)
    lm_head = torch.nn.Linear(3, 5)
    labels = torch.full((2, 4), -100)

    loss = masked_answer_cross_entropy(hidden_states, labels, lm_head, ignore_index=-100)
    loss.backward()

    assert loss.item() == 0.0
    assert hidden_states.grad is not None
