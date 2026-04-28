"""Tests for optimizer-group gradient metrics logged to Weights & Biases."""

import math

import torch

from src.training.trainer import _collect_optimizer_group_grad_metrics


def test_collect_optimizer_group_grad_metrics_splits_named_groups():
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 4, bias=False),
        torch.nn.Linear(4, 2, bias=False),
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": list(model[0].parameters()), "lr": 1e-3, "name": "encoder"},
            {"params": list(model[1].parameters()), "lr": 2e-3, "name": "head"},
        ],
        weight_decay=0.01,
    )

    x = torch.randn(5, 3)
    loss = model(x).pow(2).mean()
    loss.backward()

    metrics = _collect_optimizer_group_grad_metrics(optimizer, "toy/train")

    assert metrics["toy/train/groups/encoder/lr"] == 1e-3
    assert metrics["toy/train/groups/head/lr"] == 2e-3
    assert metrics["toy/train/groups/encoder/grad_norm"] > 0.0
    assert metrics["toy/train/groups/head/grad_norm"] > 0.0
    assert metrics["toy/train/grad_norm"] > 0.0
    assert metrics["toy/train/param_norm"] > 0.0
    assert metrics["toy/train/grad_abs_mean"] > 0.0
    assert metrics["toy/train/grad_abs_max"] > 0.0
    assert metrics["toy/train/grad_nonzero_frac"] > 0.0
    assert metrics["toy/train/grad_coverage"] == 1.0

    expected_grad_norm = math.sqrt(
        metrics["toy/train/groups/encoder/grad_norm"] ** 2
        + metrics["toy/train/groups/head/grad_norm"] ** 2
    )
    expected_param_norm = math.sqrt(
        metrics["toy/train/groups/encoder/param_norm"] ** 2
        + metrics["toy/train/groups/head/param_norm"] ** 2
    )
    assert math.isclose(metrics["toy/train/grad_norm"], expected_grad_norm, rel_tol=1e-6)
    assert math.isclose(metrics["toy/train/param_norm"], expected_param_norm, rel_tol=1e-6)
