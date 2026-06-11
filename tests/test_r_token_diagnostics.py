import math

import torch

from src.utils.r_token_diagnostics import (
    attention_mass_to_key_mask,
    bbox_metrics,
    fit_ridge_probe,
    r_token_mask,
    shifted_answer_mask,
)


def test_bbox_metrics_perfect_prediction():
    boxes = torch.tensor([[0.1, 0.2, 0.7, 0.8], [0.8, 0.6, 0.2, 0.1]])
    metrics = bbox_metrics(boxes, boxes)

    assert metrics["n"] == 2
    assert metrics["l1"] == 0.0
    assert math.isclose(metrics["iou_mean"], 1.0, rel_tol=1e-6)
    assert metrics["iou_at_0_5"] == 1.0


def test_ridge_probe_recovers_simple_linear_map():
    x = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ]
    )
    y = torch.stack([0.1 + 0.2 * x[:, 0], 0.2 + 0.1 * x[:, 1], 0.6 + 0.1 * x[:, 0], 0.7 + 0.1 * x[:, 1]], dim=1)

    probe = fit_ridge_probe(x, y, l2=1e-8)
    pred = probe.predict(x, clamp=False)

    assert torch.allclose(pred, y, atol=1e-5)


def test_shifted_answer_and_r_masks():
    labels = torch.tensor([[-100, -100, 11, 12, -100]])
    insert_positions = torch.tensor([2])

    answer = shifted_answer_mask(labels, insert_positions, n_queries=3, ignore_index=-100)
    r_mask = r_token_mask(batch_size=1, seq_len=5, insert_positions=insert_positions, n_queries=3)

    assert answer.tolist() == [[False, False, False, False, False, True, True, False]]
    assert r_mask.tolist() == [[False, False, True, True, True, False, False, False]]


def test_attention_mass_to_key_mask():
    # One layer, one batch, one head, length 4. Query rows 2 and 3 attend 0.3 and
    # 0.4 mass respectively to key 1, so the mean selected mass is 0.35.
    attn = torch.zeros(1, 1, 4, 4)
    attn[0, 0, 2] = torch.tensor([0.1, 0.3, 0.6, 0.0])
    attn[0, 0, 3] = torch.tensor([0.2, 0.4, 0.1, 0.3])
    q_mask = torch.tensor([[False, False, True, True]])
    k_mask = torch.tensor([[False, True, False, False]])

    masses = attention_mass_to_key_mask((attn,), q_mask, k_mask)

    assert len(masses) == 1
    assert math.isclose(masses[0], 0.35, rel_tol=1e-6)
