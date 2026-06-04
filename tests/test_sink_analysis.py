"""CPU unit tests for src/utils/sink_analysis.py.

Synthetic attention + hidden-state tensors with planted sinks / massive
activations verify every pure function. No GPU or model weights required; the
model-touching ``capture_sink_example`` is exercised only on a GPU node.
"""

import numpy as np
import pytest
import torch

from src.utils.sink_analysis import (
    ALL_KINDS,
    KIND_BOS,
    KIND_IMAGE,
    KIND_R,
    KIND_TEXT,
    answer_query_rows,
    bos_received_per_head,
    budget_per_head,
    build_kind_masks,
    massive_activation_stats,
    per_kind_norm_stats,
    received_per_key_headmean,
    row_entropy_per_head,
    sink_rate,
    spearman_rho,
    token_l2_norms,
    uniform_causal_received,
    uniform_col0_expectation,
)


def _causal_softmax(scores: np.ndarray) -> np.ndarray:
    """[H, L, L] raw scores -> causal post-softmax rows (sum 1 over k<=q)."""
    H, L, _ = scores.shape
    out = np.zeros_like(scores, dtype=np.float64)
    for h in range(H):
        for q in range(L):
            row = scores[h, q, : q + 1].astype(np.float64)
            row = np.exp(row - row.max())
            out[h, q, : q + 1] = row / row.sum()
    return out


# --------------------------------------------------------------------------- #
# Kind masks                                                                   #
# --------------------------------------------------------------------------- #

def test_build_kind_masks_frozen_disjoint_and_exhaustive():
    # [bos, img, img, text, text]
    seq = torch.tensor([1, 999, 999, 50, 51])
    masks = build_kind_masks(seq, image_token_id=999)
    np.testing.assert_array_equal(masks[KIND_BOS], [True, False, False, False, False])
    np.testing.assert_array_equal(masks[KIND_IMAGE], [False, True, True, False, False])
    np.testing.assert_array_equal(masks[KIND_TEXT], [False, False, False, True, True])
    np.testing.assert_array_equal(masks[KIND_R], [False] * 5)
    # disjoint + exhaustive
    stacked = np.stack([masks[k] for k in ALL_KINDS], axis=0)
    np.testing.assert_array_equal(stacked.sum(axis=0), np.ones(5, dtype=int))


def test_build_kind_masks_ri_r_block_priority():
    # spliced: [bos, img, img, R, R, text]; R sentinel = -1
    seq = torch.tensor([1, 999, 999, -1, -1, 51])
    masks = build_kind_masks(seq, image_token_id=999, insert_pos=3, n_queries=2)
    np.testing.assert_array_equal(masks[KIND_R], [False, False, False, True, True, False])
    np.testing.assert_array_equal(masks[KIND_IMAGE], [False, True, True, False, False, False])
    np.testing.assert_array_equal(masks[KIND_TEXT], [False, False, False, False, False, True])
    stacked = np.stack([masks[k] for k in ALL_KINDS], axis=0)
    np.testing.assert_array_equal(stacked.sum(axis=0), np.ones(6, dtype=int))


def test_answer_query_rows():
    np.testing.assert_array_equal(answer_query_rows(10, 3), [7, 8, 9])
    np.testing.assert_array_equal(answer_query_rows(10, 0), [9])  # fallback to last row
    np.testing.assert_array_equal(answer_query_rows(2, 5), [0, 1])  # clamped to start


# --------------------------------------------------------------------------- #
# BOS-sink statistics                                                          #
# --------------------------------------------------------------------------- #

def test_bos_received_planted_sink():
    L, H = 6, 2
    # Head 0: every query q>=2 puts ~all mass on key 0 (a sink). Head 1: uniform.
    scores = np.zeros((H, L, L), dtype=np.float64)
    scores[0, :, 0] = 20.0  # huge logit on key 0 -> softmax ~1 there
    A = _causal_softmax(scores)
    a0 = bos_received_per_head(A, q_min=2)
    assert a0[0] > 0.99                 # strong sink head
    # uniform head: a query at q puts 1/(q+1) on key 0; mean over q=2..5
    expected_uniform = np.mean([1 / 3, 1 / 4, 1 / 5, 1 / 6])
    assert abs(a0[1] - expected_uniform) < 1e-9


def test_uniform_col0_expectation_matches_mean():
    L = 6
    expected = np.mean([1 / (q + 1) for q in range(2, L)])
    assert abs(uniform_col0_expectation(L, q_min=2) - expected) < 1e-12


def test_sink_rate_threshold():
    a0 = np.array([[0.9, 0.1], [0.5, 0.05]])
    assert sink_rate(a0, eps=0.3) == 0.5      # 2 of 4 >= 0.3
    assert sink_rate(a0, eps=0.6) == 0.25     # 1 of 4 >= 0.6


# --------------------------------------------------------------------------- #
# Budget decomposition                                                         #
# --------------------------------------------------------------------------- #

def test_budget_per_head_sums_to_one_and_splits():
    # seq: [bos, img, img, text, text]; answer row = last (q=4) sees all keys.
    seq = torch.tensor([1, 999, 999, 50, 51])
    masks = build_kind_masks(seq, image_token_id=999)
    L, H = 5, 1
    scores = np.zeros((H, L, L), dtype=np.float64)
    # Row 4 mass: bos 0.4, two image cols 0.2 each, two text cols 0.1 each.
    # Use logits via log so softmax reproduces them.
    target = np.array([0.4, 0.2, 0.2, 0.1, 0.1])
    scores[0, 4, :5] = np.log(target)
    A = _causal_softmax(scores)
    bud = budget_per_head(A, [4], masks)
    assert abs(bud[KIND_BOS][0] - 0.4) < 1e-6
    assert abs(bud[KIND_IMAGE][0] - 0.4) < 1e-6   # 0.2 + 0.2
    assert abs(bud[KIND_TEXT][0] - 0.2) < 1e-6    # 0.1 + 0.1
    assert abs(bud[KIND_R][0] - 0.0) < 1e-12
    assert abs(bud["_row_sum"][0] - 1.0) < 1e-6
    # kind masses sum to the row total
    total = bud[KIND_BOS][0] + bud[KIND_IMAGE][0] + bud[KIND_TEXT][0] + bud[KIND_R][0]
    assert abs(total - 1.0) < 1e-6


def test_budget_per_head_extra_gen_cols():
    # K (key length) > L (mask length): decode step with one generated column.
    seq = torch.tensor([1, 999, 50])  # L=3 masks
    masks = build_kind_masks(seq, image_token_id=999)
    H, Lq, K = 1, 1, 4  # one query row, 4 keys (3 prefill + 1 generated)
    A = np.zeros((H, Lq, K), dtype=np.float64)
    A[0, 0, :] = [0.25, 0.25, 0.25, 0.25]  # last col is the generated token
    extra = np.array([False, False, False, True])
    bud = budget_per_head(A, [0], masks, extra_cols=extra)
    assert abs(bud["gen"][0] - 0.25) < 1e-9
    assert abs(bud[KIND_BOS][0] - 0.25) < 1e-9
    assert abs(bud[KIND_IMAGE][0] - 0.25) < 1e-9
    assert abs(bud[KIND_TEXT][0] - 0.25) < 1e-9


# --------------------------------------------------------------------------- #
# Entropy / received                                                           #
# --------------------------------------------------------------------------- #

def test_row_entropy_uniform_vs_peaked():
    L, H = 5, 1
    # Uniform head: Hnorm == 1, ESS == q+1.
    uni = _causal_softmax(np.zeros((H, L, L)))
    hnorm, ess = row_entropy_per_head(uni, [4])
    assert abs(hnorm[0] - 1.0) < 1e-9
    assert abs(ess[0] - 5.0) < 1e-6
    # Peaked head: all mass on key 0 -> Hnorm ~ 0, ESS ~ 1.
    sc = np.zeros((H, L, L)); sc[0, 4, 0] = 50.0
    pk = _causal_softmax(sc)
    hnorm2, ess2 = row_entropy_per_head(pk, [4])
    assert hnorm2[0] < 1e-3
    assert abs(ess2[0] - 1.0) < 1e-2


def test_received_per_key_vs_uniform_null():
    # Uniform causal attention: recv[k] should equal C_unif[k] exactly.
    L, H = 6, 3
    uni = _causal_softmax(np.zeros((H, L, L)))
    recv = received_per_key_headmean(uni)
    cu = uniform_causal_received(L)
    np.testing.assert_allclose(recv, cu, atol=1e-9)
    # A planted BOS sink lifts recv[0]/C_unif[0] well above 1.
    sc = np.zeros((H, L, L)); sc[:, :, 0] = 20.0
    A = _causal_softmax(sc)
    recv2 = received_per_key_headmean(A)
    assert recv2[0] / cu[0] > 2.0


# --------------------------------------------------------------------------- #
# Hidden-state norms / massive activations                                     #
# --------------------------------------------------------------------------- #

def test_token_l2_norms_and_kind_stats():
    # 4 tokens x 3 dims; norms = [sqrt(3), 0, 2, 4]
    hs = np.array([[1, 1, 1], [0, 0, 0], [2, 0, 0], [0, 4, 0]], dtype=np.float64)
    n2 = token_l2_norms(hs)
    np.testing.assert_allclose(n2, [np.sqrt(3), 0.0, 2.0, 4.0], atol=1e-9)
    masks = {KIND_BOS: np.array([True, False, False, False]),
             KIND_IMAGE: np.array([False, True, True, False]),
             KIND_TEXT: np.array([False, False, False, True]),
             KIND_R: np.array([False, False, False, False])}
    stats = per_kind_norm_stats(n2, masks)
    assert abs(stats[KIND_BOS]["median"] - np.sqrt(3)) < 1e-9
    assert stats[KIND_BOS]["n"] == 1
    assert abs(stats[KIND_IMAGE]["median"] - 1.0) < 1e-9   # median of {0, 2}
    assert stats[KIND_R]["n"] == 0
    assert np.isnan(stats[KIND_R]["median"])


def test_per_kind_norm_stats_mismatched_mask_lengths():
    """Both longer AND shorter masks must be handled by common-prefix truncation
    (the shorter-mask case used to raise IndexError)."""
    n2 = np.array([1.0, 2.0, 3.0, 4.0])
    longer = {KIND_TEXT: np.array([True, True, False, True, True, False])}
    stats = per_kind_norm_stats(n2, longer)
    assert stats[KIND_TEXT]["n"] == 3  # positions 0,1,3 of the common prefix
    shorter = {KIND_TEXT: np.array([True, False])}
    stats2 = per_kind_norm_stats(n2, shorter)  # must not raise
    assert stats2[KIND_TEXT]["n"] == 1
    assert stats2[KIND_TEXT]["median"] == 1.0


def test_budget_per_head_extra_cols_wrong_length_raises():
    seq = torch.tensor([1, 999, 50])
    masks = build_kind_masks(seq, image_token_id=999)
    A = np.full((1, 1, 4), 0.25)
    with pytest.raises(ValueError):
        budget_per_head(A, [0], masks, extra_cols=np.array([False, True]))  # len 2 != K 4


def test_massive_activation_flag():
    # Token 0: one feature of 5000, rest ~0.1 -> linf 5000, median ~0.1, ratio huge -> flagged.
    # Token 1: broadband (all ~3) -> linf 3, ratio ~1 -> not flagged.
    d = 64
    hs = np.full((2, d), 0.1, dtype=np.float64)
    hs[0, 7] = 5000.0
    hs[1, :] = 3.0
    ma = massive_activation_stats(hs, mag_thresh=100.0, ratio_thresh=1000.0)
    assert ma["flag"][0] and not ma["flag"][1]
    assert ma["argmax_dim"][0] == 7
    assert ma["linf"][0] == 5000.0


def test_spearman_monotonic_and_ties():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert abs(spearman_rho(x, 2 * x + 1) - 1.0) < 1e-9       # perfectly monotone
    assert abs(spearman_rho(x, -x) + 1.0) < 1e-9              # perfectly anti
    # With ties the average-rank path must still run and stay in [-1, 1].
    rho = spearman_rho(np.array([1.0, 1.0, 2.0, 3.0]), np.array([1.0, 2.0, 2.0, 3.0]))
    assert -1.0 <= rho <= 1.0
    assert np.isnan(spearman_rho(np.array([1.0, 2.0]), np.array([1.0, 2.0])))  # n<3
