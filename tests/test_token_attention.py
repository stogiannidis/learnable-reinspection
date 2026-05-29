"""CPU unit tests for src/utils/token_attention.py.

These cover the full extraction math with synthetic attention tensors and a fake
tokenizer — no GPU or model weights required. The model-touching
``capture_token_attention`` is exercised end-to-end only on a GPU node; here we
verify every pure function it depends on.
"""

import importlib.util
import os

import numpy as np
import pytest
import torch

from src.utils.token_attention import (
    KIND_GENERATED,
    KIND_R,
    KIND_SPECIAL,
    KIND_TEXT,
    TokenMap,
    build_spliced_ids,
    classify_prefill_rows,
    grid_from_row,
    image_column_indices,
    mean_map,
    per_token_maps_from_decode,
    per_token_maps_from_prefill,
    reduce_step_attention,
    select_layer_indices,
    side_from_vision_cfg,
)


class FakeTokenizer:
    """Minimal tokenizer: decode ids to ``<id>`` strings, with special-id set."""

    def __init__(self, special_ids=()):
        self.all_special_ids = list(special_ids)

    def decode(self, ids, skip_special_tokens=False):
        if skip_special_tokens:
            ids = [i for i in ids if i not in self.all_special_ids]
        return "".join(f"<{i}>" for i in ids)


# --------------------------------------------------------------------------- #
# Layer selection                                                              #
# --------------------------------------------------------------------------- #

def test_select_layer_indices_modes():
    assert select_layer_indices(4, "mean") == [0, 1, 2, 3]
    assert select_layer_indices(4, "last") == [3]
    assert select_layer_indices(4, 0) == [0]
    assert select_layer_indices(4, -1) == [3]
    assert select_layer_indices(6, (2, 4)) == [2, 3]
    assert select_layer_indices(6, (-2, 6)) == [4, 5]


def test_select_layer_indices_errors():
    with pytest.raises(ValueError):
        select_layer_indices(4, 9)
    with pytest.raises(ValueError):
        select_layer_indices(4, "middle")
    with pytest.raises(ValueError):
        select_layer_indices(0, "mean")
    with pytest.raises(ValueError):
        select_layer_indices(4, True)  # bool guarded


# --------------------------------------------------------------------------- #
# Head/layer reduction                                                         #
# --------------------------------------------------------------------------- #

def test_reduce_step_attention_mean_over_heads_and_layers():
    torch.manual_seed(0)
    # 2 layers, batch=1, 2 heads, q=3, k=4
    l0 = torch.rand(1, 2, 3, 4)
    l1 = torch.rand(1, 2, 3, 4)
    out = reduce_step_attention([l0, l1], layer_reduce="mean")
    expected = ((l0[0].mean(0) + l1[0].mean(0)) / 2).numpy()
    assert out.shape == (3, 4)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)


def test_reduce_step_attention_last_layer_only():
    l0 = torch.zeros(1, 2, 2, 2)
    l1 = torch.ones(1, 2, 2, 2)
    out = reduce_step_attention([l0, l1], layer_reduce="last")
    np.testing.assert_allclose(out, np.ones((2, 2)), atol=1e-6)


def test_reduce_step_attention_int_index():
    l0 = torch.zeros(1, 1, 2, 2)
    l1 = torch.full((1, 1, 2, 2), 5.0)
    out = reduce_step_attention([l0, l1], layer_reduce=1)
    np.testing.assert_allclose(out, np.full((2, 2), 5.0), atol=1e-6)


def test_reduce_step_attention_none_raises():
    with pytest.raises(ValueError):
        reduce_step_attention([None], layer_reduce="mean")
    with pytest.raises(ValueError):
        reduce_step_attention([], layer_reduce="mean")


# --------------------------------------------------------------------------- #
# Column / grid bookkeeping                                                    #
# --------------------------------------------------------------------------- #

def test_image_column_indices_and_truncation():
    seq = torch.tensor([5, 999, 999, 999, 999, 999, 999, 7, 8])
    cols = image_column_indices(seq, image_token_id=999)
    np.testing.assert_array_equal(cols, np.array([1, 2, 3, 4, 5, 6]))
    cols4 = image_column_indices(seq, image_token_id=999, max_cols=4)
    np.testing.assert_array_equal(cols4, np.array([1, 2, 3, 4]))


def test_grid_from_row_reshape_and_mismatch():
    v = np.arange(4, dtype=np.float32)
    g = grid_from_row(v, 2, 2)
    np.testing.assert_array_equal(g, np.array([[0, 1], [2, 3]], dtype=np.float32))
    with pytest.raises(ValueError):
        grid_from_row(np.arange(3), 2, 2)


def test_side_from_vision_cfg():
    assert side_from_vision_cfg(448, 14, 0.5) == 16   # InternVL3 merged grid
    assert side_from_vision_cfg(336, 14, 1.0) == 24    # LLaVA base global view


def test_build_spliced_ids():
    orig = torch.tensor([10, 11, 12, 13])
    spliced = build_spliced_ids(orig, insert_pos=2, n_queries=3, sentinel=-1)
    np.testing.assert_array_equal(spliced.numpy(), np.array([10, 11, -1, -1, -1, 12, 13]))
    # Splice must land on the same device as the input (R block + orig get
    # torch.cat'd, which refuses to mix devices — regression guard for the
    # GPU-only bug the smoke run caught).
    assert spliced.device == orig.device
    # insert at the very end (LLaVA inference path)
    spliced_end = build_spliced_ids(orig, insert_pos=4, n_queries=2)
    np.testing.assert_array_equal(spliced_end.numpy(), np.array([10, 11, 12, 13, -1, -1]))
    with pytest.raises(ValueError):
        build_spliced_ids(orig, insert_pos=99, n_queries=1)


# --------------------------------------------------------------------------- #
# Row classification                                                           #
# --------------------------------------------------------------------------- #

def test_classify_prefill_rows_kinds():
    # [special, img, img, text, text]  image_token_id=999, special id=1
    seq = torch.tensor([1, 999, 999, 50, 51])
    tok = FakeTokenizer(special_ids=[1])
    rows = classify_prefill_rows(seq, image_token_id=999, tokenizer=tok)
    # image positions (1,2) are skipped; 0 special, 3 & 4 text
    kinds = [(idx, kind) for (idx, _tid, _lab, kind) in rows]
    assert kinds == [(0, KIND_SPECIAL), (3, KIND_TEXT), (4, KIND_TEXT)]
    labels = {idx: lab for (idx, _tid, lab, _k) in rows}
    assert labels[3] == "<50>" and labels[4] == "<51>"


def test_classify_prefill_rows_skip_special_and_limit():
    seq = torch.tensor([1, 50, 51, 52])
    tok = FakeTokenizer(special_ids=[1])
    rows = classify_prefill_rows(seq, image_token_id=999, tokenizer=tok, skip_special=True)
    assert [idx for (idx, *_r) in rows] == [1, 2, 3]
    rows_lim = classify_prefill_rows(seq, image_token_id=999, tokenizer=tok, limit=2)
    assert len(rows_lim) == 2


def test_classify_prefill_rows_r_tokens():
    # spliced: [text, R, R, image, text]
    seq = torch.tensor([50, -1, -1, 999, 51])
    tok = FakeTokenizer(special_ids=[])
    rows = classify_prefill_rows(
        seq, image_token_id=999, tokenizer=tok, r_positions={1, 2}, sentinel=-1
    )
    by_idx = {idx: (tid, lab, kind) for (idx, tid, lab, kind) in rows}
    assert by_idx[1][2] == KIND_R and by_idx[2][2] == KIND_R
    assert by_idx[1][1] == "R[0]" and by_idx[2][1] == "R[1]"  # label
    assert 3 not in by_idx  # image skipped
    assert by_idx[0][2] == KIND_TEXT and by_idx[4][2] == KIND_TEXT


def test_max_input_tokens_cap_preserves_r_rows():
    """Mirror capture_token_attention's split+cap: capping text rows must not
    drop R rows (which sit near the end of the prompt)."""
    # [t, t, t, t, R, R]  -> 4 text then 2 R
    seq = torch.tensor([50, 51, 52, 53, -1, -1])
    tok = FakeTokenizer()
    rows = classify_prefill_rows(seq, image_token_id=999, tokenizer=tok,
                                 r_positions={4, 5}, limit=-1)
    input_rows = [r for r in rows if r[3] != KIND_R]
    r_rows = [r for r in rows if r[3] == KIND_R]
    max_input_tokens = 2
    input_rows = input_rows[:max_input_tokens]
    assert len(input_rows) == 2
    assert len(r_rows) == 2  # R rows survive the text cap


def test_classify_prefill_rows_attention_mask_drops_padding():
    seq = torch.tensor([50, 51, 52])
    tok = FakeTokenizer()
    rows = classify_prefill_rows(seq, image_token_id=999, tokenizer=tok,
                                 attention_mask=[1, 0, 1])
    assert [idx for (idx, *_r) in rows] == [0, 2]


# --------------------------------------------------------------------------- #
# Per-token map extraction                                                     #
# --------------------------------------------------------------------------- #

def test_per_token_maps_from_prefill_peak_location():
    L = 8
    image_cols = np.array([0, 1, 2, 3])  # 2x2 grid
    qk = np.zeros((L, L), dtype=np.float32)
    # text row 5 puts most mass on image col 1 -> grid cell (0,1)
    qk[5, [0, 1, 2, 3]] = [0.1, 0.7, 0.1, 0.1]
    rows = [(5, 50, "<50>", KIND_TEXT)]
    maps = per_token_maps_from_prefill(qk, rows, image_cols, h=2, w=2)
    assert len(maps) == 1
    g = maps[0].grid
    assert g.shape == (2, 2)
    assert np.unravel_index(np.argmax(g), g.shape) == (0, 1)
    assert maps[0].kind == KIND_TEXT and maps[0].index == 5


def test_per_token_maps_from_decode_row_selection():
    image_cols = np.array([0, 1, 2, 3])
    prefill_len = 8
    # step 0: prefill, the LAST row (idx -1) is what produced token 0
    step0 = torch.zeros(1, 1, prefill_len, prefill_len)
    step0[0, 0, -1, [0, 1, 2, 3]] = torch.tensor([0.0, 0.0, 0.9, 0.1])  # peak at col 2 -> (1,0)
    # a decoy on a non-last row that must be IGNORED
    step0[0, 0, 0, [0, 1, 2, 3]] = torch.tensor([9.0, 0.0, 0.0, 0.0])
    # step 1: decode, q=1, row 0
    step1 = torch.zeros(1, 1, 1, prefill_len + 1)
    step1[0, 0, 0, [0, 1, 2, 3]] = torch.tensor([0.0, 0.0, 0.0, 0.8])  # peak col 3 -> (1,1)
    attentions = [[step0], [step1]]
    gen_rows = [(0, 100, "a"), (1, 101, "b")]
    maps = per_token_maps_from_decode(attentions, gen_rows, image_cols, h=2, w=2)
    assert len(maps) == 2
    assert np.unravel_index(np.argmax(maps[0].grid), (2, 2)) == (1, 0)
    assert np.unravel_index(np.argmax(maps[1].grid), (2, 2)) == (1, 1)
    assert all(m.kind == KIND_GENERATED for m in maps)


def test_mean_map_normalises():
    m1 = TokenMap(0, 1, "a", KIND_TEXT, np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32))
    m2 = TokenMap(1, 2, "b", KIND_TEXT, np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    mm = mean_map([m1, m2])
    assert mm.shape == (2, 2)
    assert pytest.approx(mm.sum(), rel=1e-6) == 1.0
    np.testing.assert_allclose(mm, np.array([[0.5, 0.0], [0.0, 0.5]]), atol=1e-6)
    assert mean_map([]) is None


# --------------------------------------------------------------------------- #
# Integration: simulate a frozen prefill end-to-end (no model)                 #
# --------------------------------------------------------------------------- #

def test_frozen_prefill_pipeline_simulation():
    """seq = [special, 4 image tokens, 'where', '?']; verify the 'where' token's
    grid peaks where we planted the attention mass."""
    img_id = 999
    seq = torch.tensor([1, img_id, img_id, img_id, img_id, 50, 63])
    tok = FakeTokenizer(special_ids=[1])
    h = w = 2
    L = seq.shape[0]
    image_cols = image_column_indices(seq, img_id)
    assert image_cols.shape[0] == h * w

    # Build a 1-layer, 1-head attention; row for token idx 5 ('<50>') -> col index
    # 2 within image block (sequence position 3) gets the mass -> grid cell (1,0).
    attn = torch.zeros(1, 1, L, L)
    attn[0, 0, 5, image_cols.tolist()] = torch.tensor([0.0, 0.0, 0.95, 0.05])
    qk = reduce_step_attention([attn], "mean")

    rows = classify_prefill_rows(seq, img_id, tok)
    text_rows = [r for r in rows if r[3] == KIND_TEXT]
    maps = per_token_maps_from_prefill(qk, text_rows, image_cols, h, w)
    by_idx = {m.index: m for m in maps}
    assert np.unravel_index(np.argmax(by_idx[5].grid), (h, w)) == (1, 0)


# --------------------------------------------------------------------------- #
# Rendering smoke test (matplotlib Agg writes files)                           #
# --------------------------------------------------------------------------- #

def _load_orchestrator():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "scripts", "analysis", "visualize_token_attention.py")
    spec = importlib.util.spec_from_file_location("viz_token_attn_cli", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_render_writes_files(tmp_path):
    from PIL import Image

    mod = _load_orchestrator()
    image = Image.new("RGB", (64, 48), color=(120, 120, 120))
    h = w = 4
    rng = np.random.default_rng(0)
    frozen = [TokenMap(i, i + 50, f"<{i}>", KIND_TEXT, rng.random((h, w)).astype(np.float32))
              for i in range(3)]
    ri = [TokenMap(i, i + 50, f"<{i}>", KIND_TEXT, rng.random((h, w)).astype(np.float32))
          for i in range(3)]

    out = tmp_path / "input_tokens.png"
    mod.render_side_by_side(image, h, w, frozen, ri, str(out), suptitle="test")
    assert out.exists() and out.with_suffix(".pdf").exists()

    gen = [TokenMap(i, i, chr(97 + i), KIND_GENERATED, rng.random((h, w)).astype(np.float32))
           for i in range(2)]
    out2 = tmp_path / "generated.png"
    mod.render_single_condition(image, h, w, gen, str(out2), suptitle="gen", cmap="viridis")
    assert out2.exists()

    results = {
        "frozen": {"h": h, "w": w, "answer": "yes",
                   "input_mean": rng.random((h, w)).astype(np.float32),
                   "generated_mean": rng.random((h, w)).astype(np.float32)},
    }
    out3 = tmp_path / "summary.png"
    mod.render_summary(image, results, str(out3))
    assert out3.exists()
