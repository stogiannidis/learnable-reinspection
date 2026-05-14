"""Regression tests for attention visualization helpers and NPZ schema handling."""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.plot_motivation import plot_attention_comparison
from src.utils.attention_schema import attention_capture_plan, default_display_signal
from src.utils.visualize_attention import extract_decoder_image_attention


class _DummyTokenizer:
    all_special_ids = [0]

    _decoded = {
        21: "alpha",
        0: "<special>",
        22: "beta",
    }

    def decode(self, token_ids, skip_special_tokens=False):
        tok = token_ids[0]
        return self._decoded.get(tok, f"tok-{tok}")
def test_extract_decoder_image_attention_skips_special_tokens():
    sequences = np.array([[99, 99, 10, 11, 21, 0, 22]], dtype=np.int64)

    prefill = np.zeros((1, 1, 4, 4), dtype=np.float32)
    prefill[0, 0, -1, :] = np.array([0.6, 0.4, 0.0, 0.0], dtype=np.float32)
    step_1 = np.zeros((1, 1, 1, 5), dtype=np.float32)
    step_1[0, 0, 0, :] = np.array([0.1, 0.9, 0.0, 0.0, 0.0], dtype=np.float32)
    step_2 = np.zeros((1, 1, 1, 6), dtype=np.float32)
    step_2[0, 0, 0, :] = np.array([0.8, 0.2, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    mean_map, per_token_maps, labels, indices = extract_decoder_image_attention(
        sequences=torch.tensor(sequences),
        attentions=((torch.tensor(prefill),), (torch.tensor(step_1),), (torch.tensor(step_2),)),
        image_token_id=99,
        tokenizer=_DummyTokenizer(),
        h_patches=1,
        w_patches=2,
        n_show=8,
    )

    np.testing.assert_allclose(mean_map.reshape(-1), np.array([0.7, 0.3], dtype=np.float32), atol=1e-6)
    assert labels == ["alpha", "beta"]
    assert indices == [0, 2]
    assert len(per_token_maps) == 2


def test_attention_capture_plan_matches_backend_and_condition():
    assert attention_capture_plan("internvl3", "frozen") == ["decoder"]
    assert attention_capture_plan("internvl3", "reinspection") == ["decoder", "a_vis"]
    assert attention_capture_plan("qwen25vl", "frozen") == ["hidden_cosine"]
    assert attention_capture_plan("qwen25vl", "reinspection") == ["a_vis"]
    assert default_display_signal("reinspection", ["decoder", "a_vis"]) == "a_vis"
    assert default_display_signal("frozen", ["decoder"]) == "decoder"


def test_plot_attention_comparison_handles_legacy_and_expanded_npz(tmp_path):
    attn_dir = tmp_path / "attention_figures"
    out_dir = tmp_path / "figures"
    attn_dir.mkdir()

    image_path = tmp_path / "sample.png"
    Image.new("RGB", (12, 12), color="white").save(image_path)

    legacy_map_a = np.array([0.7, 0.3], dtype=np.float32)
    legacy_map_b = np.array([0.2, 0.8], dtype=np.float32)
    np.savez(
        attn_dir / "attn_pair_0.npz",
        attn_a=legacy_map_a,
        attn_b=legacy_map_b,
        h_merged=1,
        w_merged=2,
        image_path=str(image_path),
        question_a="legacy qa",
        question_b="legacy qb",
        answer_a="true",
        answer_b="false",
        gt_a="true",
        gt_b="false",
        relation="left of",
        condition="reinspection",
        attn_source="a_vis",
    )

    np.savez(
        attn_dir / "reinspection_attn_pair_1.npz",
        attn_a=legacy_map_a,
        attn_b=legacy_map_b,
        a_vis_a=legacy_map_a,
        a_vis_b=legacy_map_b,
        decoder_attn_a=np.array([0.4, 0.6], dtype=np.float32),
        decoder_attn_b=np.array([0.6, 0.4], dtype=np.float32),
        h_merged=1,
        w_merged=2,
        image_path=str(image_path),
        question_a="expanded qa",
        question_b="expanded qb",
        answer_a="true",
        answer_b="false",
        gt_a="true",
        gt_b="false",
        relation="right of",
        condition="reinspection",
        attn_source="a_vis",
        available_signals=np.asarray(["decoder", "a_vis"], dtype="<U32"),
        default_display_signal="a_vis",
    )

    plot_attention_comparison(str(out_dir), str(attn_dir))

    assert (out_dir / "reinspection_attention_comparison.pdf").exists()
    assert (out_dir / "reinspection_attention_comparison.png").exists()
