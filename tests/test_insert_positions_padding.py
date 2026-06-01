"""Padding-aware R-token insert positions for batched eval generation.

Batched greedy generation requires *left*-padding so every prompt ends at the
same column. The ``*WithReInspection`` wrappers must therefore locate the
generation suffix at the end of the *attended* span (derived from the attention
mask) rather than at ``real_len`` from the left, which only holds for
right-padded / unpadded inputs. These tests pin that behavior for left-pad
(batched eval), right-pad (training), and unpadded (B=1) without constructing a
real model — ``_find_insert_positions`` only touches ``self``'s suffix tensor,
``config.answer_ignore_index`` and ``self._sequence_lengths``.
"""

import torch
import pytest

# These imports require the HF model classes (present in the training/eval
# container). Skip cleanly on hosts without them so the rest of the suite runs.
internvl3 = pytest.importorskip("src.backends.internvl3")
llava_next = pytest.importorskip("src.backends.llava_next")

InternVL3WithReInspection = internvl3.InternVL3WithReInspection
LlavaNextWithReInspection = llava_next.LlavaNextWithReInspection


class _Stub:
    """Minimal stand-in exposing only what ``_find_insert_positions`` reads."""

    def __init__(self, cls, suffix):
        self._assistant_generation_suffix = torch.tensor(suffix, dtype=torch.long)
        self.config = type("C", (), {"answer_ignore_index": -100})()
        self._cls = cls

    def _sequence_lengths(self, input_ids, attention_mask):
        return self._cls._sequence_lengths(self, input_ids, attention_mask)

    def find(self, input_ids, attention_mask):
        return self._cls._find_insert_positions(
            self, input_ids, attention_mask=attention_mask, labels=None
        )


# Real content is "[1,2,3] + suffix[9,8]"; suffix_len = 2.
SUFFIX = [9, 8]


def test_internvl3_left_padding_inserts_before_suffix_uniformly():
    """Left-pad: insert column is L - suffix_len for every row (the point of left-pad)."""
    stub = _Stub(InternVL3WithReInspection, SUFFIX)
    # L = 8; row0 real len 5, row1 real len 4 — both left-padded to the right edge.
    input_ids = torch.tensor([
        [0, 0, 0, 1, 2, 3, 9, 8],
        [0, 0, 0, 0, 1, 2, 9, 8],
    ])
    attn = torch.tensor([
        [0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1],
    ])
    pos = stub.find(input_ids, attn)
    assert pos.tolist() == [6, 6]  # 8 - 2, same column for both rows


def test_internvl3_right_padding_matches_legacy():
    """Right-pad (training layout): insert before suffix within the real prefix."""
    stub = _Stub(InternVL3WithReInspection, SUFFIX)
    input_ids = torch.tensor([
        [1, 2, 3, 9, 8, 0, 0, 0],
        [1, 2, 9, 8, 0, 0, 0, 0],
    ])
    attn = torch.tensor([
        [1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0],
    ])
    pos = stub.find(input_ids, attn)
    assert pos.tolist() == [3, 2]  # real_len - suffix_len


def test_internvl3_no_suffix_left_pad_uses_attended_end():
    """InternVL3 has no generation suffix (None); the insert must still be the
    attended-span END under left-padding, not the token COUNT.

    This is the real-world case the earlier suffix-based tests missed: the bug
    returned ``attention_mask.sum()`` (count=5) as a position, landing inside the
    sequence, instead of the end index (8).
    """
    stub = _Stub(InternVL3WithReInspection, SUFFIX)
    stub._assistant_generation_suffix = None
    input_ids = torch.tensor([[0, 0, 0, 1, 2, 3, 4, 5]])  # 3 left pads + 5 real
    attn = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]])
    pos = stub.find(input_ids, attn)
    assert pos.tolist() == [8]  # END of attended span, NOT 5 (the count)


def test_internvl3_unpadded_b1():
    stub = _Stub(InternVL3WithReInspection, SUFFIX)
    input_ids = torch.tensor([[1, 2, 3, 9, 8]])
    attn = torch.tensor([[1, 1, 1, 1, 1]])
    pos = stub.find(input_ids, attn)
    assert pos.tolist() == [3]


def test_llava_inserts_after_suffix_at_attended_end():
    """LLaVA inserts AFTER the suffix, i.e. at the end of the attended span."""
    stub = _Stub(LlavaNextWithReInspection, SUFFIX)
    input_ids = torch.tensor([
        [0, 0, 0, 1, 2, 3, 9, 8],   # left-pad, real ends at col 8
        [1, 2, 9, 8, 0, 0, 0, 0],   # right-pad, real ends at col 4
    ])
    attn = torch.tensor([
        [0, 0, 0, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 0, 0],
    ])
    pos = stub.find(input_ids, attn)
    assert pos.tolist() == [8, 4]  # end of attended tokens per row
