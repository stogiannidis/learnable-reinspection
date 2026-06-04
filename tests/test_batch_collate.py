"""Left-pad collation for batched VLM generation (src/utils/batch_collate.py).

Pins the contract the batched-eval fix relies on: per-token fields are
LEFT-padded (prompts flush-right) and per-image fields are concatenated along
dim 0, so each sample's tokenization is byte-identical to its bs=1 form.
"""

import torch

from src.utils.batch_collate import left_pad_collate

PAD = 0


def _sample(ids, n_tiles):
    L = len(ids)
    return {
        "input_ids": torch.tensor([ids]),
        "attention_mask": torch.ones(1, L, dtype=torch.long),
        "pixel_values": torch.arange(n_tiles * 4, dtype=torch.float).reshape(n_tiles, 1, 2, 2),
    }


def test_left_pad_aligns_right_and_preserves_content():
    a = _sample([5, 6, 7, 8], n_tiles=2)   # len 4
    b = _sample([9, 10], n_tiles=1)        # len 2 -> gets 2 left pads
    out = left_pad_collate([a, b], pad_id=PAD)

    assert out["input_ids"].shape == (2, 4)
    assert out["attention_mask"].shape == (2, 4)
    # Row 0 unchanged; row 1 left-padded with PAD, real tokens flush-right.
    assert out["input_ids"][0].tolist() == [5, 6, 7, 8]
    assert out["input_ids"][1].tolist() == [PAD, PAD, 9, 10]
    # Attention mask zeros exactly the left pad of the shorter row.
    assert out["attention_mask"][0].tolist() == [1, 1, 1, 1]
    assert out["attention_mask"][1].tolist() == [0, 0, 1, 1]


def test_pixel_values_concatenated_along_dim0():
    a = _sample([5, 6, 7, 8], n_tiles=2)
    b = _sample([9, 10], n_tiles=1)
    out = left_pad_collate([a, b], pad_id=PAD)
    # Tiles are concatenated (2 + 1), NOT padded — matches flattened placeholders.
    assert out["pixel_values"].shape == (3, 1, 2, 2)
    assert torch.equal(out["pixel_values"][:2], a["pixel_values"])
    assert torch.equal(out["pixel_values"][2:], b["pixel_values"])


def test_single_sample_is_identity():
    a = _sample([5, 6, 7, 8], n_tiles=2)
    out = left_pad_collate([a], pad_id=PAD)
    assert torch.equal(out["input_ids"], a["input_ids"])
    assert torch.equal(out["pixel_values"], a["pixel_values"])


def test_custom_pad_id_only_fills_input_ids():
    a = _sample([5, 6, 7], n_tiles=1)
    b = _sample([9], n_tiles=1)
    out = left_pad_collate([a, b], pad_id=99)
    assert out["input_ids"][1].tolist() == [99, 99, 9]   # input_ids padded with 99
    assert out["attention_mask"][1].tolist() == [0, 0, 1]  # mask still padded with 0


def _llava_sample(ids, n_patches):
    """LLaVA-Next AnyRes: pixel_values (1, num_patches_i, C, H, W) + image_sizes."""
    L = len(ids)
    return {
        "input_ids": torch.tensor([ids]),
        "attention_mask": torch.ones(1, L, dtype=torch.long),
        "pixel_values": torch.arange(n_patches * 4, dtype=torch.float).reshape(1, n_patches, 1, 2, 2),
        "image_sizes": torch.tensor([[336, 336]]),
    }


def test_llava_anyres_variable_patches_padded_to_max():
    a = _llava_sample([5, 6, 7, 8], n_patches=5)
    b = _llava_sample([9, 10], n_patches=3)
    out = left_pad_collate([a, b], pad_id=PAD)
    # Patch dim zero-padded to the batch max, then batched along dim 0 —
    # the model slices padding back off via image_sizes.
    assert out["pixel_values"].shape == (2, 5, 1, 2, 2)
    assert torch.equal(out["pixel_values"][0], a["pixel_values"][0])
    assert torch.equal(out["pixel_values"][1, :3], b["pixel_values"][0])
    assert torch.all(out["pixel_values"][1, 3:] == 0)
    # image_sizes still plain dim-0 concat.
    assert out["image_sizes"].shape == (2, 2)


def test_llava_anyres_equal_patches_plain_concat():
    a = _llava_sample([5, 6, 7], n_patches=5)
    b = _llava_sample([9, 10], n_patches=5)
    out = left_pad_collate([a, b], pad_id=PAD)
    assert out["pixel_values"].shape == (2, 5, 1, 2, 2)
    assert torch.equal(out["pixel_values"][0], a["pixel_values"][0])
    assert torch.equal(out["pixel_values"][1], b["pixel_values"][0])
