"""Regression tests for the single-image-processing prompt/answer boundary.

``SpatialVQADataset.__getitem__`` used to run the image processor twice per
sample (once for the prompt, once for the full sequence) purely to find the
label-masking boundary. It now processes the image once and recovers the
boundary from a text-only tokenization difference via ``_supervised_prompt_len``.
These tests pin that the new boundary is identical to the old double-process
boundary and that only the answer tokens stay supervised, across a range of
image-token expansions and answer lengths.
"""

import types

import pytest
import torch

from src.data.spatial_dataset import SpatialVQADataset


class _FakeTokenizer:
    """Whitespace tokenizer; ``<image>`` is a single token in text mode."""

    def __call__(self, text, add_special_tokens=False):
        return types.SimpleNamespace(input_ids=text.split())


class _FakeProcessor:
    """Minimal VLM-processor stand-in.

    A ``<image>`` placeholder in the text expands to ``n_img_tokens`` ids when
    images are supplied (mirroring InternVL/Qwen tiling expansion); plain text
    tokenizes one id per whitespace token. The internal ``.tokenizer`` performs
    the text-only tokenization used by ``_supervised_prompt_len``.
    """

    IMAGE = "<image>"

    def __init__(self, n_img_tokens: int):
        self.n_img_tokens = n_img_tokens
        self.tokenizer = _FakeTokenizer()

    def __call__(self, text, images=None, **kwargs):
        ids = []
        for token in text[0].split():
            if token == self.IMAGE and images is not None:
                ids.extend([-1] * self.n_img_tokens)
            else:
                ids.append(1)
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


def _make_dataset(processor) -> SpatialVQADataset:
    # Bypass __init__ (which reads files); the helper only needs the processor.
    ds = SpatialVQADataset.__new__(SpatialVQADataset)
    ds.processor = processor
    ds.answer_ignore_index = -100
    return ds


@pytest.mark.parametrize("n_img_tokens", [1, 7, 256])
@pytest.mark.parametrize("answer", ["yes", "to the left", "a b c d e f g h i j"])
@pytest.mark.parametrize("system", ["", "you are a helpful assistant ."])
def test_single_process_prompt_len_matches_double_process(n_img_tokens, answer, system):
    processor = _FakeProcessor(n_img_tokens)
    ds = _make_dataset(processor)

    prompt_text = f"{system} user {processor.IMAGE} where is the cat ? assistant".strip()
    full_text = f"{prompt_text} {answer}"

    # Old behavior: process prompt + image and take its full (expanded) length.
    old_prompt_len = processor(text=[prompt_text], images=[object()])["input_ids"].shape[-1]
    # New behavior: process the full sequence once; recover boundary by text diff.
    full_len = processor(text=[full_text], images=[object()])["input_ids"].shape[-1]
    new_prompt_len = ds._supervised_prompt_len(prompt_text, full_text, full_len)

    assert new_prompt_len == old_prompt_len
    # The supervised (unmasked) span must be exactly the answer tokens.
    assert full_len - new_prompt_len == len(answer.split())


def test_masked_labels_leave_only_answer_supervised():
    processor = _FakeProcessor(n_img_tokens=7)
    ds = _make_dataset(processor)

    prompt_text = f"user {processor.IMAGE} q ? assistant"
    answer = "left side"
    full_text = f"{prompt_text} {answer}"

    full_inputs = processor(text=[full_text], images=[object()])
    prompt_len = ds._supervised_prompt_len(
        prompt_text, full_text, full_inputs["input_ids"].shape[-1]
    )
    labels = full_inputs["input_ids"].clone()
    labels[:, :prompt_len] = ds.answer_ignore_index

    supervised = int((labels[0] != ds.answer_ignore_index).sum())
    assert supervised == len(answer.split())
    assert bool((labels[0, :prompt_len] == ds.answer_ignore_index).all())


def test_no_image_still_recovers_answer_boundary():
    # Defensive: if a processor emits no image expansion, the boundary still
    # equals the answer-token count (answer_len is text-only).
    processor = _FakeProcessor(n_img_tokens=1)
    ds = _make_dataset(processor)

    prompt_text = "user describe ? assistant"
    answer = "a small red cube"
    full_text = f"{prompt_text} {answer}"

    full_len = processor(text=[full_text], images=None)["input_ids"].shape[-1]
    prompt_len = ds._supervised_prompt_len(prompt_text, full_text, full_len)
    assert full_len - prompt_len == len(answer.split())
