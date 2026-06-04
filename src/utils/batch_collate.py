"""Left-padding collation for batched VLM generation.

Batched eval tokenizes each (text, image) sample separately — the bs=1 path,
which is correct — because the HF InternVL3 processor's *batched* multi-image
path corrupts the shorter sample's attended text tokens (diagnosed in
``src/diag_batched.py``). This module reassembles those per-sample processor
outputs into one batch with LEFT-padding, so prompts stay flush-right and
decoder-only generation starts in lockstep across rows.
"""

from __future__ import annotations

from typing import Dict, List

import torch


def left_pad_collate(samples: List[Dict[str, object]], pad_id: int | None) -> Dict[str, object]:
    """Collate per-sample processor outputs into one left-padded batch.

    Per-token tensors (``input_ids`` / ``attention_mask`` / ``token_type_ids``,
    shape ``(1, L_i)``) are LEFT-padded to the longest sequence; ``input_ids`` is
    filled with ``pad_id`` (or 0 if unknown) and everything else with 0.
    Per-image tensors (``pixel_values``, ``image_grid_thw``, ``image_sizes``, …)
    are concatenated along dim 0 to match the flattened image-token placeholders
    the model scatters image features into. Non-tensor fields are list-flattened.

    Args:
        samples: One processor output dict per sample, each batch dim == 1.
        pad_id: Token id used to left-fill ``input_ids`` (falls back to 0).

    Returns:
        A single dict with batched tensors (``input_ids``/masks shape ``(B, L)``).
    """
    if not samples:
        return {}
    seq_lens = [s["input_ids"].shape[1] for s in samples]
    l_max = max(seq_lens)
    out: Dict[str, object] = {}
    for key in samples[0]:
        vals = [s[key] for s in samples]
        if not all(isinstance(v, torch.Tensor) for v in vals):
            flat: list = []
            for v in vals:
                flat.extend(v if isinstance(v, list) else [v])
            out[key] = flat
            continue
        per_token = all(
            v.dim() == 2 and v.shape[0] == 1 and v.shape[1] == seq_lens[i]
            for i, v in enumerate(vals)
        )
        if per_token:
            fill = pad_id if (key == "input_ids" and pad_id is not None) else 0
            rows = [
                torch.cat([v.new_full((1, l_max - v.shape[1]), fill), v], dim=1)
                if v.shape[1] < l_max else v
                for v in vals
            ]
            out[key] = torch.cat(rows, dim=0)
            continue
        # LLaVA-Next AnyRes emits per-sample (1, num_patches_i, C, H, W) with a
        # variable patch count, which breaks plain dim-0 concat. Zero-pad the
        # patch dim to the batch max (mirrors the trainer collate_fn); the model
        # re-derives true counts from ``image_sizes`` and slices the padding off.
        # InternVL-style (num_tiles_i, C, H, W) keeps flat dim-0 concat: its
        # leading dim is the variable one, so the all-dim0==1 guard excludes it.
        anyres = (
            vals[0].dim() >= 3
            and all(v.shape[0] == 1 for v in vals)
            and all(v.shape[2:] == vals[0].shape[2:] for v in vals)
            and len({v.shape[1] for v in vals}) > 1
        )
        if anyres:
            p_max = max(v.shape[1] for v in vals)
            vals = [
                torch.cat(
                    [v, v.new_zeros((v.shape[0], p_max - v.shape[1], *v.shape[2:]))],
                    dim=1,
                )
                if v.shape[1] < p_max else v
                for v in vals
            ]
        out[key] = torch.cat(vals, dim=0)
    return out
