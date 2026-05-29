"""Per-token text→image attention extraction for InternVL3 and LLaVA-Next.

This module turns a Hugging Face ``generate(output_attentions=True,
return_dict_in_generate=True)`` result into one spatial attention heatmap **per
text token** over the image patch grid. It complements
``src/utils/visualize_attention.py`` (which only aggregates over *generated*
answer tokens) by also exposing the **prefill** attention matrix — i.e. how
*every input token* (system prompt, special/boundary tokens, the query, and the
appended assistant-generation suffix) attends to the image patches.

Two model families are supported, both via the same decoder-self-attention path
(``attn_implementation="eager"`` is required to materialise the weights):

* **InternVL3** — single-tile inputs give a clean ``side×side`` merged grid
  (448/14·0.5 = 16 → 256 patch tokens). Image columns are every position whose
  ``input_ids == image_token_id``.

* **LLaVA-Next** — AnyRes packs ``[base_global_view, high-res grid + newlines]``.
  Only the **base global view** (the first ``base_side²`` image-token columns,
  e.g. 336/14 = 24 → 576) is cleanly reshapeable to a 2D grid and overlays onto
  the whole image, so that is what we visualise; the irregular unpadded high-res
  grid is intentionally excluded (a clean rectangular reshape of it would be a
  lie). See ``transformers.models.llava_next.modeling_llava_next.pack_image_features``.

For the **re-inspection** condition the wrapper splices ``n_queries`` learned
``R`` tokens into the sequence at ``insert_position``; those columns are neither
text nor image and are labelled ``r``. Image columns sit *before* the insert
position in both backends, so their indices are unchanged by the splice.

Design note: everything below the "model-touching" banner is the only part that
needs a GPU; every function above it operates on plain tensors/ndarrays and is
covered by ``tests/test_token_attention.py`` on CPU.
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch

LayerReduce = Union[str, int, Tuple[int, int]]

# Token "kind" tags used for labelling / colouring per-token heatmaps.
KIND_TEXT = "text"
KIND_SPECIAL = "special"
KIND_R = "r"
KIND_GENERATED = "generated"


@dataclasses.dataclass
class TokenMap:
    """One token's attention heatmap over the image patch grid.

    Attributes:
        index: Sequence position (prefill rows) or generation step (decode).
        token_id: Vocabulary id (``-1`` for spliced re-inspection ``R`` tokens).
        label: Decoded token string; special tokens are kept verbatim.
        kind: One of ``text`` / ``special`` / ``r`` / ``generated``.
        grid: ``(h, w)`` float32 attention mass over image patches (a causal
            softmax row restricted to image columns; sums to ``≤ 1``).
    """

    index: int
    token_id: int
    label: str
    kind: str
    grid: np.ndarray


# --------------------------------------------------------------------------- #
# Layer / head reduction                                                       #
# --------------------------------------------------------------------------- #

def select_layer_indices(n_layers: int, layer_reduce: LayerReduce = "mean") -> List[int]:
    """Resolve which decoder layers contribute to the aggregated attention.

    Args:
        n_layers: Number of decoder layers available for this step.
        layer_reduce: ``"mean"`` (all layers), ``"last"`` (final layer only), an
            ``int`` layer index (negatives count from the end), or a
            ``(start, end)`` half-open range.

    Returns:
        Sorted list of valid layer indices.

    Raises:
        ValueError: For out-of-range indices or an unrecognised spec.
    """
    if n_layers <= 0:
        raise ValueError(f"n_layers must be positive, got {n_layers}")

    if isinstance(layer_reduce, bool):  # guard: bool is a subclass of int
        raise ValueError(f"Invalid layer_reduce: {layer_reduce!r}")

    if isinstance(layer_reduce, int):
        i = layer_reduce if layer_reduce >= 0 else n_layers + layer_reduce
        if not 0 <= i < n_layers:
            raise ValueError(f"layer index {layer_reduce} out of range for {n_layers} layers")
        return [i]

    if isinstance(layer_reduce, tuple):
        if len(layer_reduce) != 2:
            raise ValueError(f"layer range must be (start, end), got {layer_reduce!r}")
        start, end = layer_reduce
        start = start if start >= 0 else n_layers + start
        end = end if end >= 0 else n_layers + end
        idxs = [i for i in range(start, end) if 0 <= i < n_layers]
        if not idxs:
            raise ValueError(f"layer range {layer_reduce} selects no valid layers ({n_layers} total)")
        return idxs

    if layer_reduce == "mean":
        return list(range(n_layers))
    if layer_reduce == "last":
        return [n_layers - 1]

    raise ValueError(f"Unrecognised layer_reduce: {layer_reduce!r}")


def reduce_step_attention(
    step_layer_attns: Sequence[torch.Tensor],
    layer_reduce: LayerReduce = "mean",
    batch_index: int = 0,
) -> np.ndarray:
    """Average one generation step's attention over heads, then over layers.

    Args:
        step_layer_attns: Tuple over decoder layers; each tensor is
            ``[batch, heads, q_len, k_len]`` (the HF per-step attention).
        layer_reduce: See :func:`select_layer_indices`.
        batch_index: Which batch row to extract (single-example viz uses 0).

    Returns:
        ``[q_len, k_len]`` float32 array: mean over heads, mean over the
        selected layers.

    Note:
        For grouped-query-attention LMs (InternVL3's Qwen2: 28 heads / 4 KV;
        LLaVA-Next's Mistral: 32 heads / 8 KV) transformers' eager attention
        calls ``repeat_kv`` *before* the softmax, so the materialised weights are
        already ``[batch, num_attention_heads, q, k]`` (KV heads expanded). The
        head-mean below therefore averages over all query heads, which is what we
        want. :func:`capture_token_attention` guards this with a head-count check.
    """
    if step_layer_attns is None or len(step_layer_attns) == 0 or step_layer_attns[0] is None:
        raise ValueError(
            "Attention tensors are missing/None — the model is not using eager "
            "attention. Reload with attn_implementation='eager'."
        )

    n_layers = len(step_layer_attns)
    idxs = select_layer_indices(n_layers, layer_reduce)

    acc: Optional[torch.Tensor] = None
    for li in idxs:
        # [heads, q, k] → mean over heads → [q, k]
        head_mean = step_layer_attns[li][batch_index].float().mean(dim=0)
        acc = head_mean if acc is None else acc + head_mean
    acc = acc / float(len(idxs))
    return acc.detach().cpu().numpy().astype(np.float32)


# --------------------------------------------------------------------------- #
# Column / grid bookkeeping                                                     #
# --------------------------------------------------------------------------- #

def image_column_indices(
    seq_token_ids: Union[torch.Tensor, Sequence[int], np.ndarray],
    image_token_id: int,
    max_cols: Optional[int] = None,
) -> np.ndarray:
    """Positions of image-placeholder tokens in a (possibly spliced) sequence.

    Args:
        seq_token_ids: 1D token ids for the prefill sequence. For the
            re-inspection condition this is the *spliced* sequence (R positions
            marked with a sentinel that differs from ``image_token_id``).
        image_token_id: The image placeholder id (InternVL ``image_token_id`` /
            LLaVA ``image_token_index``).
        max_cols: Keep only the first ``max_cols`` image columns. Used by
            LLaVA-Next to isolate the base global view (first ``base_side²``).

    Returns:
        int64 ndarray of column indices, in ascending sequence order.
    """
    seq = torch.as_tensor(seq_token_ids).reshape(-1)
    cols = (seq == image_token_id).nonzero(as_tuple=False).squeeze(-1)
    if max_cols is not None:
        cols = cols[:max_cols]
    return cols.cpu().numpy().astype(np.int64)


def grid_from_row(row_over_image: np.ndarray, h: int, w: int) -> np.ndarray:
    """Reshape a 1D per-image-column attention vector to an ``(h, w)`` grid.

    Raises:
        ValueError: If the vector length does not equal ``h * w`` (refuses to
            produce a misleading heatmap from a mismatched grid assumption).
    """
    v = np.asarray(row_over_image, dtype=np.float32).reshape(-1)
    if v.shape[0] != h * w:
        raise ValueError(
            f"image-column count {v.shape[0]} != grid {h}×{w}={h * w}; "
            "patch-grid assumption is off."
        )
    return v.reshape(h, w)


def side_from_vision_cfg(image_size: int, patch_size: int, downsample: float = 1.0) -> int:
    """Merged patch-grid side length: ``round((image_size // patch_size) * downsample)``.

    InternVL3: ``(448 // 14) * 0.5 = 16``. LLaVA-Next base view:
    ``(336 // 14) * 1.0 = 24``.
    """
    return int(round((image_size // patch_size) * downsample))


def build_spliced_ids(
    orig_ids: Union[torch.Tensor, Sequence[int], np.ndarray],
    insert_pos: int,
    n_queries: int,
    sentinel: int = -1,
) -> torch.Tensor:
    """Reconstruct the re-inspection condition's spliced prefill token ids.

    The wrapper inserts ``n_queries`` R tokens at ``insert_pos``; we mirror that
    by inserting ``sentinel`` values so image/text/R columns can be classified.

    Returns:
        1D LongTensor of length ``len(orig_ids) + n_queries``.
    """
    orig = torch.as_tensor(orig_ids).reshape(-1)
    if not 0 <= insert_pos <= orig.shape[0]:
        raise ValueError(f"insert_pos {insert_pos} out of range for length {orig.shape[0]}")
    # Match orig's device — input_ids live on the GPU at capture time, and
    # torch.cat refuses to mix devices.
    mid = torch.full((n_queries,), sentinel, dtype=orig.dtype, device=orig.device)
    return torch.cat([orig[:insert_pos], mid, orig[insert_pos:]], dim=0)


def classify_prefill_rows(
    seq_token_ids: Union[torch.Tensor, Sequence[int], np.ndarray],
    image_token_id: int,
    tokenizer,
    r_positions: Optional[Set[int]] = None,
    sentinel: int = -1,
    skip_special: bool = False,
    attention_mask: Optional[Sequence[int]] = None,
    limit: int = -1,
) -> List[Tuple[int, int, str, str]]:
    """Label every non-image prefill position as text / special / R.

    Args:
        seq_token_ids: 1D prefill token ids (spliced for the RI condition).
        image_token_id: Image placeholder id (these positions are *skipped* —
            they are patches, not text tokens).
        tokenizer: Anything exposing ``decode([id])`` and ``all_special_ids``.
        r_positions: Indices occupied by spliced R tokens.
        sentinel: Sentinel id marking R positions in ``seq_token_ids``.
        skip_special: Drop special tokens from the returned rows when True.
        attention_mask: Optional 1D mask; positions with 0 are dropped (padding).
        limit: Keep at most this many rows (``-1`` = all). Useful to cap very
            long prompts in the rendered figure.

    Returns:
        List of ``(seq_index, token_id, label, kind)`` for rendered rows, in
        sequence order.
    """
    seq = torch.as_tensor(seq_token_ids).reshape(-1).tolist()
    r_positions = r_positions or set()
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])

    rows: List[Tuple[int, int, str, str]] = []
    for idx, tid in enumerate(seq):
        if attention_mask is not None and not bool(attention_mask[idx]):
            continue
        if idx in r_positions or tid == sentinel:
            rows.append((idx, -1, f"R[{idx - min(r_positions)}]" if r_positions else "R", KIND_R))
            continue
        if tid == image_token_id:
            continue  # image patch column, not a text token
        is_special = tid in special_ids
        if is_special and skip_special:
            continue
        label = tokenizer.decode([tid])
        kind = KIND_SPECIAL if is_special else KIND_TEXT
        rows.append((idx, int(tid), label, kind))

    if limit is not None and limit >= 0:
        rows = rows[:limit]
    return rows


def per_token_maps_from_prefill(
    prefill_qk: np.ndarray,
    rows: Sequence[Tuple[int, int, str, str]],
    image_cols: np.ndarray,
    h: int,
    w: int,
) -> List[TokenMap]:
    """Build one :class:`TokenMap` per input token from the prefill matrix.

    Args:
        prefill_qk: ``[L, L]`` head+layer-reduced prefill attention.
        rows: ``(seq_index, token_id, label, kind)`` rows to render.
        image_cols: Image column indices (length ``h * w``).
        h, w: Patch-grid dimensions.
    """
    maps: List[TokenMap] = []
    for (idx, tid, label, kind) in rows:
        vec = prefill_qk[idx, image_cols]
        maps.append(TokenMap(idx, tid, label, kind, grid_from_row(vec, h, w)))
    return maps


def per_token_maps_from_decode(
    attentions: Sequence[Sequence[torch.Tensor]],
    gen_rows: Sequence[Tuple[int, int, str]],
    image_cols: np.ndarray,
    h: int,
    w: int,
    layer_reduce: LayerReduce = "mean",
    batch_index: int = 0,
) -> List[TokenMap]:
    """Build one :class:`TokenMap` per generated token.

    The HF ``generate`` attention structure has, for step ``t``:
        * ``t == 0`` (prefill): ``q_len == prefill_len`` → the row that produced
          the first new token is the **last** row (``-1``).
        * ``t > 0`` (decode):   ``q_len == 1`` → the only row is ``0``.

    Image columns always sit within the prefill prefix, so the same ``image_cols``
    indices are valid at every step (``k_len`` only grows on the right).
    """
    maps: List[TokenMap] = []
    for (t, tid, label) in gen_rows:
        qk = reduce_step_attention(attentions[t], layer_reduce, batch_index)
        row = qk[-1] if t == 0 else qk[0]
        vec = row[image_cols]
        maps.append(TokenMap(t, int(tid), label, KIND_GENERATED, grid_from_row(vec, h, w)))
    return maps


def mean_map(maps: Sequence[TokenMap]) -> Optional[np.ndarray]:
    """Mean grid over a set of token maps, renormalised to sum 1 (or ``None``)."""
    grids = [m.grid for m in maps if m.grid is not None]
    if not grids:
        return None
    m = np.stack(grids, axis=0).mean(axis=0).astype(np.float32)
    total = float(m.sum())
    return m / total if total > 0 else m


# --------------------------------------------------------------------------- #
# ▼▼▼ Model-touching code below this line (requires a GPU + loaded model) ▼▼▼  #
# --------------------------------------------------------------------------- #

def get_image_token_id(model) -> int:
    """Resolve the image-placeholder id from a wrapper or a base HF model."""
    tok = getattr(model, "_image_token_id", None)
    if tok is not None:
        return int(tok)
    base = getattr(model, "base_model", model)
    cfg = getattr(base, "config", None)
    tok = getattr(cfg, "image_token_id", None)
    if tok is None:
        raise RuntimeError("Could not resolve image_token_id from model/config.")
    return int(tok)


def plan_vision_grid(backend: str, model) -> Tuple[int, int, int]:
    """Return ``(h, w, max_image_cols)`` for the visualised patch grid.

    InternVL3 uses the full merged grid; LLaVA-Next uses only the base global
    view (first ``base_side²`` image columns).
    """
    base = getattr(model, "base_model", model)
    cfg = getattr(base, "config", None)
    vcfg = getattr(cfg, "vision_config", None)
    if vcfg is None:
        raise RuntimeError("model.config.vision_config missing — cannot infer patch grid.")
    img_size = vcfg.image_size[0] if isinstance(vcfg.image_size, (list, tuple)) else vcfg.image_size
    patch = vcfg.patch_size[0] if isinstance(vcfg.patch_size, (list, tuple)) else vcfg.patch_size

    if backend == "internvl3":
        downsample = getattr(cfg, "downsample_ratio", 0.5)
        side = side_from_vision_cfg(img_size, patch, downsample)
        return side, side, side * side
    if backend == "llava_next":
        side = side_from_vision_cfg(img_size, patch, 1.0)  # base global view
        return side, side, side * side
    raise ValueError(f"Unsupported backend for token-attention viz: {backend}")


def _process_inputs(backend: str, processor, text: str, image_path: str, config):
    """Build processor tensors (InternVL forced single-tile; LLaVA AnyRes default)."""
    kwargs = dict(text=[text], images=[image_path], return_tensors="pt")
    if backend == "internvl3":
        kwargs["crop_to_patches"] = False  # single tile → clean square grid
    return processor(**kwargs)


@torch.no_grad()
def capture_token_attention(
    model,
    processor,
    backend: str,
    config,
    image_path: str,
    question: str,
    is_reinspection: bool,
    build_chat_messages,
    max_new_tokens: int = 32,
    layer_reduce: LayerReduce = "mean",
    max_input_tokens: int = -1,
    skip_special: bool = False,
    include_r_tokens: bool = True,
) -> dict:
    """Run one (image, question) through ``model`` and extract per-token maps.

    Returns a dict with: ``answer`` (str), ``h``/``w`` (grid), ``input_maps``
    (list[TokenMap] for input text/special tokens), ``r_maps`` (RI only),
    ``generated_maps`` (list[TokenMap]), plus ``input_mean``/``generated_mean``
    aggregate grids.
    """
    device = next(model.parameters()).device if hasattr(model, "parameters") else model.device
    tokenizer = getattr(processor, "tokenizer", processor)

    messages = build_chat_messages(question, image_path=image_path, system_prompt=config.system_prompt)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _process_inputs(backend, processor, text, image_path, config)
    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

    image_token_id = get_image_token_id(model)
    h, w, max_cols = plan_vision_grid(backend, model)

    # Re-inspection insert position (mirror the wrapper's inference-time logic).
    insert_pos = None
    n_queries = int(config.n_queries)
    if is_reinspection:
        insert_pos = int(
            model._find_insert_positions(
                inputs["input_ids"], attention_mask=inputs.get("attention_mask")
            )[0].item()
        )

    from src.evaluate import _generate_extra_kw

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        output_attentions=True,
        return_dict_in_generate=True,
        **_generate_extra_kw(processor),
    )
    gen_out = model.generate(**inputs, **gen_kwargs)
    attentions = gen_out.attentions
    sequences = gen_out.sequences
    if attentions is None or len(attentions) == 0 or attentions[0][0] is None:
        raise RuntimeError(
            "generate() returned no attentions — reload the model with "
            "attn_implementation='eager'."
        )

    # Guard the GQA assumption: eager attention must have expanded KV heads to
    # num_attention_heads (so the head-mean averages over all query heads). If a
    # future backend materialises only KV heads, fail loudly instead of emitting
    # a silently-wrong map. See reduce_step_attention's docstring.
    _materialised_heads = int(attentions[0][0].shape[1])
    _base_cfg = getattr(getattr(model, "base_model", model), "config", None)
    _text_cfg = getattr(_base_cfg, "text_config", _base_cfg)
    _expected_heads = getattr(_text_cfg, "num_attention_heads", None)
    if _expected_heads is not None and _materialised_heads != int(_expected_heads):
        raise RuntimeError(
            f"materialised attention heads ({_materialised_heads}) != model "
            f"num_attention_heads ({_expected_heads}); eager attention did not "
            "expand GQA heads as assumed — refusing to emit a wrong heatmap."
        )

    prefill_len = attentions[0][0].shape[-1]
    orig_ids = inputs["input_ids"][0]

    # Build the prefill sequence ids + image columns + R positions.
    r_positions: Set[int] = set()
    if is_reinspection:
        seq_ids = build_spliced_ids(orig_ids, insert_pos, n_queries)
        r_positions = set(range(insert_pos, insert_pos + n_queries))
        gen_token_ids = sequences[0]            # inputs_embeds path → generated only
    else:
        seq_ids = orig_ids
        gen_token_ids = sequences[0, prefill_len:]  # input_ids path → strip prompt
    if seq_ids.shape[0] != prefill_len:
        raise RuntimeError(
            f"reconstructed prefill length {seq_ids.shape[0]} != attention k_len "
            f"{prefill_len} (insert_pos={insert_pos}, n_queries={n_queries})."
        )

    image_cols = image_column_indices(seq_ids, image_token_id, max_cols=max_cols)
    if image_cols.shape[0] != h * w:
        raise RuntimeError(
            f"found {image_cols.shape[0]} image columns but grid is {h}×{w}={h * w} "
            f"({backend}); refusing to render a misleading heatmap."
        )

    # --- Prefill: per input-token maps ---
    prefill_qk = reduce_step_attention(attentions[0], layer_reduce)
    attn_mask_row = inputs.get("attention_mask")
    attn_mask_list = None
    if attn_mask_row is not None and not is_reinspection:
        attn_mask_list = attn_mask_row[0].tolist()  # only meaningful for unspliced seq
    rows = classify_prefill_rows(
        seq_ids, image_token_id, tokenizer,
        r_positions=r_positions, skip_special=skip_special,
        attention_mask=attn_mask_list, limit=-1,
    )
    input_rows = [r for r in rows if r[3] != KIND_R]
    r_rows = [r for r in rows if r[3] == KIND_R]
    # ``max_input_tokens`` caps the rendered input text/special rows only — it must
    # not starve the R-token panels, which sit near the end of the prompt (so a
    # global row cap would silently drop them).
    if max_input_tokens is not None and max_input_tokens >= 0:
        input_rows = input_rows[:max_input_tokens]
    input_maps = per_token_maps_from_prefill(prefill_qk, input_rows, image_cols, h, w)
    r_maps = (
        per_token_maps_from_prefill(prefill_qk, r_rows, image_cols, h, w)
        if (is_reinspection and include_r_tokens) else []
    )

    # --- Decode: per generated-token maps ---
    gen_ids_list = gen_token_ids.tolist()
    n_gen = min(len(attentions), len(gen_ids_list))
    gen_rows = [(t, gen_ids_list[t], tokenizer.decode([gen_ids_list[t]])) for t in range(n_gen)]
    generated_maps = per_token_maps_from_decode(attentions, gen_rows, image_cols, h, w, layer_reduce)

    answer = tokenizer.decode(gen_ids_list, skip_special_tokens=True).strip()

    return {
        "answer": answer,
        "h": h,
        "w": w,
        "input_maps": input_maps,
        "r_maps": r_maps,
        "generated_maps": generated_maps,
        "input_mean": mean_map([m for m in input_maps if m.kind == KIND_TEXT] or input_maps),
        "generated_mean": mean_map(generated_maps),
    }
