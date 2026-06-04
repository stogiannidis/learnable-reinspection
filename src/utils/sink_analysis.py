"""Attention-sink + massive-activation (per-token L2 norm) analysis for InternVL3.

Answers three questions, with numbers, on the frozen base VLM vs the
Re-Inspection model (which splices ``n_queries`` learned ``R`` tokens before the
assistant marker):

* **Q1 — is there an attention sink?** Per-token hidden-state L2-norm spikes
  (massive activations, Sun et al. 2024) co-located with attention mass piling
  onto the first token / BOS (Xiao et al. 2024, StreamingLLM).
* **Q2 — do answer tokens look at the image, or dump into the sink?** A
  per-row-normalised attention-budget decomposition over ``{bos, image, text}``.
* **Q3 — does Re-Inspection fix / relocate it?** Paired frozen-vs-RI deltas on
  BOS mass + image mass, and whether the ``R`` tokens become the new high-norm
  sink (R-token norms + attention received).

Design mirrors ``src/utils/token_attention.py``: everything above the
"model-touching" banner is plain numpy/torch math, unit-tested on CPU in
``tests/test_sink_analysis.py``; only ``capture_sink_example`` needs a GPU.

Key measurement choices (grounded in the literature + an adversarial review of
HF-generate extraction on InternVL3; see ``docs`` / the design notes):

* Work from the **raw per-(layer, head)** prefill attention, never a
  head/layer-collapsed mean — a sink lives in a minority of heads/layers and a
  global mean washes it out.
* The BOS-sink statistic is the mass the first key **receives** from later
  queries (``q >= 2``; rows ``q in {0,1}`` are trivial), compared against the
  **position-conditioned uniform-causal null** ``mean_q 1/(q+1)`` — *not*
  ``1/L`` and *not* a raw column-sum (which over-weights early keys by
  triangular geometry).
* A query's **budget** is its softmax row (already sums to 1 over valid keys);
  the ``{bos, image, text, r}`` decomposition is therefore splice-invariant for
  ``{bos, image, text}`` and directly comparable between the frozen (len ``L``)
  and RI (len ``L + n_queries``) sequences. ``R`` mass is reported separately.
* Norms are computed in **float32** and summarised with **median + IQR** (they
  are heavy-tailed); the massive-activation flag uses the Sun et al. criterion
  on a single feature magnitude (L-inf) vs the per-token median over dims.
* ``hidden_states[0]`` is the **embeddings**, ``hidden_states[i]`` is the output
  of decoder layer ``i`` — never off-by-one.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

# Disjoint, exhaustive token-column kinds used to bucket both attention columns
# and hidden-state rows over the (possibly spliced) prefill sequence.
KIND_BOS = "bos"
KIND_IMAGE = "image"
KIND_TEXT = "text"
KIND_R = "r"
ALL_KINDS = (KIND_BOS, KIND_IMAGE, KIND_TEXT, KIND_R)


# --------------------------------------------------------------------------- #
# Token-kind column masks                                                      #
# --------------------------------------------------------------------------- #

def build_kind_masks(
    seq_ids: Union[torch.Tensor, Sequence[int], np.ndarray],
    image_token_id: int,
    insert_pos: Optional[int] = None,
    n_queries: int = 0,
) -> Dict[str, np.ndarray]:
    """Partition the prefill columns into disjoint kind masks.

    Priority (a position belongs to exactly one kind): ``bos`` (position 0) >
    ``r`` (the spliced re-inspection block, RI only) > ``image`` (placeholder
    id) > ``text`` (everything else).

    Args:
        seq_ids: 1D prefill token ids (the *spliced* sequence for the RI
            condition, where R positions hold a sentinel that is not the image
            id; image columns sit before ``insert_pos`` so the splice does not
            move them).
        image_token_id: Image-placeholder id.
        insert_pos: Start of the R block (RI only); ``None`` for frozen.
        n_queries: Number of spliced R tokens (RI only).

    Returns:
        Dict ``{bos, image, text, r}`` of boolean ndarrays of length ``L``. The
        four masks are mutually exclusive and cover every position.
    """
    seq = np.asarray(torch.as_tensor(seq_ids).reshape(-1).tolist())
    L = seq.shape[0]

    bos = np.zeros(L, dtype=bool)
    if L > 0:
        bos[0] = True

    r = np.zeros(L, dtype=bool)
    if insert_pos is not None and n_queries > 0:
        end = min(insert_pos + n_queries, L)
        r[insert_pos:end] = True

    image = (seq == image_token_id) & ~bos & ~r
    text = ~(bos | image | r)
    return {KIND_BOS: bos, KIND_IMAGE: image, KIND_TEXT: text, KIND_R: r}


def answer_query_rows(prefill_len: int, suffix_len: int) -> np.ndarray:
    """Rows that feed the answer: the assistant-marker suffix + first-gen query.

    R is spliced *before* the suffix and nothing follows the suffix in the
    prompt, so for BOTH conditions the suffix occupies the last ``suffix_len``
    prefill rows; the very last row is the query that produces the first
    generated token. Falls back to the last row when ``suffix_len <= 0``.
    """
    n = max(1, int(suffix_len))
    start = max(0, prefill_len - n)
    return np.arange(start, prefill_len, dtype=np.int64)


# --------------------------------------------------------------------------- #
# Attention-sink statistics (per layer, raw per-head)                          #
# --------------------------------------------------------------------------- #

def bos_received_per_head(layer_attn: np.ndarray, q_min: int = 2) -> np.ndarray:
    """Mean attention the first key (BOS) receives from queries ``q >= q_min``.

    Args:
        layer_attn: ``[H, L, L]`` fp32 post-softmax weights for one layer (rows
            sum to 1 over valid causal keys ``k <= q``).
        q_min: Exclude trivial early rows — ``q=0`` is ``1.0`` on key 0 by
            construction and ``q=1`` only sees ``{0, 1}``.

    Returns:
        ``[H]`` per-head BOS-received mass ``a0[h]``.
    """
    H, L, _ = layer_attn.shape
    qs = np.arange(q_min, L)
    if qs.size == 0:
        qs = np.array([L - 1])
    return layer_attn[:, qs, 0].mean(axis=1).astype(np.float64)


def uniform_col0_expectation(L: int, q_min: int = 2) -> float:
    """Position-conditioned uniform-causal expected mass on key 0.

    Under uniform attention a query at position ``q`` spreads ``1/(q+1)`` over
    its ``q+1`` valid keys, so the expected mass on key 0 is ``1/(q+1)``. The
    null for ``a0`` is the mean of that over the same ``q >= q_min`` set.
    """
    qs = np.arange(q_min, L)
    if qs.size == 0:
        qs = np.array([L - 1])
    return float(np.mean(1.0 / (qs + 1.0)))


def budget_per_head(
    layer_attn: np.ndarray,
    query_rows: Sequence[int],
    masks: Dict[str, np.ndarray],
    extra_cols: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Per-head attention budget of ``query_rows``, split by column kind.

    Each row already sums to 1 over its valid keys, so the kind masses sum to 1
    (plus any ``extra_cols`` bucket). Causal-invalid columns (``k > q``) are
    exactly 0 and contribute nothing, so summing over all kind columns is safe.

    Args:
        layer_attn: ``[H, Lq_or_L, K]`` fp32 weights. ``K`` is the key length
            (``L`` at prefill; ``L + t`` at decode step ``t``).
        query_rows: Row indices to average over.
        masks: ``{bos, image, text, r}`` boolean column masks of length ``L``.
        extra_cols: Optional boolean mask over *all* ``K`` columns flagging keys
            beyond the prefill prefix (generated tokens at decode time); their
            mass is returned under key ``"gen"``.

    Returns:
        Dict kind -> ``[H]`` mean budget, plus ``"_row_sum"`` -> ``[H]`` (a
        sanity check; should be ~1).
    """
    qr = np.asarray(query_rows, dtype=np.int64)
    sub = layer_attn[:, qr, :]  # [H, Q, K]
    K = sub.shape[2]
    out: Dict[str, np.ndarray] = {}
    for kind, m in masks.items():
        cols = m
        if m.shape[0] != K:  # pad/truncate the length-L mask to the key length
            cols = np.zeros(K, dtype=bool)
            cols[: min(K, m.shape[0])] = m[: min(K, m.shape[0])]
        out[kind] = sub[:, :, cols].sum(axis=2).mean(axis=1).astype(np.float64)
    if extra_cols is not None:
        if extra_cols.shape[0] != K:
            raise ValueError(
                f"extra_cols length {extra_cols.shape[0]} != key length {K}; "
                "build it over ALL keys (prefill + generated), not the prefill prefix."
            )
        out["gen"] = sub[:, :, extra_cols].sum(axis=2).mean(axis=1).astype(np.float64)
    out["_row_sum"] = sub.sum(axis=2).mean(axis=1).astype(np.float64)
    return out


def row_entropy_per_head(
    layer_attn: np.ndarray, query_rows: Sequence[int], eps: float = 1e-12
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalised attention entropy + effective support for ``query_rows``.

    Returns ``(Hnorm[H], ESS[H])`` where ``Hnorm = H / log(q+1)`` in ``[0, 1]``
    (1 = uniform-causal) and ``ESS = exp(H)`` is the effective number of
    attended keys. Zero-probability (masked / causal-invalid) entries
    contribute ``0`` since ``p*log(p+eps) -> 0`` as ``p -> 0``.
    """
    qr = np.asarray(query_rows, dtype=np.int64)
    sub = layer_attn[:, qr, :].astype(np.float64)  # [H, Q, K]
    ent = -(sub * np.log(sub + eps)).sum(axis=2)  # [H, Q]
    denom = np.log((qr + 1).astype(np.float64))
    denom[denom == 0] = 1.0
    hnorm = (ent / denom[None, :]).mean(axis=1)
    ess = np.exp(ent).mean(axis=1)
    return hnorm.astype(np.float64), ess.astype(np.float64)


def received_per_key_headmean(layer_attn: np.ndarray) -> np.ndarray:
    """Mean attention each key receives from strictly-later queries (head-mean).

    ``recv[k] = mean_{q > k} A_headmean[q, k]`` (normalised by the valid-query
    count so it is comparable across positions, unlike a raw column-sum).
    """
    A = layer_attn.mean(axis=0).astype(np.float64)  # [L, L]
    L = A.shape[0]
    recv = np.zeros(L, dtype=np.float64)
    for k in range(L):
        if k + 1 < L:
            recv[k] = A[k + 1 :, k].mean()
        else:
            recv[k] = A[k, k]
    return recv


def uniform_causal_received(L: int) -> np.ndarray:
    """``C_unif[k] = mean_{q > k} 1/(q+1)`` — the geometry-only received null."""
    cu = np.zeros(L, dtype=np.float64)
    for k in range(L):
        if k + 1 < L:
            qs = np.arange(k + 1, L)
            cu[k] = np.mean(1.0 / (qs + 1.0))
        else:
            cu[k] = 1.0 / (k + 1.0)
    return cu


def sink_rate(a0_lh: np.ndarray, eps: float) -> float:
    """Fraction of ``(layer, head)`` pairs whose BOS-received mass ``>= eps``."""
    a0_lh = np.asarray(a0_lh, dtype=np.float64)
    if a0_lh.size == 0:
        return float("nan")
    return float((a0_lh >= eps).mean())


# --------------------------------------------------------------------------- #
# Massive-activation / per-token L2-norm statistics (per hidden-state layer)   #
# --------------------------------------------------------------------------- #

def token_l2_norms(hs_layer: np.ndarray) -> np.ndarray:
    """Per-token L2 norm ``||h[pos, :]||_2`` over feature dims, in float64."""
    h = np.asarray(hs_layer, dtype=np.float64)
    return np.sqrt((h * h).sum(axis=1))


def per_kind_norm_stats(n2: np.ndarray, masks: Dict[str, np.ndarray]) -> Dict[str, dict]:
    """Median / IQR / max / count of per-token L2 norms within each kind."""
    out: Dict[str, dict] = {}
    n2 = np.asarray(n2, dtype=np.float64)
    for kind, m in masks.items():
        # Truncate both to the common prefix so neither a longer mask (drops
        # tail columns) nor a shorter one (boolean-index IndexError) can bite.
        K = min(m.shape[0], n2.shape[0])
        v = n2[:K][m[:K]]
        if v.size == 0:
            out[kind] = dict(median=float("nan"), q25=float("nan"), q75=float("nan"),
                             max=float("nan"), n=0)
        else:
            out[kind] = dict(
                median=float(np.median(v)),
                q25=float(np.percentile(v, 25)),
                q75=float(np.percentile(v, 75)),
                max=float(v.max()),
                n=int(v.size),
            )
    return out


def massive_activation_stats(
    hs_layer: np.ndarray, mag_thresh: float = 100.0, ratio_thresh: float = 1000.0
) -> dict:
    """Sun et al. (2024) massive-activation diagnostics for one layer.

    A position is flagged when its largest single-feature magnitude exceeds
    ``mag_thresh`` AND is ``>= ratio_thresh`` times the per-token median
    feature magnitude. The ratio test is scale-free; the absolute cutoff is
    backbone-dependent and should be sanity-checked against observed scale.

    Returns dict with per-position ``linf``, ``argmax_dim``, ``ratio``, and a
    boolean ``flag``.
    """
    h = np.abs(np.asarray(hs_layer, dtype=np.float64))  # [L, d]
    linf = h.max(axis=1)
    argmax_dim = h.argmax(axis=1)
    med = np.median(h, axis=1)
    ratio = linf / np.maximum(med, 1e-12)
    flag = (linf > mag_thresh) & (ratio >= ratio_thresh)
    return dict(linf=linf, argmax_dim=argmax_dim, ratio=ratio, flag=flag)


def _average_ranks(a: np.ndarray) -> np.ndarray:
    """Average ranks (ties shared) — the basis for Spearman without scipy."""
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.shape[0], dtype=np.float64)
    ranks[order] = np.arange(a.shape[0], dtype=np.float64)
    # Resolve ties to their average rank.
    sa = a[order]
    i = 0
    n = a.shape[0]
    while i < n:
        j = i + 1
        while j < n and sa[j] == sa[i]:
            j += 1
        if j - i > 1:
            avg = ranks[order[i:j]].mean()
            ranks[order[i:j]] = avg
        i = j
    return ranks


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (Pearson on average ranks). NaN if degenerate."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape[0] != y.shape[0] or x.shape[0] < 3:
        return float("nan")
    rx, ry = _average_ranks(x), _average_ranks(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if denom == 0:
        return float("nan")
    return float((rx * ry).sum() / denom)


# --------------------------------------------------------------------------- #
# ▼▼▼ Model-touching code below this line (requires a GPU + loaded model) ▼▼▼  #
# --------------------------------------------------------------------------- #

def generation_suffix_len(tokenizer) -> int:
    """Length of the assistant-generation suffix (the marker rows before gen).

    Mirrors ``InternVL3WithReInspection._resolve_generation_suffix``: the diff
    between ``apply_chat_template(..., add_generation_prompt=True)`` and
    ``False``. Returns 1 (last row only) when it cannot be resolved.
    """
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        return 1
    messages = [{"role": "user", "content": "ping"}]
    try:
        with_p = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        without_p = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    except Exception:
        return 1
    if isinstance(with_p, dict):
        with_p = with_p["input_ids"]
    if isinstance(without_p, dict):
        without_p = without_p["input_ids"]
    if with_p and isinstance(with_p[0], list):
        with_p = with_p[0]
    if without_p and isinstance(without_p[0], list):
        without_p = without_p[0]
    n = len(with_p) - len(without_p)
    return max(1, int(n))


@torch.no_grad()
def capture_sink_example(
    model,
    processor,
    config,
    image_path: str,
    question: str,
    is_reinspection: bool,
    build_chat_messages,
    suffix_len: int,
    backend: str = "internvl3",
    max_new_tokens: int = 8,
    q_min: int = 2,
    mag_thresh: float = 100.0,
    ratio_thresh: float = 1000.0,
) -> dict:
    """Run one (image, question) and reduce its attention + hidden states.

    Returns a compact per-example record (small numpy arrays — never the raw
    ``L x L`` matrices) used by the orchestrator for cross-example aggregation.
    Requires ``attn_implementation='eager'`` and ``batch_size == 1``.
    """
    from src.evaluate import _generate_extra_kw
    from src.utils.token_attention import (
        _process_inputs,
        build_spliced_ids,
        get_image_token_id,
        image_column_indices,
        plan_vision_grid,
    )

    device = next(model.parameters()).device if hasattr(model, "parameters") else model.device
    tokenizer = getattr(processor, "tokenizer", processor)

    messages = build_chat_messages(question, image_path=image_path, system_prompt=config.system_prompt)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _process_inputs(backend, processor, text, image_path, config)
    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    if inputs["input_ids"].shape[0] != 1:
        raise RuntimeError("capture_sink_example requires batch_size == 1 (no left-pad).")

    image_token_id = get_image_token_id(model)
    h, w, max_cols = plan_vision_grid(backend, model)

    insert_pos = None
    n_queries = int(config.n_queries)
    if is_reinspection:
        insert_pos = int(
            model._find_insert_positions(
                inputs["input_ids"], attention_mask=inputs.get("attention_mask")
            )[0].item()
        )

    # NOTE: leave use_cache at its default (True). With the KV cache disabled,
    # every decode step would recompute attention at full sequence width and HF
    # generate retains each step's [1, H, L+t, L+t] tensor on-GPU until it
    # returns — quadratic growth and a hard CUDA OOM. With the cache, decode
    # steps are [1, H, 1, L+t] and the prefill step (the only one we read) is
    # bit-identical either way.
    gen_out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        output_attentions=True,
        output_hidden_states=True,
        return_dict_in_generate=True,
        **_generate_extra_kw(processor),
    )

    attentions = gen_out.attentions
    hidden = gen_out.hidden_states
    if attentions is None or len(attentions) == 0 or attentions[0][0] is None:
        raise RuntimeError("generate() returned no attentions — reload with attn_implementation='eager'.")
    if hidden is None or len(hidden) == 0 or hidden[0][0] is None:
        raise RuntimeError("generate() returned no hidden_states — pass output_hidden_states=True.")

    prefill_attn = attentions[0]                 # tuple over layers: [1, H, L, L]
    n_layers = len(prefill_attn)
    n_heads = int(prefill_attn[0].shape[1])
    prefill_len = int(prefill_attn[0].shape[-1])

    # GQA guard (eager must expand KV heads to query heads before softmax).
    base_cfg = getattr(getattr(model, "base_model", model), "config", None)
    text_cfg = getattr(base_cfg, "text_config", base_cfg)
    expected_heads = getattr(text_cfg, "num_attention_heads", None)
    if expected_heads is not None and n_heads != int(expected_heads):
        raise RuntimeError(
            f"materialised heads ({n_heads}) != num_attention_heads ({expected_heads}); "
            "eager did not expand GQA heads — refusing to emit wrong metrics."
        )

    # Reconstruct the spliced prefill id sequence (RI) or use the original.
    orig_ids = inputs["input_ids"][0]
    if is_reinspection:
        seq_ids = build_spliced_ids(orig_ids, insert_pos, n_queries)
        gen_token_ids = gen_out.sequences[0]                 # inputs_embeds path → gen-only
    else:
        seq_ids = orig_ids
        gen_token_ids = gen_out.sequences[0, prefill_len:]   # input_ids path → strip prompt

    prefill_hs_len = int(hidden[0][0].shape[1])
    if not (seq_ids.shape[0] == prefill_len == prefill_hs_len):
        raise RuntimeError(
            f"length mismatch: seq_ids={seq_ids.shape[0]} attn={prefill_len} hs={prefill_hs_len} "
            f"(insert_pos={insert_pos}, n_queries={n_queries})."
        )

    # Check the RAW image-token count before max_cols truncation: a multi-tile
    # leak (e.g. 512 tokens clamped to 256) would otherwise pass the post-
    # truncation check below and silently bucket only the first tile as image.
    raw_image_count = int((torch.as_tensor(seq_ids).reshape(-1) == image_token_id).sum().item())
    if raw_image_count != h * w:
        raise RuntimeError(
            f"raw image-token count {raw_image_count} != grid {h}x{w}={h * w}; "
            "multi-tile input — force crop_to_patches=False (single tile)."
        )
    image_cols = image_column_indices(seq_ids, image_token_id, max_cols=max_cols)
    if image_cols.shape[0] != h * w:
        raise RuntimeError(
            f"found {image_cols.shape[0]} image columns but grid is {h}x{w}={h * w}; "
            "force crop_to_patches=False (single tile)."
        )

    masks = build_kind_masks(seq_ids, image_token_id, insert_pos=insert_pos,
                             n_queries=n_queries if is_reinspection else 0)
    qr = answer_query_rows(prefill_len, suffix_len)
    e_unif_col0 = uniform_col0_expectation(prefill_len, q_min=q_min)

    # ---- Attention: iterate layers once, keep only reduced arrays. ----
    a0_lh = np.zeros((n_layers, n_heads), dtype=np.float64)
    budget_answer = {k: np.zeros((n_layers, n_heads), dtype=np.float64) for k in ALL_KINDS}
    row_sum_check = np.zeros((n_layers, n_heads), dtype=np.float64)
    ent_l = np.zeros(n_layers, dtype=np.float64)
    ess_l = np.zeros(n_layers, dtype=np.float64)
    recv_ratio_bos_l = np.zeros(n_layers, dtype=np.float64)
    recv_per_key_by_layer: List[np.ndarray] = []   # head-mean recv[k], for norm-corr
    budget_R_answer_l = np.zeros(n_layers, dtype=np.float64)
    recv_R_max_l = np.zeros(n_layers, dtype=np.float64)

    cu = uniform_causal_received(prefill_len)
    r_idx = np.nonzero(masks[KIND_R])[0]

    for li in range(n_layers):
        A = prefill_attn[li][0].float().cpu().numpy()  # [H, L, L]
        a0_lh[li] = bos_received_per_head(A, q_min=q_min)
        bud = budget_per_head(A, qr, masks)
        for k in ALL_KINDS:
            budget_answer[k][li] = bud[k]
        row_sum_check[li] = bud["_row_sum"]
        hnorm, ess = row_entropy_per_head(A, qr)
        ent_l[li] = float(hnorm.mean())
        ess_l[li] = float(ess.mean())

        recv = received_per_key_headmean(A)
        recv_per_key_by_layer.append(recv)
        recv_ratio_bos_l[li] = float(recv[0] / max(cu[0], 1e-12))

        if r_idx.size > 0:
            # R answer-budget (head-mean) and the strongest R sink on the
            # per-valid-query received scale (comparable to a0).
            budget_R_answer_l[li] = float(budget_answer[KIND_R][li].mean())
            recv_R = recv[r_idx] / np.maximum(cu[r_idx], 1e-12)
            recv_R_max_l[li] = float(recv_R.max())

    # Enforce the budget invariant: every answer row must sum to ~1 over its
    # valid keys (loose bound — bf16 rounding gives ~±0.01; a masking /
    # normalisation bug gives ~0.5 or ~2.0). A silent failure here would
    # mis-normalise the whole {bos, image, text, r} decomposition.
    rs_min, rs_max = float(row_sum_check.min()), float(row_sum_check.max())
    if not (0.9 <= rs_min and rs_max <= 1.1):
        raise RuntimeError(
            f"answer-row attention budgets do not sum to ~1: [{rs_min:.4f}, {rs_max:.4f}] "
            "— masking/normalisation bug; refusing to emit a mis-normalised decomposition."
        )

    # ---- Hidden states: iterate layers (n_layers + 1; index 0 = embeddings). ----
    n_hs = len(hidden[0])
    n2_by_kind = {k: {"median": [], "q25": [], "q75": [], "max": [], "n": []} for k in ALL_KINDS}
    bos_norm_traj = np.zeros(n_hs, dtype=np.float64)
    linf_flag_frac = np.zeros(n_hs, dtype=np.float64)
    argmax_dim_counts: Dict[int, int] = {}
    norm_recv_spearman = np.full(n_layers, np.nan, dtype=np.float64)
    R_norm_ratio_bos = np.full(n_hs, np.nan, dtype=np.float64)
    n2_layers: List[np.ndarray] = []

    for i in range(n_hs):
        hs = hidden[0][i][0].float().cpu().numpy()  # [L, d]
        n2 = token_l2_norms(hs)
        n2_layers.append(n2)
        bos_norm_traj[i] = float(n2[0])
        stats = per_kind_norm_stats(n2, masks)
        for k in ALL_KINDS:
            for f in ("median", "q25", "q75", "max", "n"):
                n2_by_kind[k][f].append(stats[k][f])
        if r_idx.size > 0:
            R_norm_ratio_bos[i] = float(np.median(n2[r_idx]) / max(n2[0], 1e-12))
        ma = massive_activation_stats(hs, mag_thresh=mag_thresh, ratio_thresh=ratio_thresh)
        linf_flag_frac[i] = float(ma["flag"].mean())
        for d in ma["argmax_dim"][ma["flag"]].tolist():
            argmax_dim_counts[int(d)] = argmax_dim_counts.get(int(d), 0) + 1

    # Norm (input to attn layer l = hidden_states[l]) vs received-attention rank
    # correlation, per attention layer.
    for li in range(n_layers):
        norm_recv_spearman[li] = spearman_rho(n2_layers[li], recv_per_key_by_layer[li])

    answer = tokenizer.decode(gen_token_ids.tolist(), skip_special_tokens=True).strip()

    return {
        "answer": answer,
        "prefill_len": prefill_len,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_hs": n_hs,
        "has_r": bool(r_idx.size > 0),
        "insert_pos": insert_pos,
        "n_queries": n_queries if is_reinspection else 0,
        "image_cols_count": int(image_cols.shape[0]),
        "answer_rows": qr.tolist(),
        "e_unif_col0": e_unif_col0,
        "row_sum_check_minmax": [float(row_sum_check.min()), float(row_sum_check.max())],
        # attention (per layer/head and per layer)
        "a0_lh": a0_lh,
        "budget_answer": {k: budget_answer[k] for k in ALL_KINDS},
        "entropy_l": ent_l,
        "ess_l": ess_l,
        "recv_ratio_bos_l": recv_ratio_bos_l,
        "budget_R_answer_l": budget_R_answer_l,
        "recv_R_max_l": recv_R_max_l,
        # hidden states
        "n2_by_kind": n2_by_kind,
        "bos_norm_traj": bos_norm_traj,
        "linf_flag_frac": linf_flag_frac,
        "argmax_dim_counts": argmax_dim_counts,
        "norm_recv_spearman": norm_recv_spearman,
        "R_norm_ratio_bos": R_norm_ratio_bos,
    }
