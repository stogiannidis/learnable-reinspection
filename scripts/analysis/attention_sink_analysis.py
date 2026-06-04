"""Attention-sink + per-token L2-norm analysis: frozen base VLM vs Re-Inspection.

Answers, with numbers and figures, on a set of VSR examples:
  Q1  Is there an attention sink in the frozen model?
        -> per-token hidden-state L2-norm spikes (massive activations) +
           attention mass piling onto BOS above the uniform-causal null.
  Q2  Do answer tokens attend to the image, or dump into the sink?
        -> per-row-normalised answer-query budget over {bos, image, text}.
  Q3  Does Re-Inspection fix / relocate it?
        -> paired frozen-vs-RI deltas (bos down, image up) and whether the
           learned R tokens become the new high-norm sink (R norms + received).

Pipeline (each model loaded once; mirrors select_and_visualize_vsr.py):
  1. load RI model -> scan shuffled VSR (cheap greedy, no attention) until
     n_correct + n_wrong are found (correctness under the RI model);
  2. capture RI sink metrics for the selected examples;
  3. free RI, load frozen, capture frozen metrics for the SAME examples (paired);
  4. aggregate (per-example-then-median + IQR), write metrics.json, figures,
     and a markdown verdict.

Run on a GPU (see k8s/attention_sink_analysis.yaml). bs=1 + eager attention +
single-tile InternVL are enforced for bit-exact, splice-comparable numbers.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
from typing import Dict, List, Optional

import numpy as np
import torch

from src.utils.sink_analysis import (
    ALL_KINDS,
    KIND_BOS,
    KIND_IMAGE,
    KIND_R,
    KIND_TEXT,
    capture_sink_example,
    generation_suffix_len,
    sink_rate,
)
from src.utils.token_attention import _process_inputs


def _load_cli():
    """Import visualize_token_attention.py for its config/processor/chat helpers."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "visualize_token_attention.py")
    spec = importlib.util.spec_from_file_location("viz_token_attn_cli", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CLI = _load_cli()


@torch.no_grad()
def generate_answer(model, processor, backend, config, image_path, question, is_ri,
                    build_chat, max_new_tokens: int = 8) -> str:
    """Greedy answer (no attention) under the same input pipeline as capture."""
    from src.evaluate import _generate_extra_kw
    device = next(model.parameters()).device
    tokenizer = getattr(processor, "tokenizer", processor)
    messages = build_chat(question, image_path=image_path, system_prompt=config.system_prompt)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _process_inputs(backend, processor, text, image_path, config)
    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                         **_generate_extra_kw(processor))
    seq = gen[0] if is_ri else gen[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(seq, skip_special_tokens=True).strip()


# --------------------------------------------------------------------------- #
# Aggregation                                                                   #
# --------------------------------------------------------------------------- #

# --- Verdict thresholds (named so the discrete conclusions are auditable) --- #
SINK_RATE_STRONG = 0.10      # >=10% of (layer,head) pairs qualify as sink heads (eps=0.3)
A0_STRONG = 0.30             # canonical "sink head": >=30% of mass received by BOS
A0_WEAK = 0.20
BOS_NORM_RATIO_STRONG = 5.0  # BOS L2 norm vs text-median at plateau (massive activation)
BOS_NORM_RATIO_WEAK = 3.0
REDISTRIB_DELTA = 0.02       # min paired median budget shift to call it redistribution
R_RELOCATE_NORM = 0.50       # median ||R||/||BOS|| at which R is a co-sink
R_RELOCATE_RECV_FRAC = 0.50  # strongest-R received >= this fraction of BOS max-received


def paired_budget_deltas(fr_records: List[dict], ri_records: List[dict]) -> Optional[dict]:
    """Per-example paired frozen-vs-RI budget deltas, matched on image id.

    Capture failures are dropped independently per condition, so the two
    record lists can cover different example subsets; pairing on ``img``
    restricts to the intersection (composition differences would otherwise be
    indistinguishable from a real bos→image redistribution). Returns medians
    + IQRs of the per-example deltas, or ``None`` if fewer than 3 pairs match.
    """
    fr_by_img = {r["img"]: r for r in fr_records}
    ri_by_img = {r["img"]: r for r in ri_records}
    common = sorted(set(fr_by_img) & set(ri_by_img))
    if len(common) < 3:
        return None

    def scalar(rec, kind):
        return float(np.asarray(rec["budget_answer"][kind]).mean())  # over layers+heads

    d_img = np.array([scalar(ri_by_img[i], KIND_IMAGE) - scalar(fr_by_img[i], KIND_IMAGE)
                      for i in common])
    d_bos = np.array([scalar(fr_by_img[i], KIND_BOS) - scalar(ri_by_img[i], KIND_BOS)
                      for i in common])

    def summarize(d):
        return dict(median=float(np.median(d)),
                    q25=float(np.percentile(d, 25)),
                    q75=float(np.percentile(d, 75)),
                    frac_positive=float((d > 0).mean()))

    return dict(
        n_matched=len(common),
        n_frozen_only=len(set(fr_by_img) - set(ri_by_img)),
        n_ri_only=len(set(ri_by_img) - set(fr_by_img)),
        delta_image=summarize(d_img),   # RI − frozen (positive = more image attention)
        delta_bos=summarize(d_bos),     # frozen − RI (positive = less BOS sink under RI)
    )


def _plateau_band(n: int):
    """Indices of the persistence plateau (exclude embeddings/first/last few)."""
    lo = min(3, max(1, n // 6))
    hi = max(lo + 1, n - 2)
    return lo, hi


def aggregate(records: List[dict]) -> dict:
    """Reduce a list of per-example records to one summary (per-example-then-median)."""
    if not records:
        return {}
    if len({r["has_r"] for r in records}) != 1:
        raise ValueError("aggregate() received mixed frozen/RI records — never pool conditions.")
    n_layers = records[0]["n_layers"]
    n_heads = records[0]["n_heads"]
    n_hs = records[0]["n_hs"]
    has_r = records[0]["has_r"]
    E = len(records)

    a0 = np.stack([r["a0_lh"] for r in records], axis=0)            # [E, L, H]
    a0_med = np.median(a0, axis=0)                                  # [L, H]
    # Same reducer as a0_med so sink_excess subtracts like-for-like.
    e_unif = float(np.median([r["e_unif_col0"] for r in records]))

    budget = {}
    for k in ALL_KINDS:
        b = np.stack([r["budget_answer"][k] for r in records], axis=0)  # [E, L, H]
        b_lh = np.median(b, axis=0)                                     # [L, H]
        per_layer = b_lh.mean(axis=1)                                   # mean over heads
        mol = float(per_layer.mean())
        # mean-over-layers per example, for IQR across examples
        mol_per_ex = b.mean(axis=(1, 2))
        budget[k] = dict(
            per_layer=per_layer.tolist(),
            mean_over_layers=mol,
            mol_iqr=[float(np.percentile(mol_per_ex, 25)), float(np.percentile(mol_per_ex, 75))],
        )

    # Uniform-causal image-share reference: pooled (mean counts, then divide)
    # rather than mean-of-ratios, to avoid a small Jensen bias. NOTE this null
    # is condition-specific: RI answer rows genuinely see n_queries more keys,
    # so RI's null is legitimately lower than frozen's — compare each
    # condition's image budget to ITS OWN null, never the nulls to each other.
    mean_img_cols = float(np.mean([r["image_cols_count"] for r in records]))
    mean_keys = float(np.mean([np.mean(np.asarray(r["answer_rows"], dtype=np.float64) + 1.0)
                               for r in records]))
    uniform_image_share = mean_img_cols / mean_keys

    ent = np.stack([r["entropy_l"] for r in records], axis=0)
    ess = np.stack([r["ess_l"] for r in records], axis=0)
    recv_ratio_bos = np.stack([r["recv_ratio_bos_l"] for r in records], axis=0)

    att = dict(
        a0_lh_median=a0_med.tolist(),
        a0_per_layer_mean=a0_med.mean(axis=1).tolist(),
        a0_per_layer_max=a0_med.max(axis=1).tolist(),
        a0_max_overall=float(a0_med.max()),
        e_unif_col0=e_unif,
        sink_excess_max=float(a0_med.max() - e_unif),
        sink_rate_0_3=sink_rate(a0_med, 0.3),
        sink_rate_0_6=sink_rate(a0_med, 0.6),
        sink_rate_0_3_triples=sink_rate(a0, 0.3),
        sink_rate_0_6_triples=sink_rate(a0, 0.6),
        entropy_per_layer=np.median(ent, axis=0).tolist(),
        ess_per_layer=np.median(ess, axis=0).tolist(),
        recv_ratio_bos_per_layer=np.median(recv_ratio_bos, axis=0).tolist(),
        budget_answer=budget,
        uniform_image_share=uniform_image_share,
    )
    if has_r:
        bR = np.stack([r["budget_R_answer_l"] for r in records], axis=0)
        recvR = np.stack([r["recv_R_max_l"] for r in records], axis=0)
        att["budget_R_answer_per_layer"] = np.median(bR, axis=0).tolist()
        att["budget_R_mean_over_layers"] = float(np.median(bR, axis=0).mean())
        att["recv_R_max_per_layer"] = np.median(recvR, axis=0).tolist()
        att["recv_R_max_overall"] = float(np.median(recvR, axis=0).max())

    # ---- hidden states ----
    bos_traj = np.stack([r["bos_norm_traj"] for r in records], axis=0)  # [E, n_hs]
    bos_med = np.median(bos_traj, axis=0)
    norm_by_kind = {}
    for k in ALL_KINDS:
        med = np.stack([np.asarray(r["n2_by_kind"][k]["median"], dtype=np.float64)
                        for r in records], axis=0)  # [E, n_hs]
        norm_by_kind[k] = np.nanmedian(med, axis=0)
    text_med = norm_by_kind[KIND_TEXT]
    bos_over_text = bos_med / np.maximum(text_med, 1e-12)

    lo, hi = _plateau_band(n_hs)
    plateau_bos_over_text = float(np.median(bos_over_text[lo:hi]))

    linf_flag = np.stack([r["linf_flag_frac"] for r in records], axis=0)
    spear = np.stack([r["norm_recv_spearman"] for r in records], axis=0)  # [E, n_layers]

    dim_counts: Dict[int, int] = {}
    for r in records:
        for d, c in r["argmax_dim_counts"].items():
            dim_counts[int(d)] = dim_counts.get(int(d), 0) + int(c)
    top_dims = sorted(dim_counts.items(), key=lambda kv: -kv[1])[:8]

    hid = dict(
        bos_norm_traj=bos_med.tolist(),
        norm_median_by_kind={k: norm_by_kind[k].tolist() for k in ALL_KINDS},
        bos_over_text_ratio_traj=bos_over_text.tolist(),
        plateau_band=[lo, hi],
        bos_over_text_ratio_plateau=plateau_bos_over_text,
        linf_flag_frac=np.median(linf_flag, axis=0).tolist(),
        massive_activation_top_dims=[[int(d), int(c)] for d, c in top_dims],
        norm_recv_spearman_per_layer=np.nanmedian(spear, axis=0).tolist(),
    )
    if has_r:
        rr = np.stack([r["R_norm_ratio_bos"] for r in records], axis=0)  # [E, n_hs]
        rr_med = np.nanmedian(rr, axis=0)
        hid["R_norm_ratio_bos_traj"] = rr_med.tolist()
        hid["R_norm_ratio_bos_plateau"] = float(np.nanmedian(rr_med[lo:hi]))

    return dict(n_examples=E, n_layers=n_layers, n_heads=n_heads, n_hs=n_hs,
                has_r=has_r, attention=att, hidden=hid)


# --------------------------------------------------------------------------- #
# Verdict                                                                       #
# --------------------------------------------------------------------------- #

def build_verdict(summ: Dict[str, dict], paired: Optional[dict] = None) -> str:
    """Plain-language answers to Q1/Q2/Q3 from the aggregated 'all' buckets.

    Note: "BOS" throughout means *position 0* (the StreamingLLM first-position
    sink). InternVL3's tokenizer has no literal BOS token — position 0 is
    ``<|im_start|>`` — but the first-position sink phenomenon is identical and
    the column is unmoved by the R splice.
    """
    fr = summ.get("frozen", {}).get("all")
    ri = summ.get("reinspection", {}).get("all")
    lines = ["# Attention-sink analysis verdict", ""]

    if fr:
        a = fr["attention"]; hd = fr["hidden"]
        a0_max = a["a0_max_overall"]
        ratio = hd["bos_over_text_ratio_plateau"]
        degenerate = (a0_max != a0_max) or (ratio != ratio)  # NaN guard
        q1 = ("UNDETERMINED (NaN inputs)" if degenerate
              else "STRONG" if (a["sink_rate_0_3"] >= SINK_RATE_STRONG and a0_max >= A0_STRONG
                                and ratio >= BOS_NORM_RATIO_STRONG)
              else "WEAK/PARTIAL" if (a0_max >= A0_WEAK or ratio >= BOS_NORM_RATIO_WEAK)
              else "NONE")
        lines += [
            f"## Q1 — Attention sink in the frozen model: **{q1}**",
            f"- sink_rate(ε=0.3) = {a['sink_rate_0_3']:.3f}, sink_rate(ε=0.6) = {a['sink_rate_0_6']:.3f} "
            f"(fraction of (layer,head) pairs dumping ≥ε of mass on position 0, on the "
            f"median-over-examples matrix; over all example×layer×head triples: "
            f"{a['sink_rate_0_3_triples']:.3f} / {a['sink_rate_0_6_triples']:.3f} — a large gap "
            f"means the sink is example-sparse)",
            f"- max BOS-received over (layer,head) = {a0_max:.3f} vs uniform-causal null "
            f"{a['e_unif_col0']:.3f} (excess {a['sink_excess_max']:.3f})",
            f"- BOS hidden-state norm / text-median at plateau = {ratio:.1f}× "
            f"(massive-activation signature; co-located with the sink)",
            "",
        ]
        b = a["budget_answer"]
        ignores = (b[KIND_IMAGE]["mean_over_layers"] < b[KIND_BOS]["mean_over_layers"]
                   and b[KIND_IMAGE]["mean_over_layers"] <= a["uniform_image_share"])
        lines += [
            f"## Q2 — Do answer tokens look at the image? **{'NO — dumps into sink' if ignores else 'PARTIALLY'}**",
            f"- answer-query budget (mean over layers): BOS {b[KIND_BOS]['mean_over_layers']:.3f}, "
            f"image {b[KIND_IMAGE]['mean_over_layers']:.3f}, text {b[KIND_TEXT]['mean_over_layers']:.3f}",
            f"- uniform-causal image share (reference) = {a['uniform_image_share']:.3f} "
            f"→ image budget is {'BELOW' if b[KIND_IMAGE]['mean_over_layers'] <= a['uniform_image_share'] else 'above'} the geometry null",
            "",
        ]

    if fr and ri:
        af, ar = fr["attention"], ri["attention"]
        budget_R = ar.get("budget_R_mean_over_layers", 0.0)
        recvR = ar.get("recv_R_max_overall", 0.0)
        rnorm = ri["hidden"].get("R_norm_ratio_bos_plateau", float("nan"))
        relocated = (recvR >= af["a0_max_overall"] * R_RELOCATE_RECV_FRAC) \
            or (rnorm == rnorm and rnorm >= R_RELOCATE_NORM)

        if paired:
            di, db = paired["delta_image"], paired["delta_bos"]
            # Paired per-example deltas; require the IQR not to straddle 0 so a
            # composition artifact cannot pass as redistribution.
            redistributed = (di["median"] > REDISTRIB_DELTA and di["q25"] > 0
                            and db["median"] > REDISTRIB_DELTA and db["q25"] > 0)
            delta_lines = [
                f"- paired Δ image budget (RI − frozen, n={paired['n_matched']}): "
                f"median {di['median']:+.3f} [IQR {di['q25']:+.3f}, {di['q75']:+.3f}], "
                f"{di['frac_positive']:.0%} of examples positive",
                f"- paired Δ BOS budget (frozen − RI): median {db['median']:+.3f} "
                f"[IQR {db['q25']:+.3f}, {db['q75']:+.3f}], {db['frac_positive']:.0%} positive",
                (f"- coverage: {paired['n_frozen_only']} frozen-only / {paired['n_ri_only']} "
                 f"RI-only examples excluded from pairing" if (paired["n_frozen_only"]
                 or paired["n_ri_only"]) else None),
            ]
        else:
            bf, br = af["budget_answer"], ar["budget_answer"]
            d_img = br[KIND_IMAGE]["mean_over_layers"] - bf[KIND_IMAGE]["mean_over_layers"]
            d_bos = bf[KIND_BOS]["mean_over_layers"] - br[KIND_BOS]["mean_over_layers"]
            redistributed = (d_img > REDISTRIB_DELTA and d_bos > REDISTRIB_DELTA)
            delta_lines = [
                f"- UNPAIRED Δ image budget (RI − frozen) = {d_img:+.3f}; "
                f"Δ BOS budget (frozen − RI) = {d_bos:+.3f} "
                f"(too few matched examples for a paired comparison — treat with caution)",
            ]

        verdict = ("FIX (redistributes bos→image)" if redistributed and not relocated
                   else "RELOCATES sink onto R tokens" if relocated and not redistributed
                   else "BOTH redistributes AND relocates" if redistributed and relocated
                   else "NO measurable change")
        lines += [
            f"## Q3 — Does Re-Inspection change the sink? **{verdict}**",
            *[ln for ln in delta_lines if ln],
            f"- answer mass on R tokens = {budget_R:.3f}; strongest R received/uniform = {recvR:.2f} "
            f"(vs BOS max-received {af['a0_max_overall']:.3f})",
            f"- median ||R||₂ / ||BOS||₂ at plateau = {rnorm:.2f} "
            f"({'R approaches/exceeds BOS norm → activation-level relocation' if rnorm==rnorm and rnorm>=R_RELOCATE_NORM else 'R below BOS norm'})",
            "",
        ]
        # correctness split, if available
        ric, riw = summ.get("reinspection", {}).get("correct"), summ.get("reinspection", {}).get("wrong")
        if ric and riw:
            lines += [
                "### RI correct vs wrong (mechanism ↔ behaviour)",
                f"- image budget: correct {ric['attention']['budget_answer'][KIND_IMAGE]['mean_over_layers']:.3f} "
                f"vs wrong {riw['attention']['budget_answer'][KIND_IMAGE]['mean_over_layers']:.3f}",
                f"- R budget: correct {ric['attention'].get('budget_R_mean_over_layers', 0):.3f} "
                f"vs wrong {riw['attention'].get('budget_R_mean_over_layers', 0):.3f}",
                "",
            ]
    return "\n".join([ln for ln in lines if ln is not None])


# --------------------------------------------------------------------------- #
# Plots                                                                         #
# --------------------------------------------------------------------------- #

def render_plots(summ: Dict[str, dict], out_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conds = [c for c in ("frozen", "reinspection") if c in summ and summ[c].get("all")]
    if not conds:
        return
    colors = {"frozen": "tab:blue", "reinspection": "tab:red"}

    def save(fig, name):
        os.makedirs(out_dir, exist_ok=True)
        p = os.path.join(out_dir, name)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        fig.savefig(p.replace(".png", ".pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {p}", flush=True)

    # 1. Per-(layer,head) BOS-received heatmaps.
    fig, axes = plt.subplots(1, len(conds), figsize=(6 * len(conds), 5), squeeze=False)
    for c, cond in enumerate(conds):
        m = np.asarray(summ[cond]["all"]["attention"]["a0_lh_median"])
        im = axes[0, c].imshow(m, aspect="auto", cmap="inferno", vmin=0, vmax=1)
        axes[0, c].set_title(f"{cond}: BOS-received a0[layer,head]")
        axes[0, c].set_xlabel("head"); axes[0, c].set_ylabel("layer")
        fig.colorbar(im, ax=axes[0, c], fraction=0.046)
    fig.suptitle("Attention mass received by BOS per (layer, head) — sink signature", y=1.02)
    save(fig, "a0_layerhead_heatmap.png")

    # 2. Per-layer BOS-received mean & max + uniform null.
    fig, ax = plt.subplots(figsize=(8, 5))
    for cond in conds:
        a = summ[cond]["all"]["attention"]
        ax.plot(a["a0_per_layer_mean"], color=colors[cond], label=f"{cond} mean-over-heads")
        ax.plot(a["a0_per_layer_max"], color=colors[cond], ls="--", label=f"{cond} max-over-heads")
        ax.axhline(a["e_unif_col0"], color=colors[cond], ls=":", alpha=0.6)
    ax.set_xlabel("layer"); ax.set_ylabel("BOS-received mass"); ax.legend(fontsize=8)
    ax.set_title("BOS attention sink vs layer (dotted = uniform-causal null)")
    save(fig, "bos_received_per_layer.png")

    # 3. sink_rate bars.
    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(len(conds)); width = 0.35
    for i, eps in enumerate(("0_3", "0_6")):
        vals = [summ[c]["all"]["attention"][f"sink_rate_{eps}"] for c in conds]
        ax.bar(x + (i - 0.5) * width, vals, width, label=f"ε={eps.replace('_', '.')}")
    ax.set_xticks(x); ax.set_xticklabels(conds); ax.set_ylabel("fraction of sink heads")
    ax.legend(); ax.set_title("sink_rate (fraction of layer,head with BOS mass ≥ ε)")
    save(fig, "sink_rate_bars.png")

    # 4. Answer-query budget stacked vs layer.
    fig, axes = plt.subplots(1, len(conds), figsize=(6 * len(conds), 5), squeeze=False)
    for c, cond in enumerate(conds):
        a = summ[cond]["all"]["attention"]
        L = len(a["budget_answer"][KIND_BOS]["per_layer"])
        xs = np.arange(L)
        stacks = [KIND_BOS, KIND_IMAGE, KIND_TEXT] + ([KIND_R] if a["budget_answer"][KIND_R]["mean_over_layers"] > 0 else [])
        bottom = np.zeros(L)
        palette = {KIND_BOS: "0.3", KIND_IMAGE: "tab:green", KIND_TEXT: "tab:orange", KIND_R: "tab:red"}
        for k in stacks:
            v = np.asarray(a["budget_answer"][k]["per_layer"])
            axes[0, c].fill_between(xs, bottom, bottom + v, label=k, color=palette[k], alpha=0.8)
            bottom += v
        axes[0, c].axhline(a["uniform_image_share"], color="k", ls=":", label="uniform image share")
        axes[0, c].set_title(f"{cond}: answer-query budget"); axes[0, c].set_xlabel("layer")
        axes[0, c].set_ylim(0, 1); axes[0, c].legend(fontsize=8)
    fig.suptitle("Where do answer tokens attend? {bos, image, text, R} per layer", y=1.02)
    save(fig, "answer_budget_stacked.png")

    # 5. Hidden-state norm trajectories.
    fig, ax = plt.subplots(figsize=(8, 5))
    for cond in conds:
        hd = summ[cond]["all"]["hidden"]
        ax.plot(hd["bos_norm_traj"], color=colors[cond], label=f"{cond} BOS")
        ax.plot(hd["norm_median_by_kind"][KIND_IMAGE], color=colors[cond], ls="--", alpha=0.7, label=f"{cond} image med")
        ax.plot(hd["norm_median_by_kind"][KIND_TEXT], color=colors[cond], ls=":", alpha=0.7, label=f"{cond} text med")
        if hd.get("norm_median_by_kind", {}).get(KIND_R) and summ[cond]["all"]["has_r"]:
            ax.plot(hd["norm_median_by_kind"][KIND_R], color="tab:purple", label=f"{cond} R med")
    ax.set_xlabel("hidden_states index (0 = embeddings)"); ax.set_ylabel("median L2 norm")
    ax.set_yscale("log"); ax.legend(fontsize=8)
    ax.set_title("Per-token hidden-state L2 norm — massive-activation signature")
    save(fig, "hidden_norm_trajectory.png")


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #

def _jsonify(o):
    if isinstance(o, dict):
        return {k: _jsonify(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonify(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonify(o.tolist())
    if isinstance(o, (np.floating,)):
        o = float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, float) and o != o:  # NaN -> null (strict-JSON safe)
        return None
    return o


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default="internvl3", choices=["internvl3", "llava_next"])
    p.add_argument("--checkpoint_dir", default="models/internvl3/s2_internvl/stage2/epoch_2")
    p.add_argument("--lora_checkpoint_dir", default="models/internvl3/s2_internvl/stage2/epoch_2")
    p.add_argument("--output_dir", default="outputs/sink_analysis")
    p.add_argument("--vsr_file", default="/data/datasets/vsr/test.jsonl")
    p.add_argument("--image_root", default="/data/datasets/vsr/images")
    p.add_argument("--n_correct", type=int, default=20)
    p.add_argument("--n_wrong", type=int, default=20)
    p.add_argument("--max_scan", type=int, default=300)
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--q_min", type=int, default=2)
    p.add_argument("--mag_thresh", type=float, default=100.0)
    p.add_argument("--ratio_thresh", type=float, default=1000.0)
    p.add_argument("--conditions", default="frozen,reinspection")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    overrides = dict(backend=args.backend, attn_implementation="eager", bf16=True,
                     checkpoint_dir=args.checkpoint_dir, lora_checkpoint_dir=args.lora_checkpoint_dir)
    config = CLI._config_from_backend_yaml(args.backend, overrides)
    processor = CLI._load_processor(args.backend, config)
    build_chat = CLI._build_chat_fn(args.backend)
    tokenizer = getattr(processor, "tokenizer", processor)
    suffix_len = generation_suffix_len(tokenizer)
    print(f"generation suffix length = {suffix_len}", flush=True)

    from src.evaluate import load_condition_model, match_answer

    # Read + shuffle VSR examples whose image exists.
    examples = []
    for line in open(args.vsr_file, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        img = os.path.basename(r.get("image", ""))
        if img and os.path.isfile(os.path.join(args.image_root, img)):
            examples.append({"img": img, "question": r["question"], "gt": str(r["answer"]).strip()})
    random.seed(args.seed)
    random.shuffle(examples)

    # ---- Phase 1: scan with RI model for n_correct + n_wrong (by RI correctness) ----
    print(f"\n=== Phase 1: scan ({args.backend}) ===", flush=True)
    ri_model, is_ri = load_condition_model(
        args.backend, "reinspection", config, processor,
        checkpoint_dir=args.checkpoint_dir, lora_checkpoint_dir=args.lora_checkpoint_dir,
        attn_implementation="eager")
    ri_model.eval()

    correct, wrong = [], []
    for ex in examples[: args.max_scan]:
        if len(correct) >= args.n_correct and len(wrong) >= args.n_wrong:
            break
        path = os.path.join(args.image_root, ex["img"])
        try:
            pred = generate_answer(ri_model, processor, args.backend, config, path,
                                   ex["question"], is_ri, build_chat, args.max_new_tokens)
        except Exception as e:
            print(f"  scan skip {ex['img']}: {e}", flush=True)
            continue
        ok = match_answer(pred, ex["gt"])
        ex = {**ex, "ri_pred": pred, "ri_correct": ok}
        bucket, cap = (correct, args.n_correct) if ok else (wrong, args.n_wrong)
        if len(bucket) < cap:
            bucket.append(ex)
            print(f"  [{len(correct)}c/{len(wrong)}w] {ex['img']} GT={ex['gt']} RI={pred!r} "
                  f"{'OK' if ok else 'WRONG'}", flush=True)
    selected = correct + wrong
    if not selected:
        raise SystemExit("No examples selected.")
    print(f"\nselected {len(correct)} correct + {len(wrong)} wrong", flush=True)

    # ---- Phase 2: capture per-condition sink metrics for the SAME examples ----
    def capture_all(model, is_ri_flag):
        recs = []
        for ex in selected:
            path = os.path.join(args.image_root, ex["img"])
            try:
                rec = capture_sink_example(
                    model, processor, config, image_path=path, question=ex["question"],
                    is_reinspection=is_ri_flag, build_chat_messages=build_chat,
                    suffix_len=suffix_len, backend=args.backend,
                    max_new_tokens=args.max_new_tokens, q_min=args.q_min,
                    mag_thresh=args.mag_thresh, ratio_thresh=args.ratio_thresh)
                rec["ri_correct"] = ex["ri_correct"]
                rec["img"] = ex["img"]
                recs.append(rec)
            except Exception as e:
                print(f"  capture skip {ex['img']}: {e}", flush=True)
        print(f"  captured {len(recs)}/{len(selected)} examples", flush=True)
        return recs

    captures: Dict[str, List[dict]] = {}
    if "reinspection" in conditions:
        print("\n=== Phase 2: RI capture ===", flush=True)
        captures["reinspection"] = capture_all(ri_model, True)
    del ri_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if "frozen" in conditions:
        print("\n=== Phase 3: frozen capture ===", flush=True)
        fr_model, _ = load_condition_model(args.backend, "frozen", config, processor,
                                           attn_implementation="eager")
        fr_model.eval()
        captures["frozen"] = capture_all(fr_model, False)
        del fr_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Phase 4: aggregate (all / correct / wrong), verdict, plots ----
    print("\n=== Phase 4: aggregate ===", flush=True)
    summ: Dict[str, dict] = {}
    for cond, recs in captures.items():
        buckets = {"all": aggregate(recs)}
        # Correctness was measured under the RI model only (phase 1), so the
        # correct/wrong stratification is meaningful only for reinspection;
        # labelling frozen metrics by RI correctness would invite a confounded
        # reading ("frozen ignores the image when wrong" on RI-wrong items).
        if cond == "reinspection":
            buckets["correct"] = aggregate([r for r in recs if r["ri_correct"]])
            buckets["wrong"] = aggregate([r for r in recs if not r["ri_correct"]])
        summ[cond] = buckets

    # Paired frozen-vs-RI deltas on the intersection of successfully-captured
    # examples — the Q3 redistribution claim rides on these, not on differences
    # of independently aggregated condition scalars.
    paired = None
    if "frozen" in captures and "reinspection" in captures:
        paired = paired_budget_deltas(captures["frozen"], captures["reinspection"])

    os.makedirs(args.output_dir, exist_ok=True)
    meta = dict(backend=args.backend, checkpoint_dir=args.checkpoint_dir,
                n_correct=len(correct), n_wrong=len(wrong), suffix_len=suffix_len,
                q_min=args.q_min, mag_thresh=args.mag_thresh, ratio_thresh=args.ratio_thresh,
                eval_batch_size=1, attn_implementation="eager")
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(_jsonify({"meta": meta, "summary": summ, "paired": paired}), f, indent=2)
    print(f"  wrote {os.path.join(args.output_dir, 'metrics.json')}", flush=True)

    verdict = build_verdict(summ, paired=paired)
    with open(os.path.join(args.output_dir, "VERDICT.md"), "w", encoding="utf-8") as f:
        f.write(verdict)
    print("\n" + verdict, flush=True)

    render_plots(summ, args.output_dir)
    print(f"\nDone → {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
