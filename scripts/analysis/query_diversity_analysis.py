"""Per-query attention diversity diagnostic: distinct correspondences or collapse?

The removed Stage-1 attention KL used to broadcast ONE bbox target to
all n_selector_queries supervised queries — its optimum is all of them producing
identical attention maps. This script measures, on real VSR samples, whether the
trained module's per-query attention rows A_vis (N_q, N_vis) actually ended up
distinct, before deciding whether the loss needs a diversity-aware redesign:

  Q1  Are the supervised selector queries (0..k-1) near-duplicates?
        -> pairwise JSD + cosine within the selector block.
  Q2  Are the free queries (k..Q-1) informative, or near-uniform mush?
        -> per-query normalized entropy H/ln(V) (1.0 == uniform).
  Q3  How many effectively distinct patterns do the Q rows span?
        -> participation ratio (Σs²)²/Σs⁴ of the row matrix's singular values.

Outputs per checkpoint: attn_raw.npz (raw per-sample (Q, V) stacks), metrics.json,
VERDICT.md, fig_jsd_heatmap.png, fig_entropy.png, fig_query_maps_*.png.

GPU-free re-aggregation/re-render:  --from_npz outputs/.../attn_raw.npz
CPU sanity check of the metric math: --selfcheck
GPU capture otherwise (see k8s/query_diversity.yaml).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

EPS = 1e-12

# --- Verdict thresholds (named so the discrete conclusions are auditable) ---- #
SEL_JSD_COLLAPSED = 0.10   # median within-selector normalized JSD below -> COLLAPSED
SEL_JSD_DIVERSE = 0.30     # above -> DIVERSE; between -> PARTIAL
NEAR_UNIFORM_H = 0.95      # H/ln(V) above this counts as "near-uniform mush"
PR_COLLAPSED_FRAC = 0.15   # participation ratio / Q below -> rows span few patterns
PR_DIVERSE_FRAC = 0.40


# --------------------------------------------------------------------------- #
# Pure metric math (numpy only — unit-testable, reusable from NPZ)            #
# --------------------------------------------------------------------------- #

def row_normalize(A: np.ndarray) -> np.ndarray:
    """Clamp + renormalize each query row to a distribution over V."""
    P = np.clip(np.asarray(A, dtype=np.float64), EPS, None)
    return P / P.sum(axis=-1, keepdims=True)


def pairwise_jsd(P: np.ndarray) -> np.ndarray:
    """(Q, Q) Jensen-Shannon divergence between rows, normalized to [0, 1]."""
    logP = np.log(P)
    H = -(P * logP).sum(-1)                                   # (Q,)
    M = 0.5 * (P[:, None, :] + P[None, :, :])                 # (Q, Q, V)
    Hm = -(M * np.log(np.clip(M, EPS, None))).sum(-1)         # (Q, Q)
    jsd = Hm - 0.5 * (H[:, None] + H[None, :])
    return np.clip(jsd / np.log(2.0), 0.0, 1.0)


def pairwise_cosine(P: np.ndarray) -> np.ndarray:
    """(Q, Q) cosine similarity between rows."""
    X = P / (np.linalg.norm(P, axis=-1, keepdims=True) + EPS)
    return np.clip(X @ X.T, -1.0, 1.0)


def normalized_entropy(P: np.ndarray) -> np.ndarray:
    """(Q,) per-row entropy normalized by ln(V): 0 = one-hot, 1 = uniform."""
    H = -(P * np.log(P)).sum(-1)
    return H / np.log(P.shape[-1])


def participation_ratio(P: np.ndarray) -> float:
    """Effective number of distinct row patterns: (Σs²)² / Σs⁴ of the SVD.

    1.0 when all rows are identical (rank 1), up to Q when orthogonal.
    """
    s = np.linalg.svd(P, compute_uv=False)
    s2 = s ** 2
    denom = float((s2 ** 2).sum())
    return float(s2.sum() ** 2 / denom) if denom > 0 else 1.0


def _offdiag_block_mean(mat: np.ndarray, rows: slice, cols: slice, exclude_diag: bool) -> float:
    blk = mat[rows, cols]
    if exclude_diag:
        n = blk.shape[0]
        mask = ~np.eye(n, dtype=bool)
        return float(blk[mask].mean()) if n > 1 else float("nan")
    return float(blk.mean())


def metrics_for_sample(A: np.ndarray, k_sel: int) -> Dict[str, float]:
    """All scalar diversity metrics for one sample's (Q, V) attention stack."""
    P = row_normalize(A)
    Q = P.shape[0]
    jsd = pairwise_jsd(P)
    cos = pairwise_cosine(P)
    ent = normalized_entropy(P)
    sel, free = slice(0, k_sel), slice(k_sel, Q)
    return {
        "jsd_sel_within": _offdiag_block_mean(jsd, sel, sel, True),
        "jsd_free_within": _offdiag_block_mean(jsd, free, free, True),
        "jsd_cross": _offdiag_block_mean(jsd, sel, free, False),
        "jsd_all": _offdiag_block_mean(jsd, slice(0, Q), slice(0, Q), True),
        "cos_sel_within": _offdiag_block_mean(cos, sel, sel, True),
        "cos_free_within": _offdiag_block_mean(cos, free, free, True),
        "ent_sel_mean": float(ent[sel].mean()),
        "ent_free_mean": float(ent[free].mean()),
        "frac_free_near_uniform": float((ent[free] > NEAR_UNIFORM_H).mean()),
        "participation_ratio": participation_ratio(P),
    }


def aggregate(per_sample: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """median + IQR over samples for every scalar metric."""
    out: Dict[str, Dict[str, float]] = {}
    for key in per_sample[0]:
        vals = np.array([m[key] for m in per_sample], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        out[key] = {
            "median": float(np.median(vals)),
            "q25": float(np.percentile(vals, 25)),
            "q75": float(np.percentile(vals, 75)),
        }
    return out


def selfcheck() -> None:
    """Synthetic sanity checks for the metric math (CPU, no model)."""
    rng = np.random.default_rng(0)
    V, Q = 64, 16
    # identical rows -> JSD 0, cosine 1, PR 1
    base = row_normalize(rng.random(V))
    P_same = np.tile(base, (Q, 1))
    assert pairwise_jsd(P_same).max() < 1e-6
    assert pairwise_cosine(P_same).min() > 1 - 1e-9
    assert abs(participation_ratio(row_normalize(P_same)) - 1.0) < 1e-6
    # disjoint one-hots -> normalized JSD 1, cosine ~0, PR Q
    P_hot = row_normalize(np.eye(Q, V))
    j = pairwise_jsd(P_hot)
    assert j[~np.eye(Q, dtype=bool)].min() > 0.99
    assert abs(participation_ratio(P_hot) - Q) < 1e-3
    # uniform rows -> entropy 1
    P_uni = row_normalize(np.ones((Q, V)))
    assert normalized_entropy(P_uni).min() > 1 - 1e-9
    # one-hot rows -> entropy ~0
    assert normalized_entropy(P_hot).max() < 0.05
    m = metrics_for_sample(np.tile(base, (Q, 1)), k_sel=4)
    assert m["jsd_sel_within"] < 1e-6 and m["participation_ratio"] < 1.001
    print("selfcheck OK: identical->collapsed, one-hots->diverse, uniform->H=1", flush=True)


# --------------------------------------------------------------------------- #
# Verdict + figures                                                            #
# --------------------------------------------------------------------------- #

def build_verdict(agg: Dict[str, Dict[str, float]], n_samples: int, Q: int, k_sel: int,
                  label: str) -> str:
    sel_jsd = agg["jsd_sel_within"]["median"]
    if sel_jsd < SEL_JSD_COLLAPSED:
        q1 = f"COLLAPSED — the {k_sel} supervised queries are near-duplicates"
    elif sel_jsd < SEL_JSD_DIVERSE:
        q1 = f"PARTIAL — the {k_sel} supervised queries overlap heavily but are not identical"
    else:
        q1 = f"DIVERSE — the {k_sel} supervised queries attend to distinct patterns"

    mush = agg["frac_free_near_uniform"]["median"]
    ent_free = agg["ent_free_mean"]["median"]
    if mush > 0.5:
        q2 = f"MUSH — {mush:.0%} of free queries are near-uniform (H/lnV > {NEAR_UNIFORM_H})"
    elif ent_free > 0.85:
        q2 = "DIFFUSE — free queries are broad but not strictly uniform"
    else:
        q2 = "INFORMATIVE — free queries carry sharp, non-uniform attention"

    pr = agg["participation_ratio"]["median"]
    frac = pr / Q
    if frac < PR_COLLAPSED_FRAC:
        q3 = f"LOW — ~{pr:.1f} effective patterns out of {Q} queries"
    elif frac < PR_DIVERSE_FRAC:
        q3 = f"MODERATE — ~{pr:.1f} effective patterns out of {Q} queries"
    else:
        q3 = f"HIGH — ~{pr:.1f} effective patterns out of {Q} queries"

    def fmt(key):
        a = agg[key]
        return f"{a['median']:.3f} [{a['q25']:.3f}, {a['q75']:.3f}]"

    lines = [
        f"# Query-diversity verdict — {label}",
        "",
        f"{n_samples} VSR samples, Q={Q} queries, k_sel={k_sel} supervised. "
        "All values: median [IQR] over samples. JSD normalized to [0,1].",
        "",
        f"**Q1 (selector duplication): {q1}.**",
        f"- within-selector JSD: {fmt('jsd_sel_within')}  (collapse < {SEL_JSD_COLLAPSED}, diverse > {SEL_JSD_DIVERSE})",
        f"- within-selector cosine: {fmt('cos_sel_within')}",
        "",
        f"**Q2 (free-query informativeness): {q2}.**",
        f"- free-query normalized entropy: {fmt('ent_free_mean')}  (selector: {fmt('ent_sel_mean')})",
        f"- fraction of free queries near-uniform: {fmt('frac_free_near_uniform')}",
        "",
        f"**Q3 (effective distinct patterns): {q3}.**",
        f"- participation ratio: {fmt('participation_ratio')} of Q={Q}",
        f"- all-pairs JSD: {fmt('jsd_all')}; selector-vs-free JSD: {fmt('jsd_cross')}",
        "",
        "## Interpretation",
        "- Q1 COLLAPSED + Q3 LOW  -> the broadcast-target KL did its anti-diversity work;"
        " a coverage+diversity (or matching/OT) redesign is justified.",
        "- Q1 DIVERSE + Q3 HIGH   -> the loss is not the bottleneck; look elsewhere"
        " (insertion position, W_up bottleneck, Stage-2 washout).",
        "- Q2 MUSH on 56/64 free queries means most R tokens carry near-zero"
        " image-specific signal regardless of what the supervised 8 do.",
    ]
    return "\n".join(lines) + "\n"


def render_figures(attn: np.ndarray, k_sel: int, h: int, w: int, out_dir: str,
                   image_names: List[str], label: str) -> None:
    """JSD heatmap (mean over samples), entropy bars, per-query map grids."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N, Q, V = attn.shape
    P_all = np.stack([row_normalize(attn[i]) for i in range(N)])
    jsd_mean = np.mean([pairwise_jsd(P) for P in P_all], axis=0)
    ent_med = np.median(np.stack([normalized_entropy(P) for P in P_all]), axis=0)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(jsd_mean, cmap="viridis", vmin=0, vmax=max(0.5, jsd_mean.max()))
    ax.axhline(k_sel - 0.5, color="w", lw=1.2, ls="--")
    ax.axvline(k_sel - 0.5, color="w", lw=1.2, ls="--")
    ax.set_title(f"Mean pairwise JSD between query rows — {label}\n"
                 f"(dashes mark the {k_sel} supervised selector queries)")
    ax.set_xlabel("query"); ax.set_ylabel("query")
    fig.colorbar(im, ax=ax, label="JSD (0 = identical, 1 = disjoint)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_jsd_heatmap.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 3.5))
    colors = ["#dc2626" if q < k_sel else "#3b82f6" for q in range(Q)]
    ax.bar(range(Q), ent_med, color=colors)
    ax.axhline(NEAR_UNIFORM_H, color="k", lw=1, ls=":", label=f"near-uniform ({NEAR_UNIFORM_H})")
    ax.set_xlabel("query index"); ax.set_ylabel("H / ln(V)")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Per-query normalized attention entropy (median over {N} samples) — {label}\n"
                 "red = supervised selector queries, blue = free queries")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_entropy.png"), dpi=150)
    plt.close(fig)

    # Per-query maps for the first samples: selector row + evenly-spaced free row.
    n_show = min(2, N)
    free_idx = np.linspace(k_sel, Q - 1, num=min(8, Q - k_sel)).astype(int)
    sel_idx = np.arange(min(8, k_sel))
    for i in range(n_show):
        P = P_all[i]
        if V < h * w:
            continue
        fig, axes = plt.subplots(2, max(len(sel_idx), len(free_idx)),
                                 figsize=(2.1 * max(len(sel_idx), len(free_idx)), 4.6))
        for row, (idxs, name) in enumerate([(sel_idx, "sel"), (free_idx, "free")]):
            for col in range(axes.shape[1]):
                ax = axes[row, col]
                ax.axis("off")
                if col >= len(idxs):
                    continue
                q = int(idxs[col])
                ax.imshow(P[q, : h * w].reshape(h, w), cmap="inferno")
                ax.set_title(f"{name} q{q}", fontsize=9)
        fig.suptitle(f"Per-query attention maps — {label} — {image_names[i]}", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"fig_query_maps_{i}.png"), dpi=150)
        plt.close(fig)


def analyze_and_write(attn: np.ndarray, k_sel: int, h: int, w: int, out_dir: str,
                      image_names: List[str], label: str) -> None:
    N, Q, V = attn.shape
    per_sample = [metrics_for_sample(attn[i], k_sel) for i in range(N)]
    agg = aggregate(per_sample)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"label": label, "n_samples": N, "n_queries": Q, "n_selector": k_sel,
                   "n_vision_tokens": V, "aggregate": agg, "per_sample": per_sample},
                  f, indent=2)
    verdict = build_verdict(agg, N, Q, k_sel, label)
    with open(os.path.join(out_dir, "VERDICT.md"), "w", encoding="utf-8") as f:
        f.write(verdict)
    render_figures(attn, k_sel, h, w, out_dir, image_names, label)
    print("\n" + verdict, flush=True)
    print(f"wrote {out_dir}/{{metrics.json, VERDICT.md, fig_*.png}}", flush=True)


# --------------------------------------------------------------------------- #
# GPU capture                                                                   #
# --------------------------------------------------------------------------- #

def _load_cli():
    """Import visualize_token_attention.py for its config/processor/chat helpers."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "visualize_token_attention.py")
    spec = importlib.util.spec_from_file_location("viz_token_attn_cli", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def capture(args) -> Tuple[np.ndarray, List[str], int, int, int]:
    """Load the RI model once and capture A_vis (Q, V) on n_samples VSR images."""
    import torch
    from src.utils.token_attention import _process_inputs, plan_vision_grid

    CLI = _load_cli()
    overrides = dict(backend=args.backend, attn_implementation="eager", bf16=True,
                     checkpoint_dir=args.checkpoint_dir,
                     lora_checkpoint_dir=args.lora_checkpoint_dir)
    config = CLI._config_from_backend_yaml(args.backend, overrides)
    processor = CLI._load_processor(args.backend, config)
    build_chat = CLI._build_chat_fn(args.backend)

    from src.evaluate import _generate_extra_kw, load_condition_model

    model, is_ri = load_condition_model(
        args.backend, "reinspection", config, processor,
        checkpoint_dir=args.checkpoint_dir,
        lora_checkpoint_dir=args.lora_checkpoint_dir,
        attn_implementation="eager")
    if not is_ri:
        raise SystemExit("load_condition_model did not return a Re-Inspection model.")
    model.eval()
    device = next(model.parameters()).device
    h, w, _ = plan_vision_grid(args.backend, model)
    k_sel = int(config.n_selector_queries)

    examples = []
    for line in open(args.vsr_file, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        img = os.path.basename(r.get("image", ""))
        if img and os.path.isfile(os.path.join(args.image_root, img)):
            examples.append({"img": img, "question": r["question"]})
    random.seed(args.seed)
    random.shuffle(examples)

    stacks: List[np.ndarray] = []
    names: List[str] = []
    with torch.no_grad():
        for ex in examples[: args.max_scan]:
            if len(stacks) >= args.n_samples:
                break
            path = os.path.join(args.image_root, ex["img"])
            try:
                messages = build_chat(ex["question"], image_path=path,
                                      system_prompt=config.system_prompt)
                text = processor.apply_chat_template(messages, tokenize=False,
                                                     add_generation_prompt=True)
                inputs = _process_inputs(args.backend, processor, text, path, config)
                inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                          for k, v in inputs.items()}
                model.generate(**inputs, max_new_tokens=1, do_sample=False,
                               **_generate_extra_kw(processor))
                _, attn_vis = model.get_attention_maps()
                if attn_vis is None:
                    print(f"  skip {ex['img']}: no attn_vis captured", flush=True)
                    continue
                A = attn_vis[0].float().cpu().numpy()       # (Q, V)
                stacks.append(A.astype(np.float32))
                names.append(ex["img"])
                print(f"  [{len(stacks)}/{args.n_samples}] {ex['img']} A_vis={A.shape}", flush=True)
            except Exception as e:  # capture failures shouldn't kill the sweep
                print(f"  skip {ex['img']}: {e}", flush=True)

    if not stacks:
        raise SystemExit("No samples captured.")
    V_min = min(a.shape[-1] for a in stacks)
    attn = np.stack([a[:, :V_min] for a in stacks])          # (N, Q, V)
    return attn, names, h, w, k_sel


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default="internvl3", choices=["internvl3", "llava_next"])
    p.add_argument("--checkpoint_dir", default=None, help="Re-Inspection module dir (stage1/2).")
    p.add_argument("--lora_checkpoint_dir", default=None,
                   help="Stage-2 dir (module + LoRA). Pair with the SAME checkpoint.")
    p.add_argument("--vsr_file", default="/data/datasets/vsr/test.jsonl")
    p.add_argument("--image_root", default="/data/datasets/vsr/images")
    p.add_argument("--n_samples", type=int, default=32)
    p.add_argument("--max_scan", type=int, default=200)
    p.add_argument("--output_dir", default="outputs/query_diversity")
    p.add_argument("--label", default=None, help="Verdict/figure label (default: output dir name).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--from_npz", default=None,
                   help="Re-aggregate + re-render from an existing attn_raw.npz — no GPU needed.")
    p.add_argument("--selfcheck", action="store_true",
                   help="Run CPU sanity checks of the metric math and exit.")
    args = p.parse_args()

    if args.selfcheck:
        selfcheck()
        return

    if args.from_npz:
        data = np.load(args.from_npz, allow_pickle=False)
        attn = np.asarray(data["attn"])
        names = [str(x) for x in np.asarray(data["image_names"]).tolist()]
        h, w = int(data["h"]), int(data["w"])
        k_sel = int(data["n_selector"])
        out_dir = args.output_dir if args.output_dir != "outputs/query_diversity" \
            else os.path.dirname(os.path.abspath(args.from_npz))
        label = args.label or os.path.basename(out_dir)
        analyze_and_write(attn, k_sel, h, w, out_dir, names, label)
        return

    if not args.checkpoint_dir:
        raise SystemExit("--checkpoint_dir is required (or use --from_npz / --selfcheck).")

    attn, names, h, w, k_sel = capture(args)
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    label = args.label or os.path.basename(os.path.normpath(out_dir))
    np.savez(os.path.join(out_dir, "attn_raw.npz"),
             attn=attn, image_names=np.asarray(names, dtype="<U128"),
             h=h, w=w, n_selector=k_sel,
             backend=args.backend,
             checkpoint_dir=str(args.checkpoint_dir),
             lora_checkpoint_dir=str(args.lora_checkpoint_dir))
    print(f"saved raw stacks: {os.path.join(out_dir, 'attn_raw.npz')}  attn={attn.shape}", flush=True)
    analyze_and_write(attn, k_sel, h, w, out_dir, names, label)


if __name__ == "__main__":
    main()
