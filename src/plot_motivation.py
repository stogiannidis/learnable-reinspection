"""Generate publication-quality figures for the motivation experiment.

Usage:
    python -m src.plot_motivation [--output_dir outputs/motivation/figures]

Reads the three per-model result JSONs from outputs/motivation/ and produces:
  - Hero figure (Fig 1): qualitative examples (left) + category bar chart (right)
  - Supporting figures for the appendix
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

from src.utils.attention_schema import (
    ATTN_SIGNAL_CMAP,
    ATTN_SIGNAL_LABELS,
    ATTN_SIGNAL_STORAGE,
    default_display_signal as _default_display_signal,
    single_condition_display_signals as _display_signals_for_condition,
)

# ------------------------------------------------------------------ #
#  Style                                                               #
# ------------------------------------------------------------------ #

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

MODEL_LABELS = {
    "internvl3": "InternVL3 (8B)",
    "qwen25vl": "Qwen2.5-VL (7B)",
    "gemma4": "Gemma 4 (12B)",
}

MODEL_COLORS = {
    "internvl3": "#55A868",
    "qwen25vl": "#4C72B0",
    "gemma4": "#C44E52",
}

MODEL_ORDER = ["internvl3", "qwen25vl", "gemma4"]

IMAGE_ROOT = "/data/datasets/vsr/images"

# Hand-picked examples: (image, subject, relation, object, GT answer, category_label)
# These are Gemma4 failures (all say "False" to both the original and the inverted question)
EXAMPLES = [
    {
        "image": "000000407386.jpg",
        "stmt_a": "The laptop is left of the tv.",
        "stmt_b": "The laptop is right of the tv.",
        "gt_a": "True", "gt_b": "False",
        "gen_a": "True", "gen_b": "False",
        "category": "Horizontal",
        "is_success": True,
    },
    {
        "image": "000000452072.jpg",
        "stmt_a": "The cat is on top of the laptop.",
        "stmt_b": "The cat is under the laptop.",
        "gt_a": "False", "gt_b": "True",
        "gen_a": "False", "gen_b": "False",
        "category": "Vertical",
        "is_success": False,
    },
    {
        "image": "000000169660.jpg",
        "stmt_a": "The oven is above the cake.",
        "stmt_b": "The oven is below the cake.",
        "gt_a": "False", "gt_b": "True",
        "gen_a": "False", "gen_b": "False",
        "category": "Containment",
        "is_success": False,
    },
]

CATEGORIES = {
    "Horizontal\n(left/right)": ["left of", "right of"],
    "Vertical\n(above/below)": ["above", "below", "over", "beneath"],
    "Depth\n(front/behind)": ["in front of", "behind"],
    "Surface\n(on/under)": ["on", "on top of", "under"],
    "Containment\n(in/at/inside)": ["at", "in", "inside", "into", "enclosed by"],
}


def _load_results():
    results = {}
    for model in MODEL_ORDER:
        path = f"outputs/motivation/motivation_{model}_results.json"
        if Path(path).exists():
            results[model] = json.load(open(path))
    return results


def _pct(num, denom):
    return num / denom * 100 if denom > 0 else 0


# ------------------------------------------------------------------ #
#  Hero Figure (Figure 1): Qualitative + Category Chart                #
# ------------------------------------------------------------------ #

def plot_hero_figure(results, output_dir):
    """Composite figure: qualitative examples (left) + category bars (right).

    Layout (roughly 2-column paper width, ~7in x 4in):

        [Example 1]  [Example 2]  [Example 3]  |  [Category bar chart]
        [  check  ]  [  cross  ]  [  cross  ]  |
    """
    fig = plt.figure(figsize=(14, 5.5))

    # GridSpec: 1 row, 2 columns. Left=qualitative (55%), Right=chart (45%)
    gs_top = gridspec.GridSpec(1, 2, width_ratios=[55, 45], wspace=0.08, figure=fig)

    # Left: 3 qualitative examples in a row
    gs_left = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs_top[0],
                                               hspace=0.05, wspace=0.15,
                                               height_ratios=[3, 2])

    for col, ex in enumerate(EXAMPLES):
        # Image
        ax_img = fig.add_subplot(gs_left[0, col])
        img_path = Path(IMAGE_ROOT) / ex["image"]
        if img_path.exists():
            img = Image.open(img_path).convert("RGB")
            ax_img.imshow(img)
        ax_img.set_xticks([])
        ax_img.set_yticks([])

        # Border color: green for success, red for failure
        border_color = "#2ca02c" if ex["is_success"] else "#d62728"
        for spine in ax_img.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2.5)

        ax_img.set_title(ex["category"], fontweight="bold", fontsize=11,
                         color=border_color, pad=4)

        # Text panel below the image
        ax_txt = fig.add_subplot(gs_left[1, col])
        ax_txt.axis("off")

        def _fmt_answer(gen, gt):
            correct = gen.strip().lower() == gt.strip().lower()
            color = "#2ca02c" if correct else "#d62728"
            symbol = "\u2713" if correct else "\u2717"
            return f"{gen}", color, symbol

        gen_a, col_a, sym_a = _fmt_answer(ex["gen_a"], ex["gt_a"])
        gen_b, col_b, sym_b = _fmt_answer(ex["gen_b"], ex["gt_b"])

        text_lines = []
        # Statement A
        text_lines.append(f'"{ex["stmt_a"]}"')
        text_lines.append(f'   GT: {ex["gt_a"]}    Model: {gen_a}  {sym_a}')
        text_lines.append("")
        # Statement B
        text_lines.append(f'"{ex["stmt_b"]}"')
        text_lines.append(f'   GT: {ex["gt_b"]}    Model: {gen_b}  {sym_b}')

        # Use colored text
        y_pos = 0.95
        for i, line in enumerate(text_lines):
            if i == 1:
                color = col_a
                fontweight = "bold"
            elif i == 4:
                color = col_b
                fontweight = "bold"
            else:
                color = "#333"
                fontweight = "normal"
            fontsize = 8.5 if i in (1, 4) else 9
            ax_txt.text(0.5, y_pos, line, transform=ax_txt.transAxes,
                        ha="center", va="top", fontsize=fontsize,
                        color=color, fontweight=fontweight,
                        fontfamily="monospace")
            y_pos -= 0.22

        # Same answer indicator for failures
        if not ex["is_success"]:
            ax_txt.text(0.5, -0.05, "Same answer to both \u2192 spatial failure",
                        transform=ax_txt.transAxes, ha="center", va="top",
                        fontsize=8, color="#d62728", fontstyle="italic")

    # Right: Category bar chart
    ax_bar = fig.add_subplot(gs_top[1])
    _draw_category_bars(ax_bar, results)

    # Global title
    fig.suptitle(
        "Spatial Minimal-Pair Evaluation of Frozen VLMs",
        fontsize=14, fontweight="bold", y=1.02,
    )

    # Panel labels
    fig.text(0.01, 0.98, "(a) Qualitative examples", fontsize=11,
             fontweight="bold", va="top", transform=fig.transFigure)
    fig.text(0.57, 0.98, "(b) Same-answer failure rate by spatial category", fontsize=11,
             fontweight="bold", va="top", transform=fig.transFigure)

    out = Path(output_dir) / "fig1_hero.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"Saved: {out}")


def _draw_category_bars(ax, results):
    """Draw the category comparison bar chart on the given axes."""
    models_present = [m for m in MODEL_ORDER if m in results]
    categories = list(CATEGORIES.keys())

    x = np.arange(len(categories))
    n_models = len(models_present)
    width = 0.7 / n_models

    for j, model in enumerate(models_present):
        fc = results[model]["flip_consistency"]["frozen"]
        cat_vals = []
        for cat, rels in CATEGORIES.items():
            total = 0
            same = 0
            for rel in rels:
                stats = fc["per_relation"].get(rel)
                if stats:
                    total += stats["total"]
                    same += stats["same_answer"]
            cat_vals.append(_pct(same, total) if total > 0 else 0)

        offset = (j - (n_models - 1) / 2) * width
        bars = ax.bar(
            x + offset, cat_vals, width,
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model], edgecolor="white", linewidth=0.5,
        )
        for bar, val in zip(bars, cat_vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=7.5,
                    fontweight="bold", color=MODEL_COLORS[model],
                )

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel("Same-Answer Rate (%)")
    ax.set_ylim(0, 72)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(loc="upper left", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # "Higher = worse" annotation
    ax.text(0.98, 0.97, "higher = worse spatial reasoning",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, fontstyle="italic", color="#888")

    # Gradient arrow
    ax.annotate("", xy=(4.4, -6), xytext=(-0.4, -6),
                xycoords="data",
                arrowprops=dict(arrowstyle="->", color="#999", lw=1.2))
    ax.text(2, -9.5, "increasing spatial complexity",
            ha="center", fontsize=8, fontstyle="italic", color="#888")


# ------------------------------------------------------------------ #
#  Appendix figures                                                    #
# ------------------------------------------------------------------ #

def plot_relation_heatmap(results, output_dir):
    """Heatmap: same-answer rate per relation per model."""
    RELATIONS_TO_SHOW = [
        "left of", "right of", "above", "behind", "in front of",
        "over", "on top of", "below", "beneath", "under", "on", "at", "in", "inside",
    ]
    models_present = [m for m in MODEL_ORDER if m in results]
    all_rels = set()
    for model in models_present:
        fc = results[model]["flip_consistency"]["frozen"]
        all_rels |= set(fc["per_relation"].keys())

    rels = [r for r in RELATIONS_TO_SHOW if r in all_rels]

    matrix = np.full((len(rels), len(models_present)), np.nan)
    annotations = np.empty_like(matrix, dtype=object)

    for j, model in enumerate(models_present):
        fc = results[model]["flip_consistency"]["frozen"]
        for i, rel in enumerate(rels):
            stats = fc["per_relation"].get(rel)
            if stats and stats["total"] >= 5:
                val = _pct(stats["same_answer"], stats["total"])
                matrix[i, j] = val
                annotations[i, j] = f"{val:.0f}%"
            else:
                annotations[i, j] = "\u2014"

    fig, ax = plt.subplots(figsize=(4.5, 5.5))
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=70)

    ax.set_xticks(range(len(models_present)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in models_present], fontsize=9)
    ax.set_yticks(range(len(rels)))
    ax.set_yticklabels(rels, fontsize=9)

    for i in range(len(rels)):
        for j in range(len(models_present)):
            text = annotations[i, j]
            color = "white" if (not np.isnan(matrix[i, j]) and matrix[i, j] > 45) else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color, fontweight="bold")

    ax.set_title("Same-Answer Rate by Relation (%)", fontweight="bold", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Same-Answer Rate (%)")

    plt.tight_layout()
    out = Path(output_dir) / "fig_appendix_relation_heatmap.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"Saved: {out}")


def plot_accuracy_asymmetry(results, output_dir):
    """Grouped bar: original vs. flipped accuracy per model."""
    models_present = [m for m in MODEL_ORDER if m in results]

    x = np.arange(len(models_present))
    width = 0.3

    acc_orig = []
    acc_flip = []
    for model in models_present:
        fc = results[model]["flip_consistency"]["frozen"]
        n = fc["total_pairs"]
        acc_orig.append(_pct(fc["a_correct"], n))
        acc_flip.append(_pct(fc["b_correct"], n))

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    bars1 = ax.bar(x - width / 2, acc_orig, width, label="Original question", color="#4C72B0", edgecolor="white")
    bars2 = ax.bar(x + width / 2, acc_flip, width, label="Flipped question", color="#C44E52", edgecolor="white")

    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

    for i in range(len(models_present)):
        delta = acc_orig[i] - acc_flip[i]
        ax.annotate(
            f"\u0394={delta:.0f}pp",
            xy=(x[i] + width / 2 + 0.05, acc_flip[i] + 1),
            fontsize=8, color="#888", fontstyle="italic", ha="left",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models_present])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Accuracy Asymmetry: Original vs. Flipped Questions", fontweight="bold")

    plt.tight_layout()
    out = Path(output_dir) / "fig_appendix_accuracy_asymmetry.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"Saved: {out}")


# ------------------------------------------------------------------ #
#  Attention comparison (from saved .npz files)                        #
# ------------------------------------------------------------------ #

def _npz_available_signals(data):
    """Read expanded or legacy attention signal metadata from an NPZ."""
    if "available_signals" in data.files:
        return [str(x) for x in np.asarray(data["available_signals"]).tolist()]
    if "attn_source" in data.files:
        return [str(data["attn_source"])]
    condition = str(data["condition"]) if "condition" in data.files else "reinspection"
    return ["a_vis" if condition == "reinspection" else "hidden_cosine"]


def _npz_signal_pair(data, signal):
    """Load one signal pair, falling back to legacy attn_a/attn_b when needed."""
    prefix = ATTN_SIGNAL_STORAGE[signal]
    key_a = f"{prefix}_a"
    key_b = f"{prefix}_b"
    if key_a in data.files and key_b in data.files:
        return np.asarray(data[key_a]), np.asarray(data[key_b])

    legacy_source = str(data["attn_source"]) if "attn_source" in data.files else None
    if legacy_source == signal or (legacy_source is None and signal in _npz_available_signals(data)):
        return np.asarray(data["attn_a"]), np.asarray(data["attn_b"])
    return None, None


def _render_attention_condition(npz_files, output_dir, condition, suffix=""):
    """Render one condition-specific attention figure from saved NPZ files.

    Layout per row: [reference image] [Q_A overlay] [Q_B overlay] (per signal).
    Question text is rendered as a wrapped caption *below* each overlay so
    titles cannot collide across columns. The reference image is shown once
    per row at reduced width; overlays render the attention as a smoothly
    blended heatmap over the photograph.
    """
    import textwrap

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if not npz_files:
        print(f"No NPZ files to render for condition={condition}{suffix or ''}.")
        return
    available_signals = []
    for npz_file in npz_files:
        data = np.load(npz_file, allow_pickle=True)
        for signal in _npz_available_signals(data):
            if signal not in available_signals:
                available_signals.append(signal)
    display_signals = _display_signals_for_condition(condition, available_signals)
    n = len(npz_files)
    n_signals = len(display_signals)
    n_cols = 1 + 2 * n_signals
    fig_w = 3.2 + 3.6 * 2 * n_signals
    fig_h = 4.4 * n
    width_ratios = [0.85] + [1.0] * (2 * n_signals)
    fig, axes = plt.subplots(
        n, n_cols, figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": width_ratios, "wspace": 0.18, "hspace": 0.6},
    )
    if n == 1:
        axes = axes[np.newaxis, :]

    # Use a vivid perceptually-uniform colormap regardless of signal — the
    # schema defaults (cividis/hot) blend into dark image regions.
    _HEATMAP_CMAP = "turbo"

    def _overlay(ax, attn_map, h_p, w_p, img, cmap):
        """Render `attn_map` as a heatmap blended over `img`.

        Steps:
          1. Reshape to the patch grid (h_p, w_p) and normalize by its max.
             Max-normalization (not percentile clipping) is essential because
             A_vis is extremely sparse — a single peak at 1.0 with the rest
             near 1e-30 — so percentile clipping would erase the signal.
          2. Lanczos-upsample to image resolution for smooth contours.
          3. Dim the photograph slightly so warm-color peaks pop, then
             stack the colorized heatmap with a smoothstep alpha curve so
             the noise floor stays transparent and peaks saturate.
        """
        del cmap  # signal-specific cmap intentionally overridden for legibility

        attn_2d = attn_map.reshape(h_p, w_p).astype(np.float32)
        amax = float(attn_2d.max())
        if amax < 1e-20:
            norm = np.zeros_like(attn_2d)
        else:
            norm = np.clip(attn_2d / amax, 0.0, 1.0)

        img_w, img_h = img.size
        norm_img = np.asarray(
            Image.fromarray((norm * 255.0).astype(np.uint8)).resize(
                (img_w, img_h), Image.LANCZOS,
            ),
            dtype=np.float32,
        ) / 255.0
        norm_img = np.clip(norm_img, 0.0, 1.0)

        cmap_obj = plt.get_cmap(_HEATMAP_CMAP)
        rgba = cmap_obj(norm_img)

        # Smoothstep alpha above a small knee: noise floor → transparent,
        # peaks → ~0.9 alpha. 3y^2 - 2y^3 gives a smooth S-curve.
        knee = 0.05
        ramp = np.clip((norm_img - knee) / (1.0 - knee), 0.0, 1.0)
        ramp = ramp * ramp * (3.0 - 2.0 * ramp)
        rgba[..., 3] = ramp * 0.90

        # Render the photograph slightly dimmed so the heatmap reads cleanly.
        ax.imshow(img, alpha=0.85)
        ax.imshow(rgba, interpolation="bilinear")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    def _wrap(text, width=38):
        return "\n".join(textwrap.wrap(text, width=width)) or text

    axes[0, 0].set_title("Image", fontsize=10, fontweight="bold", pad=8)
    for s_idx, signal in enumerate(display_signals):
        label = ATTN_SIGNAL_LABELS.get(signal, signal)
        axes[0, 1 + 2 * s_idx].set_title(
            f"{label}\nQuestion A", fontsize=10, fontweight="bold", pad=8,
        )
        axes[0, 2 + 2 * s_idx].set_title(
            f"{label}\nQuestion B", fontsize=10, fontweight="bold", pad=8,
        )

    for row, npz_file in enumerate(npz_files):
        data = np.load(npz_file, allow_pickle=True)
        img_path = str(data["image_path"])
        if not Path(img_path).exists():
            for c in range(n_cols):
                axes[row, c].axis("off")
            continue

        image = Image.open(img_path).convert("RGB")
        h_m = int(data["h_merged"])
        w_m = int(data["w_merged"])
        q_a = str(data["question_a"]).strip()
        q_b = str(data["question_b"]).strip()
        ans_a = str(data["answer_a"]).strip()
        ans_b = str(data["answer_b"]).strip()
        gt_a = str(data["gt_a"]).strip()
        gt_b = str(data["gt_b"]).strip()
        relation = str(data["relation"])

        ax_ref = axes[row, 0]
        ax_ref.imshow(image)
        ax_ref.set_xticks([]); ax_ref.set_yticks([])
        for spine in ax_ref.spines.values():
            spine.set_visible(False)
        ax_ref.set_ylabel(
            f'"{relation}"', fontsize=10, fontweight="bold", rotation=0,
            ha="right", va="center", labelpad=14,
        )

        a_correct = ans_a.lower() == gt_a.lower()
        b_correct = ans_b.lower() == gt_b.lower()
        a_sym = "\u2713" if a_correct else "\u2717"
        b_sym = "\u2713" if b_correct else "\u2717"
        a_color = "#2ca02c" if a_correct else "#d62728"
        b_color = "#2ca02c" if b_correct else "#d62728"

        for s_idx, signal in enumerate(display_signals):
            ax_a = axes[row, 1 + 2 * s_idx]
            ax_b = axes[row, 2 + 2 * s_idx]
            attn_a, attn_b = _npz_signal_pair(data, signal)
            if attn_a is None or attn_b is None:
                ax_a.axis("off"); ax_b.axis("off")
                continue
            cmap = ATTN_SIGNAL_CMAP.get(signal, "cividis")
            _overlay(ax_a, attn_a, h_m, w_m, image, cmap)
            _overlay(ax_b, attn_b, h_m, w_m, image, cmap)
            ax_a.set_xlabel(
                f"{_wrap(q_a)}\nModel: {ans_a} {a_sym}   GT: {gt_a}",
                fontsize=8, color="#222", labelpad=6,
            )
            ax_b.set_xlabel(
                f"{_wrap(q_b)}\nModel: {ans_b} {b_sym}   GT: {gt_b}",
                fontsize=8, color="#222", labelpad=6,
            )
            ax_a.plot([0.0, 1.0], [-0.02, -0.02], transform=ax_a.transAxes,
                      color=a_color, lw=2.0, clip_on=False)
            ax_b.plot([0.0, 1.0], [-0.02, -0.02], transform=ax_b.transAxes,
                      color=b_color, lw=2.0, clip_on=False)

    label = " + ".join(ATTN_SIGNAL_LABELS.get(sig, sig) for sig in display_signals)
    fig.suptitle(
        f"Attention comparison ({condition}){suffix.replace('_', ' ')} \u2014 {label}",
        fontsize=13, fontweight="bold", y=0.995,
    )
    out = Path(output_dir) / f"{condition}{suffix}_attention_comparison.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def _correct(ans, gt) -> bool:
    return str(ans).strip().lower() == str(gt).strip().lower()


def _pair_key(data) -> tuple:
    return (
        str(data["image_path"]),
        str(data["question_a"]).strip(),
        str(data["question_b"]).strip(),
    )


def _select_reinspection_wins(reins_paths, frozen_paths):
    """Return reinspection NPZ paths where the wrapper outperforms the frozen baseline.

    A "win" = reinspection is correct on **both** Q_A and Q_B for the pair,
    and the frozen baseline is wrong on at least one of them. Matching is by
    (image_path, question_a, question_b) so it is robust to reordering across
    runs. Falls back to "either question is a strict gain" when no clean wins
    exist, then to all reinspection paths.
    """
    frozen_by_key = {}
    for p in frozen_paths:
        d = np.load(p, allow_pickle=True)
        frozen_by_key[_pair_key(d)] = {
            "a_correct": _correct(d["answer_a"], d["gt_a"]),
            "b_correct": _correct(d["answer_b"], d["gt_b"]),
        }

    strong_wins, weak_wins = [], []
    for p in reins_paths:
        d = np.load(p, allow_pickle=True)
        key = _pair_key(d)
        if key not in frozen_by_key:
            continue
        fr = frozen_by_key[key]
        ri_a = _correct(d["answer_a"], d["gt_a"])
        ri_b = _correct(d["answer_b"], d["gt_b"])
        ri_both = ri_a and ri_b
        gained = (ri_a and not fr["a_correct"]) or (ri_b and not fr["b_correct"])
        if ri_both and (not fr["a_correct"] or not fr["b_correct"]):
            strong_wins.append(p)
        elif gained:
            weak_wins.append(p)

    if strong_wins:
        print(f"  reinspection wins (both correct, frozen errs): {len(strong_wins)}")
        return strong_wins
    if weak_wins:
        print(f"  reinspection partial gains (no clean wins): {len(weak_wins)}")
        return weak_wins
    print("  no reinspection wins found — rendering all reinspection pairs")
    return list(reins_paths)


def plot_attention_comparison(
    output_dir,
    attn_dir="outputs/motivation/attention_figures",
    wins_only=True,
):
    """Re-render condition attention figures from saved NPZ files.

    The .npz files are generated by ``src.motivation.run_attention_visualization``
    during the motivation experiment.  This function allows regenerating the
    figure without re-running the model.

    When ``wins_only`` is True, the reinspection figure is restricted to pairs
    where the re-inspection module beats the frozen baseline (see
    ``_select_reinspection_wins``), making the comparison story explicit. The
    frozen figure always renders all pairs so the reader sees the failure
    modes in context.
    """
    attn_path = Path(attn_dir)
    npz_files = sorted(set(attn_path.glob("*_attn_pair_*.npz")) | set(attn_path.glob("attn_pair_*.npz")))
    if not npz_files:
        print(f"No attention .npz files found in {attn_dir} — skipping attention comparison plot.")
        return

    grouped = defaultdict(list)
    for npz_file in npz_files:
        data = np.load(npz_file, allow_pickle=True)
        if "condition" in data.files:
            condition = str(data["condition"])
        elif npz_file.name.startswith("frozen_"):
            condition = "frozen"
        else:
            condition = "reinspection"
        grouped[condition].append(npz_file)

    for condition, paths in sorted(grouped.items()):
        sorted_paths = sorted(paths)
        if condition == "reinspection" and wins_only and grouped.get("frozen"):
            win_paths = _select_reinspection_wins(sorted_paths, sorted(grouped["frozen"]))
            _render_attention_condition(win_paths, output_dir, condition)
        else:
            _render_attention_condition(sorted_paths, output_dir, condition)


# ------------------------------------------------------------------ #
#  Mirror test bar chart                                               #
# ------------------------------------------------------------------ #

def plot_mirror_test(results, output_dir):
    """Bar chart: mirror consistency rate per model."""
    models_present = [m for m in MODEL_ORDER if m in results]
    if not models_present:
        return

    has_mirror = any(
        "mirror_test" in results[m] and "frozen" in results[m]["mirror_test"]
        for m in models_present
    )
    if not has_mirror:
        print("No mirror test results found — skipping mirror plot.")
        return

    x = np.arange(len(models_present))
    width = 0.35

    consistency_vals = []
    correct_vals = []
    for model in models_present:
        mt = results[model].get("mirror_test", {}).get("frozen", {})
        n = mt.get("total", 0)
        if n > 0:
            consistency_vals.append(mt["mirror_consistent"] / n * 100)
            correct_vals.append(mt["mirror_correct"] / n * 100)
        else:
            consistency_vals.append(0)
            correct_vals.append(0)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    bars1 = ax.bar(x - width / 2, consistency_vals, width,
                    label="Mirror consistency", color=[MODEL_COLORS[m] for m in models_present],
                    edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, correct_vals, width,
                    label="Mirror correct", color=[MODEL_COLORS[m] for m in models_present],
                    edgecolor="white", linewidth=0.5, alpha=0.6)

    for bars in [bars1, bars2]:
        for bar in bars:
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                        f"{bar.get_height():.0f}", ha="center", va="bottom",
                        fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models_present])
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Mirror Test: Left/Right Image Flip (Frozen VLMs)", fontweight="bold")

    plt.tight_layout()
    out = Path(output_dir) / "fig_mirror_test.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"Saved: {out}")


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Plot motivation experiment figures")
    parser.add_argument("--output_dir", type=str, default="outputs/motivation/figures")
    parser.add_argument("--attn_dir", type=str, default="outputs/motivation/attention_figures",
                        help="Directory containing attention .npz files")
    parser.add_argument(
        "--no-wins-only", dest="wins_only", action="store_false",
        help="Render all reinspection pairs (default: only show pairs where reinspection beats the frozen baseline).",
    )
    parser.set_defaults(wins_only=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = _load_results()
    if not results:
        print("No result files found in outputs/motivation/")
        return

    print(f"Loaded results for: {', '.join(results.keys())}")
    print(f"Output dir: {output_dir}\n")

    plot_hero_figure(results, output_dir)
    plot_relation_heatmap(results, output_dir)
    plot_accuracy_asymmetry(results, output_dir)
    plot_mirror_test(results, output_dir)
    plot_attention_comparison(output_dir, attn_dir=args.attn_dir, wins_only=args.wins_only)

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
