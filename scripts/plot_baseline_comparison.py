"""Render the default-matcher baseline as a grouped bar chart.

Reads the per-sample dumps under outputs/internvl3/grounding/ and produces a
per-benchmark accuracy comparison across the three eval conditions, plus a
"Mean" group on the right.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BENCHES = ["vsr", "gqa_spatial", "whatsup", "3dsrbench", "blink", "srbench"]
CONDS = ["frozen", "lora_only", "reinspection"]
COLORS = {"frozen": "#6c757d", "lora_only": "#4c9ed9", "reinspection": "#d9534f"}
LABELS = {"frozen": "Frozen", "lora_only": "LoRA-only", "reinspection": "Re-inspection"}


def acc(samples_dir, cond, bench):
    p = Path(samples_dir) / f"{cond}_{bench}_samples.json"
    if not p.exists():
        return None
    samples = json.load(open(p))
    if not samples:
        return None
    return sum(1 for s in samples if s["correct"]) / len(samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-dir", default="outputs/internvl3/grounding")
    ap.add_argument("--out", default="outputs/internvl3/grounding/baseline_comparison")
    args = ap.parse_args()

    data = {c: [acc(args.samples_dir, c, b) for b in BENCHES] for c in CONDS}
    means = {c: np.mean([v for v in data[c] if v is not None]) for c in CONDS}

    groups = BENCHES + ["Mean"]
    x = np.arange(len(groups))
    width = 0.26

    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    for i, cond in enumerate(CONDS):
        vals = [v if v is not None else 0 for v in data[cond]] + [means[cond]]
        offset = (i - 1) * width
        bars = ax.bar(
            x + offset,
            [v * 100 for v in vals],
            width,
            color=COLORS[cond],
            label=LABELS[cond],
            edgecolor="white",
            linewidth=0.5,
        )
        for j, (bar, v) in enumerate(zip(bars, vals)):
            if v > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8,
                    f"{v*100:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#222",
                )

    # vertical separator before Mean
    ax.axvline(len(BENCHES) - 0.5, color="#ccc", linewidth=0.8, linestyle=":")

    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=0)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title(
        "Stage-2 epoch_1 baseline — fair matcher\n"
        "InternVL3-8B + Re-inspection (option-text MCQ + bool/direction synonyms)",
        fontsize=11,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    for ext in (".png", ".pdf"):
        out_path = args.out + ext
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
