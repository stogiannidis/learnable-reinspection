"""Combine attention-sink analysis PNGs into a single montage figure.

Expects the five plots written by ``attention_sink_analysis.py``::

  a0_layerhead_heatmap.png
  bos_received_per_layer.png
  sink_rate_bars.png
  answer_budget_stacked.png
  hidden_norm_trajectory.png

Example::

    PYTHONPATH=. python scripts/analysis/combine_sink_figures.py \\
        --input_dir outputs/sink_analysis/s2e2
"""

from __future__ import annotations

import argparse
import os

PANELS = (
    ("a0_layerhead_heatmap.png", "A  BOS-received per (layer, head)"),
    ("bos_received_per_layer.png", "B  BOS sink vs layer"),
    ("sink_rate_bars.png", "C  sink_rate (frozen vs reinspection)"),
    ("answer_budget_stacked.png", "D  Answer-query attention budget"),
    ("hidden_norm_trajectory.png", "E  Hidden-state L2 norm"),
)


def combine(input_dir: str, output_path: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import image as mpimg

    paths = []
    titles = []
    for name, title in PANELS:
        p = os.path.join(input_dir, name)
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
        paths.append(p)
        titles.append(title)

    # Top: heatmap (wide) + sink bars; bottom: three line/stack plots.
    fig = plt.figure(figsize=(18, 11), dpi=100, facecolor="#0f1117")
    gs = fig.add_gridspec(
        2, 3,
        height_ratios=[1.05, 1.0],
        width_ratios=[1.0, 1.0, 0.55],
        hspace=0.28,
        wspace=0.12,
        left=0.02,
        right=0.98,
        top=0.96,
        bottom=0.02,
    )

    axes_spec = [
        (gs[0, 0:2], paths[0], titles[0]),
        (gs[0, 2], paths[2], titles[2]),
        (gs[1, 0], paths[1], titles[1]),
        (gs[1, 1], paths[3], titles[3]),
        (gs[1, 2], paths[4], titles[4]),
    ]
    for spec, path, title in axes_spec:
        ax = fig.add_subplot(spec)
        ax.imshow(mpimg.imread(path))
        ax.set_title(title, fontsize=11, color="#e8ecf4", loc="left", pad=8)
        ax.axis("off")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    pdf_path = output_path.replace(".png", ".pdf")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    print(f"Wrote {output_path}")
    print(f"Wrote {pdf_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_dir", default="outputs/sink_analysis/s2e2")
    p.add_argument("--output", default=None, help="Output PNG (default: <input_dir>/combined.png)")
    args = p.parse_args()
    out = args.output or os.path.join(args.input_dir, "combined.png")
    combine(args.input_dir, out)


if __name__ == "__main__":
    main()
