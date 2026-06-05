"""Interactive (Plotly) per-token text→image attention explorer.

Builds a standalone HTML from the ``maps.npz`` saved by
``scripts/analysis/visualize_token_attention.py`` — pure post-processing, no
GPU/model. One animated figure per token group:

- input tokens: Frozen | Re-Inspection side-by-side (same token sequence);
- generated tokens: one figure per condition (answers differ);
- R tokens: reinspection only.

Each figure gets a slider (one step per token) plus Play/Pause, with the
attention grid rendered as a smoothed semi-transparent heatmap over the image.

Usage:
    python scripts/analysis/interactive_token_attention.py \
        --run_dir outputs/token_attn/internvl3_cat_laptop \
        [--image /data/datasets/vsr/images/x.jpg] [--output .../interactive.html]
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

GROUPS = ("input_maps", "generated_maps", "r_maps")
COND_ORDER = ("frozen", "reinspection")


def load_maps(npz_path: str) -> Tuple[Dict[str, Dict[str, List[Tuple[str, np.ndarray]]]], Dict[str, str]]:
    """Parse the flat npz into {cond: {group: [(label, grid), ...]}} + meta."""
    raw = np.load(npz_path, allow_pickle=False)
    by_cond: Dict[str, Dict[str, Dict[int, dict]]] = defaultdict(lambda: defaultdict(dict))
    meta: Dict[str, str] = {}
    answers: Dict[str, str] = {}
    pat = re.compile(r"^(?P<cond>[^/]+)/(?P<group>input_maps|generated_maps|r_maps)/(?P<idx>\d+)/(?P<field>grid|label)$")
    for key in raw.files:
        if key.startswith("meta/"):
            meta[key[5:]] = str(raw[key])
            continue
        if key.endswith("/answer"):
            answers[key.split("/", 1)[0]] = str(raw[key])
            continue
        m = pat.match(key)
        if m:
            entry = by_cond[m["cond"]][m["group"]].setdefault(int(m["idx"]), {})
            entry[m["field"]] = raw[key]
    out: Dict[str, Dict[str, List[Tuple[str, np.ndarray]]]] = {}
    for cond, groups in by_cond.items():
        out[cond] = {}
        for group, entries in groups.items():
            ordered = [entries[i] for i in sorted(entries)]
            out[cond][group] = [(str(e.get("label", "?")), np.asarray(e["grid"], dtype=np.float32))
                                for e in ordered if "grid" in e]
    meta["answers"] = answers  # type: ignore[assignment]
    return out, meta


def _image_data_uri(image_path: str) -> Tuple[str, int, int]:
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), img.width, img.height


def _heatmap(grid: np.ndarray, w: int, h: int, zmax: float, colorscale: str, opacity: float):
    import plotly.graph_objects as go

    gh, gw = grid.shape
    return go.Heatmap(
        z=grid,
        # Cell centers spanning the full image extent.
        x=(np.arange(gw) + 0.5) * (w / gw),
        y=(np.arange(gh) + 0.5) * (h / gh),
        zmin=0.0, zmax=zmax,
        colorscale=colorscale, opacity=opacity, zsmooth="best",
        showscale=False, hovertemplate="attn=%{z:.4f}<extra></extra>",
    )


def build_group_figure(
    panels: List[Tuple[str, List[Tuple[str, np.ndarray]]]],
    image_uri: str,
    w: int,
    h: int,
    title: str,
    colorscale: str = "Turbo",
    opacity: float = 0.55,
    per_token_scale: bool = False,
    frame_ms: int = 350,
):
    """One animated figure: `panels` = [(condition, [(label, grid), ...]), ...].

    All panels must have the same number of steps; step k shows panel p's k-th
    grid. Returns a plotly Figure with slider + play/pause.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n_steps = min(len(maps) for _, maps in panels)
    zmax_global = max(float(g.max()) for _, maps in panels for _, g in maps[:n_steps]) or 1.0

    fig = make_subplots(
        rows=1, cols=len(panels),
        subplot_titles=[c for c, _ in panels],
        horizontal_spacing=0.04,
    )

    def step_traces(k: int):
        traces = []
        for _, maps in panels:
            label, grid = maps[k]
            zmax = (float(grid.max()) or 1.0) if per_token_scale else zmax_global
            traces.append(_heatmap(grid, w, h, zmax, colorscale, opacity))
        return traces

    for col, tr in enumerate(step_traces(0), start=1):
        fig.add_trace(tr, row=1, col=col)

    frames, steps = [], []
    for k in range(n_steps):
        label = panels[0][1][k][0]
        name = f"{k}"
        frames.append(go.Frame(data=step_traces(k), name=name,
                               layout=go.Layout(title_text=f"{title} — token [{k}] {label!r}")))
        steps.append(dict(
            method="animate",
            args=[[name], {"mode": "immediate",
                           "frame": {"duration": 0, "redraw": True},
                           "transition": {"duration": 0}}],
            label=label[:14],
        ))
    fig.frames = frames

    for col in range(1, len(panels) + 1):
        ax = "" if col == 1 else str(col)
        fig.add_layout_image(dict(
            source=image_uri, xref=f"x{ax}", yref=f"y{ax}",
            x=0, y=0, sizex=w, sizey=h, sizing="stretch", layer="below",
        ))
        fig.update_xaxes(range=[0, w], visible=False, row=1, col=col)
        # Image y grows downward.
        fig.update_yaxes(range=[h, 0], visible=False, scaleanchor=f"x{ax}", row=1, col=col)

    fig.update_layout(
        title_text=f"{title} — token [0] {panels[0][1][0][0]!r}",
        height=520, margin=dict(l=10, r=10, t=90, b=10),
        sliders=[dict(
            active=0, steps=steps, len=0.92, x=0.04, y=-0.04,
            currentvalue=dict(prefix="token: ", font=dict(size=13)),
            font=dict(size=9),
        )],
        updatemenus=[dict(
            type="buttons", direction="left", x=0.0, y=1.18, showactive=False,
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, {"frame": {"duration": frame_ms, "redraw": True},
                                  "fromcurrent": True, "transition": {"duration": 0}}]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], {"mode": "immediate",
                                    "frame": {"duration": 0, "redraw": True}}]),
            ],
        )],
    )
    return fig


def build_html(npz_path: str, image_path: Optional[str], output: Optional[str],
               colorscale: str = "Turbo", opacity: float = 0.55,
               per_token_scale: bool = False) -> str:
    maps, meta = load_maps(npz_path)
    run_dir = os.path.dirname(os.path.abspath(npz_path))
    image_path = image_path or meta.get("image", "")
    if not (image_path and os.path.isfile(image_path)):
        raise FileNotFoundError(f"image not found: {image_path!r} (pass --image)")
    image_uri, w, h = _image_data_uri(image_path)
    answers = meta.get("answers", {})

    conds = [c for c in COND_ORDER if c in maps] + sorted(set(maps) - set(COND_ORDER))
    sections: List[str] = []
    first = True

    def add(fig):
        nonlocal first
        sections.append(fig.to_html(full_html=False, include_plotlyjs=("inline" if first else False)))
        first = False

    # Input tokens: side-by-side across conditions (same token sequence).
    input_panels = [(c, maps[c]["input_maps"]) for c in conds if maps[c].get("input_maps")]
    if input_panels:
        add(build_group_figure(input_panels, image_uri, w, h, "Input token → image attention",
                               colorscale, opacity, per_token_scale))
    # Generated tokens: per condition (answers differ).
    for c in conds:
        if maps[c].get("generated_maps"):
            add(build_group_figure([(c, maps[c]["generated_maps"])], image_uri, w, h,
                                   f"Generated token → image attention ({c})",
                                   colorscale, opacity, per_token_scale))
    # R tokens (reinspection only).
    for c in conds:
        if maps[c].get("r_maps"):
            add(build_group_figure([(c, maps[c]["r_maps"])], image_uri, w, h,
                                   f"R token → image attention ({c})",
                                   "Viridis", opacity, per_token_scale))

    head = (
        f"<h2>Per-token text→image attention — {html.escape(meta.get('backend', '?'))}</h2>"
        f"<p><b>Q:</b> {html.escape(meta.get('question', '?'))}<br>"
        + "".join(f"<b>A ({html.escape(c)}):</b> {html.escape(a)}<br>" for c, a in answers.items())
        + f"<small>layer_reduce={html.escape(meta.get('layer_reduce', '?'))} · grid from {html.escape(os.path.basename(npz_path))}</small></p>"
    )
    doc = ("<!DOCTYPE html><html><head><meta charset='utf-8'>"
           "<title>token attention</title></head><body style='font-family:sans-serif'>"
           + head + "<hr>".join(sections) + "</body></html>")

    output = output or os.path.join(run_dir, "interactive.html")
    with open(output, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"  saved {output} ({os.path.getsize(output) / 1e6:.1f} MB)", flush=True)
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_dir", required=True, help="dir containing maps.npz (or path to a maps.npz)")
    ap.add_argument("--image", default=None, help="override image path (default: meta/image in the npz)")
    ap.add_argument("--output", default=None, help="output html (default: <run_dir>/interactive.html)")
    ap.add_argument("--colorscale", default="Turbo")
    ap.add_argument("--opacity", type=float, default=0.55)
    ap.add_argument("--per_token_scale", action="store_true",
                    help="normalize each token's heatmap to its own max (default: global per group)")
    args = ap.parse_args()

    npz = args.run_dir if args.run_dir.endswith(".npz") else os.path.join(args.run_dir, "maps.npz")
    build_html(npz, args.image, args.output, args.colorscale, args.opacity, args.per_token_scale)


if __name__ == "__main__":
    main()
