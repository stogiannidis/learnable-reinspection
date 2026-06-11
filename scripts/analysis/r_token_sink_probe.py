"""Probe the per-token text->image attention grids saved in maps.npz for a
first-image-patch attention *sink* (the "always bright top-left corner").

For every (condition, token-group) in each run's maps.npz it reports, on the
raw HxW patch grids (NOT the bicubic-resized display, so any display/normalize
artifact is bypassed):

* mean-grid argmax cell (y,x) and the top-3 cells of the per-token-normalised
  mean grid — where the aggregate hotspot actually sits;
* how many of the N tokens have their individual argmax at the top-left corner
  (0,0) and within the top-left 2x2 — tests "always";
* the mean normalised weight at (0,0) vs uniform 1/(H*W) — how dominant the
  corner is;
* row-0 vs rest and col-0 vs rest marginals — distinguishes a single-corner
  sink from a whole top edge / left column.

Run over any number of run dirs (each holding maps.npz) or parent dirs.
"""
from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np

GRID_KEY = re.compile(r"^(?P<cond>[^/]+)/(?P<group>input_maps|generated_maps|r_maps)/(?P<idx>\d+)/grid$")


def _collect(npz_path):
    raw = np.load(npz_path, allow_pickle=False)
    h, w = int(raw["meta/h"]), int(raw["meta/w"])
    groups: dict = {}
    for k in raw.files:
        m = GRID_KEY.match(k)
        if m:
            groups.setdefault((m["cond"], m["group"]), {})[int(m["idx"])] = raw[k]
    return h, w, groups


def _stats(grids, h, w):
    n = grids.shape[0]
    flat = grids.reshape(n, -1).astype(np.float64)
    sums = flat.sum(1, keepdims=True)
    norm = flat / np.clip(sums, 1e-12, None)            # per-token distribution over patches
    mean = norm.mean(0).reshape(h, w)                   # aggregate distribution
    my, mx = np.unravel_index(mean.argmax(), mean.shape)
    locs = [np.unravel_index(a, (h, w)) for a in norm.argmax(1)]
    at_corner = sum(1 for (y, x) in locs if y == 0 and x == 0)
    at_tl2 = sum(1 for (y, x) in locs if y <= 1 and x <= 1)
    uniform = 1.0 / (h * w)
    v00 = float(norm.reshape(n, h, w)[:, 0, 0].mean())
    order = np.argsort(mean.ravel())[::-1][:3]
    top3 = [(int(c // w), int(c % w), round(float(mean.ravel()[c]), 4)) for c in order]
    row0 = float(mean[0].sum())
    col0 = float(mean[:, 0].sum())
    return dict(n=n, mean_argmax=(int(my), int(mx)), at_corner=at_corner, at_tl2=at_tl2,
                v00=v00, uniform=uniform, top3=top3, row0=row0, col0=col0,
                row_share=row0, col_share=col0)


def main(argv):
    targets = argv or ["outputs/token_attn"]
    npzs = []
    for t in targets:
        if t.endswith(".npz"):
            npzs.append(t)
        elif os.path.isfile(os.path.join(t, "maps.npz")):
            npzs.append(os.path.join(t, "maps.npz"))
        else:
            npzs += sorted(glob.glob(os.path.join(t, "**", "maps.npz"), recursive=True))
    npzs = sorted(dict.fromkeys(npzs))
    if not npzs:
        raise SystemExit(f"no maps.npz under {targets}")

    for npz in npzs:
        run = os.path.relpath(os.path.dirname(npz))
        h, w, groups = _collect(npz)
        print(f"\n===== {run}  (grid {h}x{w}, uniform={1.0/(h*w):.4f}) =====")
        for (cond, group), d in sorted(groups.items()):
            grids = np.stack([d[i] for i in sorted(d)], 0)
            s = _stats(grids, h, w)
            print(
                f"  {cond:12s} {group:15s} n={s['n']:<4d} "
                f"mean-argmax={str(s['mean_argmax']):7s} "
                f"argmax@(0,0)={s['at_corner']:>3d}/{s['n']:<3d} "
                f"@TL2x2={s['at_tl2']:>3d}/{s['n']:<3d} "
                f"w(0,0)={s['v00']:.4f} ({s['v00']/s['uniform']:.1f}x unif) "
                f"row0={s['row0']:.3f} col0={s['col0']:.3f}"
            )
            print(f"               top-3 mean cells (y,x,val): {s['top3']}")


if __name__ == "__main__":
    main(sys.argv[1:])
