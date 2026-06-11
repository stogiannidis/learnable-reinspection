"""CLI: per-token text→image attention overlays for InternVL3 / LLaVA-Next.

For one (image, question) pair this renders, for **every input text token**
(system prompt, special/boundary tokens, the query, and the appended
assistant-generation suffix) *and* every generated answer token, a heatmap of
that token's decoder self-attention over the image patch grid — overlaid on the
image. Frozen base VLM and the Re-Inspection model are shown side by side.

Requires a GPU + ``attn_implementation="eager"`` (set automatically here). The
pure extraction math lives in ``src/utils/token_attention.py`` and is unit
tested on CPU; this script is the model-touching orchestrator.

Examples
--------
Frozen + Re-Inspection, InternVL3::

    PYTHONPATH=. python scripts/analysis/visualize_token_attention.py \\
        --backend internvl3 \\
        --image /data/datasets/vsr/images/000000000142.jpg \\
        --question "Is the cat to the left of the laptop?" \\
        --checkpoint_dir models/internvl3/stage1/epoch_5 \\
        --lora_checkpoint_dir models/internvl3/gqa_sg/stage2/epoch_1 \\
        --output_dir outputs/token_attn/internvl3_demo

Frozen only (no checkpoint), LLaVA-Next (base global view)::

    PYTHONPATH=. python scripts/analysis/visualize_token_attention.py \\
        --backend llava_next --conditions frozen \\
        --image <img> --question "<q>" --output_dir outputs/token_attn/llava_demo
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from typing import Dict, List, Optional

import numpy as np
import yaml
from PIL import Image

import torch

from src.config import ReInspectionConfig
from src.utils.token_attention import (
    KIND_R,
    KIND_TEXT,
    TokenMap,
    capture_token_attention,
)
from src.utils.visualize_attention import plot_attention_heatmap

# Distinct colormaps per condition so the side-by-side panels are easy to tell
# apart at a glance.
FROZEN_CMAP = "viridis"
RI_CMAP = "inferno"


# --------------------------------------------------------------------------- #
# Config / model loading                                                        #
# --------------------------------------------------------------------------- #

def _config_from_backend_yaml(backend: str, overrides: Dict) -> ReInspectionConfig:
    """Build a ReInspectionConfig seeded from ``configs/backend/<backend>.yaml``."""
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    yaml_path = os.path.join(repo_root, "src", "configs", "backend", f"{backend}.yaml")
    field_names = {f.name for f in dataclasses.fields(ReInspectionConfig)}
    kw: Dict = {}
    if os.path.exists(yaml_path):
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        for k, v in raw.items():
            if k == "processor_path":
                kw["processor_name_or_path"] = v
            elif k in field_names:
                kw[k] = v
    kw.update(overrides)
    return ReInspectionConfig(**kw)


def _load_processor(backend: str, config: ReInspectionConfig):
    if backend == "internvl3":
        from src.backends.internvl3 import load_processor
        return load_processor(config)
    if backend == "llava_next":
        from src.backends.llava_next import load_processor
        return load_processor(config)
    raise ValueError(f"Unsupported backend: {backend}")


def _build_chat_fn(backend: str):
    if backend == "internvl3":
        from src.data.chat_template import build_chat_messages
        return build_chat_messages
    if backend == "llava_next":
        from src.data.llava_next_chat import build_chat_messages
        return build_chat_messages
    raise ValueError(f"Unsupported backend: {backend}")


# --------------------------------------------------------------------------- #
# Rendering                                                                      #
# --------------------------------------------------------------------------- #

def _header(m: TokenMap) -> str:
    return f'Token {m.index} | "{m.label}"'


def _context_header(maps: List[TokenMap], i: int, window: int = 4, width: int = 58) -> str:
    """Panel title showing token i bracketed in its surrounding sequence.

    e.g. ``Token 12 | …the [cat] sat on the…`` — same treatment as the
    interactive explorer's live token strip.
    """
    import textwrap

    labels = [m.label for m in maps]
    lo, hi = max(0, i - window), min(len(labels), i + window + 1)
    parts = (["…"] if lo > 0 else []) + [
        f"[{labels[j]}]" if j == i else labels[j] for j in range(lo, hi)
    ] + (["…"] if hi < len(labels) else [])
    ctx = textwrap.fill(" ".join(parts), width=width, max_lines=2, placeholder=" …")
    return f"Token {maps[i].index} | {ctx}"


def _overlay(ax, image, m: Optional[TokenMap], h: int, w: int, title: str, cmap: str):
    if m is None or m.grid is None:
        ax.imshow(image)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        return
    plot_attention_heatmap(image, m.grid, h, w, title=title, ax=ax, cmap=cmap, alpha=0.5)


def render_side_by_side(
    image: Image.Image,
    h: int,
    w: int,
    frozen_maps: List[TokenMap],
    ri_maps: List[TokenMap],
    save_path: str,
    suptitle: str,
    max_rows: int = 40,
):
    """One row per input token; columns = Frozen vs Re-Inspection overlays."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conds = [("Frozen", frozen_maps, FROZEN_CMAP), ("Re-Inspection", ri_maps, RI_CMAP)]
    conds = [(name, maps, cmap) for name, maps, cmap in conds if maps]
    if not conds:
        return
    ncols = len(conds)
    nrows = max(len(maps) for _, maps, _ in conds)
    truncated = nrows > max_rows
    nrows = min(nrows, max_rows)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    for c, (name, _, _) in enumerate(conds):
        axes[0, c].set_title(name, fontsize=12, fontweight="bold")

    for r in range(nrows):
        for c, (name, maps, cmap) in enumerate(conds):
            m = maps[r] if r < len(maps) else None
            # Token header lives on the left-most column to avoid repetition;
            # it shows the token bracketed in its surrounding sequence.
            title = _context_header(maps, r) if (m is not None and c == 0) else ""
            _overlay(axes[r, c], image, m, h, w, title, cmap)

    extra = "  (truncated)" if truncated else ""
    fig.suptitle(suptitle + extra, fontsize=14, fontweight="bold", y=1.005)
    plt.tight_layout()
    _savefig(fig, save_path)


def render_single_condition(
    image: Image.Image,
    h: int,
    w: int,
    maps: List[TokenMap],
    save_path: str,
    suptitle: str,
    cmap: str,
    ncols: int = 4,
    max_panels: int = 32,
):
    """Grid of per-token overlays for a single set of maps (e.g. generated tokens).

    ``max_panels <= 0`` renders every token (no cap).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not maps:
        return
    truncated = max_panels > 0 and len(maps) > max_panels
    if max_panels > 0:
        maps = maps[:max_panels]
    n = len(maps)
    cols = min(ncols, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    flat = axes.flatten()
    for i, m in enumerate(maps):
        _overlay(flat[i], image, m, h, w, _context_header(maps, i), cmap)
    for i in range(n, len(flat)):
        flat[i].axis("off")
    extra = "  (truncated)" if truncated else ""
    fig.suptitle(suptitle + extra, fontsize=14, fontweight="bold", y=1.005)
    plt.tight_layout()
    _savefig(fig, save_path)


def render_summary(
    image: Image.Image,
    results: Dict[str, dict],
    save_path: str,
    question: str = "",
):
    """Original + per-condition aggregate input-text-mean and generated-mean maps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("Original", None, None, None, None)]
    cmap_for = {"frozen": FROZEN_CMAP, "reinspection": RI_CMAP}
    for cond, res in results.items():
        cmap = cmap_for.get(cond, "viridis")
        panels.append((f"{cond}: input-text mean", res.get("input_mean"), res["h"], res["w"], cmap))
        panels.append((f"{cond}: generated mean", res.get("generated_mean"), res["h"], res["w"], cmap))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    for i, (title, grid, h, w, cmap) in enumerate(panels):
        ax = axes[0, i]
        if grid is None:
            ax.imshow(image)
            ax.set_title(title, fontsize=11)
            ax.axis("off")
        else:
            plot_attention_heatmap(image, grid, h, w, title=title, ax=ax, cmap=cmap, alpha=0.5)

    ans = "   ".join(f"[{c}] {r['answer']!r}" for c, r in results.items())
    q = f"Q: {question}\n" if question else ""
    fig.suptitle(f"Aggregate attention over image patches\n{q}{ans}", fontsize=13, y=1.06)
    plt.tight_layout()
    _savefig(fig, save_path)


def _savefig(fig, save_path: str):
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    fig.savefig(save_path.replace(".png", ".pdf"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {save_path}", flush=True)


def _load_npz_results(npz_path: str):
    """Rebuild the results dict (+meta) from a maps.npz for GPU-free re-rendering."""
    import re

    raw = np.load(npz_path, allow_pickle=False)
    meta = {k[5:]: str(raw[k]) for k in raw.files if k.startswith("meta/")}
    h, w = int(meta["h"]), int(meta["w"])
    pat = re.compile(
        r"^(?P<cond>[^/]+)/(?P<group>input_maps|generated_maps|r_maps)/(?P<idx>\d+)/(?P<field>grid|label)$"
    )
    results: Dict[str, dict] = {}
    acc: Dict[str, Dict[str, Dict[int, dict]]] = {}
    for key in raw.files:
        if key.startswith("meta/"):
            continue
        cond = key.split("/", 1)[0]
        res = results.setdefault(cond, dict(
            input_maps=[], generated_maps=[], r_maps=[], answer="",
            input_mean=None, generated_mean=None, h=h, w=w,
        ))
        if key.endswith("/answer"):
            res["answer"] = str(raw[key])
        elif key.endswith("/input_mean"):
            res["input_mean"] = raw[key]
        elif key.endswith("/generated_mean"):
            res["generated_mean"] = raw[key]
        else:
            m = pat.match(key)
            if m:
                acc.setdefault(cond, {}).setdefault(m["group"], {}) \
                   .setdefault(int(m["idx"]), {})[m["field"]] = raw[key]
    kind_for = {"input_maps": KIND_TEXT, "generated_maps": "generated", "r_maps": KIND_R}
    for cond, groups in acc.items():
        for group, entries in groups.items():
            results[cond][group] = [
                TokenMap(index=i, token_id=-1, label=str(e.get("label", "?")),
                         kind=kind_for[group], grid=np.asarray(e["grid"], dtype=np.float32))
                for i, e in sorted(entries.items()) if "grid" in e
            ]
    return results, meta


def _save_npz(results: Dict[str, dict], save_path: str, meta: dict):
    payload: Dict[str, object] = {}
    for cond, res in results.items():
        for group in ("input_maps", "r_maps", "generated_maps"):
            for m in res.get(group, []):
                payload[f"{cond}/{group}/{m.index}/grid"] = m.grid
                payload[f"{cond}/{group}/{m.index}/label"] = np.asarray(m.label, dtype="<U64")
        if res.get("input_mean") is not None:
            payload[f"{cond}/input_mean"] = res["input_mean"]
        if res.get("generated_mean") is not None:
            payload[f"{cond}/generated_mean"] = res["generated_mean"]
        payload[f"{cond}/answer"] = np.asarray(res["answer"], dtype="<U8192")
    for k, v in meta.items():
        payload[f"meta/{k}"] = np.asarray(v)
    np.savez(save_path, **payload)
    print(f"  saved {save_path}", flush=True)


# --------------------------------------------------------------------------- #
# Main                                                                           #
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["internvl3", "llava_next"], default="internvl3")
    p.add_argument("--image", default=None, help="Required unless --from_npz (then meta default).")
    p.add_argument("--question", default=None, help="Required unless --from_npz (then meta default).")
    p.add_argument("--from_npz", default=None,
                   help="Re-render figures from an existing maps.npz — no GPU/model needed; "
                        "backend/question/image default from its meta, figures land next to it.")
    p.add_argument("--conditions", default="frozen,reinspection",
                   help="Comma list subset of {frozen,reinspection}.")
    p.add_argument("--checkpoint_dir", default=None, help="Re-Inspection module dir (stage1/2).")
    p.add_argument("--lora_checkpoint_dir", default=None, help="Stage-2 LoRA dir (also carries module).")
    p.add_argument("--output_dir", default="outputs/token_attn")
    p.add_argument("--sample_name", default="sample")
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--layer_reduce", default="mean",
                   help="'mean' (all layers), 'last', an int layer index, or 'start:end' range.")
    p.add_argument("--max_input_tokens", type=int, default=-1,
                   help="Cap rendered input tokens (-1 = all).")
    p.add_argument("--max_panels", type=int, default=32,
                   help="Cap generated/R-token panels in static grids (0 = all tokens).")
    p.add_argument("--skip_special", action="store_true", help="Drop special tokens from input rows.")
    p.add_argument("--no_r_tokens", action="store_true", help="Skip R-token panels for reinspection.")
    p.add_argument("--system_prompt", default=None, help="Override the default system prompt.")
    p.add_argument("--n_queries", type=int, default=None,
                   help="Override n_queries (must match the checkpoint's module, e.g. 256 for nq256).")
    args = p.parse_args()

    if args.from_npz:
        results, meta = _load_npz_results(args.from_npz)
        backend = meta.get("backend", args.backend)
        question = args.question or meta.get("question", "?")
        image_path = args.image or meta.get("image", "")
        if not os.path.isfile(image_path):
            raise SystemExit(f"image not found: {image_path!r} (pass --image)")
        image = Image.open(image_path).convert("RGB")
        out_dir = os.path.dirname(os.path.abspath(args.from_npz))
        _render_figures(results, image, out_dir, backend=backend, question=question,
                        max_panels=args.max_panels)
        print(f"\nDone (re-render) → {out_dir}", flush=True)
        return
    if not (args.image and args.question):
        raise SystemExit("--image and --question are required unless --from_npz is given.")

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for c in conditions:
        if c not in ("frozen", "reinspection"):
            raise SystemExit(f"Unknown condition {c!r}; choose from frozen,reinspection.")

    layer_reduce = args.layer_reduce
    if isinstance(layer_reduce, str):
        if layer_reduce.isdigit() or (layer_reduce.startswith("-") and layer_reduce[1:].isdigit()):
            layer_reduce = int(layer_reduce)
        elif ":" in layer_reduce:
            a, b = layer_reduce.split(":")
            layer_reduce = (int(a), int(b))

    overrides = dict(
        backend=args.backend,
        attn_implementation="eager",  # required to materialise attention weights
        bf16=True,
        checkpoint_dir=args.checkpoint_dir,
        lora_checkpoint_dir=args.lora_checkpoint_dir,
    )
    if args.system_prompt is not None:
        overrides["system_prompt"] = args.system_prompt
    if args.n_queries is not None:
        overrides["n_queries"] = args.n_queries
    config = _config_from_backend_yaml(args.backend, overrides)

    processor = _load_processor(args.backend, config)
    build_chat = _build_chat_fn(args.backend)
    image = Image.open(args.image).convert("RGB")

    from src.evaluate import load_condition_model

    out_dir = os.path.join(args.output_dir, f"{args.backend}_{args.sample_name}")
    os.makedirs(out_dir, exist_ok=True)

    results: Dict[str, dict] = {}
    for condition in conditions:
        print(f"\n=== condition: {condition} ({args.backend}) ===", flush=True)
        if condition == "reinspection" and not (args.checkpoint_dir or args.lora_checkpoint_dir):
            print("  skipping reinspection: no --checkpoint_dir / --lora_checkpoint_dir given.", flush=True)
            continue
        model, is_ri = load_condition_model(
            args.backend, condition, config, processor,
            checkpoint_dir=args.checkpoint_dir,
            lora_checkpoint_dir=args.lora_checkpoint_dir,
            attn_implementation="eager",
        )
        model.eval()
        res = capture_token_attention(
            model, processor, args.backend, config,
            image_path=args.image, question=args.question,
            is_reinspection=is_ri, build_chat_messages=build_chat,
            max_new_tokens=args.max_new_tokens, layer_reduce=layer_reduce,
            max_input_tokens=args.max_input_tokens, skip_special=args.skip_special,
            include_r_tokens=not args.no_r_tokens,
        )
        results[condition] = res
        print(f"  answer: {res['answer']!r}  | input tokens: {len(res['input_maps'])}"
              f"  generated: {len(res['generated_maps'])}  R: {len(res['r_maps'])}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not results:
        raise SystemExit("No conditions produced results.")

    _render_figures(results, image, out_dir, backend=args.backend, question=args.question,
                    max_panels=args.max_panels)
    _save_npz(results, os.path.join(out_dir, "maps.npz"),
              meta=dict(backend=args.backend, question=args.question, image=args.image,
                        layer_reduce=str(layer_reduce), max_new_tokens=args.max_new_tokens,
                        checkpoint=(args.lora_checkpoint_dir or args.checkpoint_dir or ""),
                        h=results[next(iter(results))]["h"],
                        w=results[next(iter(results))]["w"]))
    print(f"\nDone → {out_dir}", flush=True)


def _render_figures(results: Dict[str, dict], image: Image.Image, out_dir: str,
                    backend: str, question: str, max_panels: int = 32):
    """All static figures for a captured/reloaded results dict."""
    h = next(iter(results.values()))["h"]
    w = next(iter(results.values()))["w"]
    grid_note = f"grid {h}×{w}" + (" (LLaVA base global view)" if backend == "llava_next" else "")

    answers_line = "   ".join(f"A ({c}): {r['answer']!r}" for c, r in results.items())
    render_side_by_side(
        image, h, w,
        results.get("frozen", {}).get("input_maps", []),
        results.get("reinspection", {}).get("input_maps", []),
        os.path.join(out_dir, "input_tokens.png"),
        suptitle=(f"Input token → image attention | {backend} | {grid_note}\n"
                  f"Q: {question}\n{answers_line}"),
    )
    for cond, res in results.items():
        cmap = FROZEN_CMAP if cond == "frozen" else RI_CMAP
        render_single_condition(
            image, h, w, res["generated_maps"],
            os.path.join(out_dir, f"generated_tokens_{cond}.png"),
            suptitle=(f"Generated token → image attention | {cond}\n"
                      f"Q: {question}\nA ({cond}): {res['answer']!r}"),
            cmap=cmap,
            max_panels=max_panels,
        )
        if res.get("r_maps"):
            render_single_condition(
                image, h, w, res["r_maps"],
                os.path.join(out_dir, f"r_tokens_{cond}.png"),
                suptitle=(f"Re-Inspection R token → image attention | {cond}\n"
                          f"Q: {question}\nA ({cond}): {res['answer']!r}"),
                cmap=RI_CMAP,
                max_panels=max_panels,
            )
    render_summary(image, results, os.path.join(out_dir, "summary.png"), question=question)


if __name__ == "__main__":
    main()
