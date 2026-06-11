"""Build an interactive HTML gallery from token-attention selection outputs.

Reads ``selection.json`` plus the per-instance PNG figures produced by
``select_and_visualize_vsr.py`` and writes ``index.html`` into the same directory.
With ``--compare_dir``, builds a side-by-side tiling-on vs tiling-off page.

Example::

    PYTHONPATH=. python scripts/analysis/build_token_attention_html.py \\
        --selection_dir outputs/token_attn/selection_s2e2 \\
        --image_root /data/datasets/vsr/images

    PYTHONPATH=. python scripts/analysis/build_token_attention_html.py \\
        --selection_dir outputs/token_attn/selection_s2e2 \\
        --compare_dir outputs/token_attn/selection_notile_e4 \\
        --compare_label_on "Tiling ON (s2_internvl e2)" \\
        --compare_label_off "Tiling OFF (s2_internvl_notile e4)" \\
        --output outputs/token_attn/compare_tiling_on_off/index.html

    PYTHONPATH=. python scripts/analysis/build_token_attention_html.py \\
        --runs_dir outputs/token_attn/notile_e4_cot
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil

import numpy as np
from typing import Dict, List, Optional, Tuple


FIGURE_KEYS = (
    ("summary", "Summary (frozen vs reinspection)"),
    ("input_tokens", "Input tokens (frozen vs reinspection)"),
    ("generated_tokens_frozen", "Generated tokens — frozen"),
    ("generated_tokens_reinspection", "Generated tokens — reinspection"),
)

COMPARE_FIGURE_KEYS = (
    ("summary", "Summary"),
    ("generated_tokens_reinspection", "Generated tokens (reinspection)"),
    ("input_tokens", "Input tokens"),
)

GALLERY_ASSETS_DIR = "gallery_assets"


def _find_instance_dir(selection_dir: str, group: str, img: str, idx: int) -> Optional[str]:
    stem = os.path.splitext(img)[0]
    prefix = f"{group}_{idx}_{stem}"
    group_dir = os.path.join(selection_dir, group)
    if not os.path.isdir(group_dir):
        return None
    exact = os.path.join(group_dir, prefix)
    if os.path.isdir(exact):
        return exact
    for name in os.listdir(group_dir):
        if name.endswith(stem):
            return os.path.join(group_dir, name)
    return None


def _rel(path: str, base: str) -> str:
    return os.path.relpath(path, base).replace(os.sep, "/")


def _link_image_for_http(
    image_abs: str,
    html_dir: str,
    asset_rel: str,
    cache: Dict[Tuple[str, str, str], str],
) -> str:
    """Symlink (or copy) an image under *html_dir* so ``http.server`` can serve it."""
    image_abs = os.path.abspath(image_abs)
    cache_key = (html_dir, asset_rel, image_abs)
    if cache_key in cache:
        return cache[cache_key]

    rel_src = ""
    if image_abs and os.path.isfile(image_abs):
        dest = os.path.join(html_dir, GALLERY_ASSETS_DIR, asset_rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if not os.path.lexists(dest):
            try:
                os.symlink(image_abs, dest)
            except OSError:
                try:
                    shutil.copy2(image_abs, dest)
                except OSError:
                    dest = ""
        if dest and os.path.lexists(dest):
            rel_src = f"{GALLERY_ASSETS_DIR}/{asset_rel.replace(os.sep, '/')}"

    cache[cache_key] = rel_src
    return rel_src


def _strip_vsr_question(question: str) -> str:
    m = re.search(r'"([^"]+)"', question)
    return m.group(1) if m else question


def _load_selection(selection_dir: str) -> tuple[dict, List[dict]]:
    manifest_path = os.path.join(selection_dir, "selection.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(manifest_path)
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    instances = manifest.get("selected", [])
    if not instances:
        raise ValueError(f"{manifest_path} has no 'selected' entries")
    return manifest, instances


def _index_by_image(selected: List[dict]) -> Dict[str, tuple[int, dict]]:
    return {item["img"]: (idx, item) for idx, item in enumerate(selected)}


def _merge_instances(selected_a: List[dict], selected_b: List[dict]) -> List[dict]:
    by_img: Dict[str, dict] = {}
    for item in selected_a:
        by_img[item["img"]] = {"img": item["img"], "a": item, "b": None}
    for item in selected_b:
        if item["img"] in by_img:
            by_img[item["img"]]["b"] = item
        else:
            by_img[item["img"]] = {"img": item["img"], "a": None, "b": item}

    def _sort_key(row: dict):
        ref = row.get("a") or row.get("b")
        group_rank = 0 if ref and ref.get("group") == "correct" else 1
        return (group_rank, row["img"])

    return sorted(by_img.values(), key=_sort_key)


def _render_figure_png(
    png: str,
    html_base: str,
    caption: str,
    caption_prefix: str = "",
    *,
    scrollable: bool = False,
) -> str:
    full_caption = f"{caption_prefix}{caption}" if caption_prefix else caption
    block_cls = "fig-block fig-scroll" if scrollable else "fig-block"
    return (
        f'<figure class="{block_cls}">'
        f'<figcaption>{html.escape(full_caption)}</figcaption>'
        f'<img class="zoomable" src="{html.escape(_rel(png, html_base))}" '
        f'alt="{html.escape(full_caption)}" loading="lazy" '
        f'data-caption="{html.escape(full_caption)}">'
        f"</figure>"
    )


def _render_instance_figures(
    inst_dir: Optional[str],
    html_base: str,
    keys=FIGURE_KEYS,
    caption_prefix: str = "",
) -> str:
    if not inst_dir:
        return '<p class="missing">Not in this selection run.</p>'
    blocks: List[str] = []
    for key, caption in keys:
        png = os.path.join(inst_dir, f"{key}.png")
        if os.path.isfile(png):
            scrollable = key.startswith("generated_tokens") or key == "input_tokens"
            blocks.append(
                _render_figure_png(
                    png, html_base, caption, caption_prefix, scrollable=scrollable
                )
            )
    return "".join(blocks) if blocks else '<p class="missing">No figures found.</p>'


def _render_source_thumb(
    image_abs: str,
    html_base: str,
    asset_rel: str,
    cache: Dict[Tuple[str, str, str], str],
) -> str:
    thumb_rel = _link_image_for_http(image_abs, html_base, asset_rel, cache)
    if not thumb_rel:
        return ""
    caption = os.path.basename(asset_rel)
    return (
        f'<div class="source-image">'
        f'<img class="zoomable" src="{html.escape(thumb_rel)}" alt="source image" '
        f'loading="lazy" data-caption="{html.escape(caption)}">'
        f"</div>"
    )


def _load_run_meta(run_dir: str) -> dict:
    meta: dict = {}
    npz_path = os.path.join(run_dir, "maps.npz")
    if not os.path.isfile(npz_path):
        return meta
    data = np.load(npz_path, allow_pickle=True)
    for key in data.files:
        if key.startswith("meta/"):
            val = data[key]
            meta[key.split("/", 1)[1]] = val.item() if val.ndim == 0 else str(val)
        elif key.endswith("/answer"):
            cond = key.split("/", 1)[0]
            val = data[key]
            meta[f"{cond}_answer"] = val.item() if val.ndim == 0 else str(val)
    return meta


def discover_runs(runs_dir: str, run_names: Optional[List[str]] = None) -> List[dict]:
    """Find capture subdirs (each has ``summary.png``) under ``runs_dir``."""
    found: List[dict] = []
    names = run_names or sorted(os.listdir(runs_dir))
    for name in names:
        run_dir = os.path.join(runs_dir, name)
        if not os.path.isdir(run_dir) or not os.path.isfile(os.path.join(run_dir, "summary.png")):
            continue
        meta = _load_run_meta(run_dir)
        image_path = meta.get("image", "")
        img_base = os.path.basename(image_path) if image_path else name
        found.append({
            "run_id": name,
            "run_dir": run_dir,
            "question": meta.get("question", ""),
            "image_path": image_path,
            "img": img_base,
            "frozen_answer": meta.get("frozen_answer", ""),
            "reinspection_answer": meta.get("reinspection_answer", ""),
        })
    return found


def build_runs_gallery_html(
    runs_dir: str,
    runs: List[dict],
    html_base: str,
    title: Optional[str] = None,
    subtitle: str = "",
) -> str:
    """Gallery for flat ``<runs_dir>/<run_id>/`` capture folders (CoT demos, benches)."""
    title = title or f"Token attention — {os.path.basename(runs_dir.rstrip('/'))}"
    checkpoint = ""
    if runs:
        m0 = _load_run_meta(runs[0]["run_dir"])
        checkpoint = m0.get("checkpoint", "")

    asset_cache: Dict[Tuple[str, str, str], str] = {}
    cards: List[str] = []
    for idx, run in enumerate(runs):
        inst_dir = run["run_dir"]
        statement = _strip_vsr_question(run.get("question", ""))
        img_path = run.get("image_path", "")
        thumb_html = ""
        if img_path and os.path.isfile(img_path):
            ext = os.path.splitext(img_path)[1] or ".jpg"
            thumb_html = _render_source_thumb(
                img_path,
                html_base,
                f"{run['run_id']}/source{ext}",
                asset_cache,
            )
        figures_html = _render_instance_figures(inst_dir, html_base)
        interactive = os.path.join(inst_dir, "interactive.html")
        interactive_link = ""
        if os.path.isfile(interactive):
            interactive_link = (
                f'<p class="interactive-link"><a href="{html.escape(_rel(interactive, html_base))}">'
                f"Open interactive explorer (all CoT tokens)</a></p>"
            )

        def _short_answer(text: str, n: int = 280) -> str:
            text = (text or "").strip()
            if len(text) <= n:
                return text
            return text[: n - 1] + "…"

        cards.append(
            f"""
            <article class="card" data-group="demo" id="run-{idx}">
              <header class="card-header">
                <h2>{html.escape(run["run_id"])}</h2>
                {interactive_link}
                <dl class="meta">
                  <div><dt>Statement</dt><dd>{html.escape(statement)}</dd></div>
                  <div><dt>Frozen answer</dt><dd class="pred">{html.escape(_short_answer(run.get("frozen_answer", "")))}</dd></div>
                  <div><dt>Reinspection answer</dt><dd class="pred">{html.escape(_short_answer(run.get("reinspection_answer", "")))}</dd></div>
                </dl>
              </header>
              <div class="card-body">
                {thumb_html}
                <div class="figures">{figures_html}</div>
              </div>
            </article>
            """
        )

    shell = build_html("", {"checkpoint": checkpoint, "backend": "internvl3"}, [], "")
    style_start = shell.index("<style>")
    style_end = shell.index("</style>") + len("</style>")
    script_start = shell.index("<script>")
    lightbox_start = shell.index('<div class="lightbox"')
    lightbox_end = shell.index("</div>\n\n  <script>")

    sub = f"<p>{html.escape(subtitle)}</p>" if subtitle else ""
    ckpt_line = f"<p>Checkpoint: <code>{html.escape(checkpoint)}</code></p>" if checkpoint else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  {shell[style_start:style_end]}
  <style>
    .interactive-link {{ margin: 0.35rem 0 0.6rem; font-size: 0.9rem; }}
    .interactive-link a {{ font-weight: 600; }}
  </style>
</head>
<body>
  <div class="wrap">
    <header class="page">
      <h1>{html.escape(title)}</h1>
      <p>Per-token text→image decoder attention · eval CoT protocol · frozen (viridis) vs reinspection (inferno)</p>
      {sub}
      {ckpt_line}
      <p>{len(runs)} captures under <code>{html.escape(os.path.basename(runs_dir))}</code></p>
      <p class="hero-note">Click any image to open · scroll to pan · Ctrl+scroll to zoom · double-click to reset</p>
    </header>
    <div class="grid" id="cards">
      {"".join(cards)}
    </div>
    <footer>
      Generated by <code>scripts/analysis/build_token_attention_html.py --runs_dir …</code>
    </footer>
  </div>
  {shell[lightbox_start:lightbox_end]}
  {shell[script_start:]}
</body>
</html>
"""


def _compare_side_column(
    label: str,
    item: Optional[dict],
    selected: List[dict],
    selection_dir: str,
    html_base: str,
    css_class: str,
) -> str:
    if item is None:
        return (
            f'<div class="compare-col {css_class}">'
            f'<h3>{html.escape(label)}</h3>'
            f'<p class="missing">Instance not selected in this run.</p>'
            f"</div>"
        )
    idx, _ = _index_by_image(selected)[item["img"]]
    inst_dir = _find_instance_dir(selection_dir, item["group"], item["img"], idx)
    badge_cls = "badge-ok" if item["group"] == "correct" else "badge-fail"
    badge_text = "Correct" if item["group"] == "correct" else "Wrong"
    return (
        f'<div class="compare-col {css_class}">'
        f'<h3>{html.escape(label)}</h3>'
        f'<p class="col-checkpoint"><code>{html.escape(os.path.basename(selection_dir))}</code></p>'
        f'<span class="badge {badge_cls}">{badge_text}</span> '
        f'<span class="col-pred">RI = <strong>{html.escape(item.get("ri_pred", ""))}</strong></span>'
        f'<div class="figures compare-figures">'
        f'{_render_instance_figures(inst_dir, html_base, COMPARE_FIGURE_KEYS, caption_prefix=f"{label} · ")}'
        f"</div></div>"
    )


def build_html(
    selection_dir: str,
    manifest: dict,
    instances: List[dict],
    image_root: str,
) -> str:
    checkpoint = manifest.get("checkpoint", "")
    backend = manifest.get("backend", "")
    title = f"Token attention — {os.path.basename(selection_dir.rstrip('/'))}"

    montage_png = os.path.join(selection_dir, "montage.png")
    montage_rel = _rel(montage_png, selection_dir) if os.path.isfile(montage_png) else None

    asset_cache: Dict[Tuple[str, str, str], str] = {}
    cards: List[str] = []
    for idx, item in enumerate(instances):
        group = item["group"]
        img = item["img"]
        inst_dir = _find_instance_dir(selection_dir, group, img, idx)
        label = "correct" if group == "correct" else "failed"
        badge_cls = "badge-ok" if group == "correct" else "badge-fail"
        badge_text = "Reinspection correct" if group == "correct" else "Reinspection wrong"
        statement = _strip_vsr_question(item.get("question", ""))

        figures_html = _render_instance_figures(inst_dir, selection_dir)
        thumb_html = _render_source_thumb(
            os.path.join(image_root, img),
            selection_dir,
            f"vsr/{img}",
            asset_cache,
        )

        cards.append(
            f"""
            <article class="card" data-group="{html.escape(group)}" id="inst-{idx}">
              <header class="card-header">
                <span class="badge {badge_cls}">{html.escape(badge_text)}</span>
                <h2>{html.escape(img)}</h2>
                <dl class="meta">
                  <div><dt>Statement</dt><dd>{html.escape(statement)}</dd></div>
                  <div><dt>Ground truth</dt><dd class="gt">{html.escape(item.get("gt", ""))}</dd></div>
                  <div><dt>Reinspection</dt><dd class="pred">{html.escape(item.get("ri_pred", ""))}</dd></div>
                </dl>
              </header>
              <div class="card-body">
                {thumb_html}
                <div class="figures">{figures_html}</div>
              </div>
            </article>
            """
        )

    n_correct = sum(1 for x in instances if x["group"] == "correct")
    n_failed = sum(1 for x in instances if x["group"] == "failed")

    montage_section = ""
    if montage_rel:
        montage_section = f"""
        <section class="hero">
          <h2>Overview montage</h2>
          <p class="hero-note">Top row: reinspection correct ({n_correct}). Bottom row: reinspection wrong ({n_failed}). Heatmaps show mean answer-token attention (inferno overlay).</p>
          <img class="montage zoomable" src="{html.escape(montage_rel)}" alt="2×5 montage of reinspection attention"
               data-caption="Overview montage">
        </section>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #0f1117;
      --panel: #171a22;
      --border: #2a3040;
      --text: #e8ecf4;
      --muted: #9aa3b5;
      --accent: #6ea8fe;
      --ok: #3dd68c;
      --fail: #ff6b6b;
      --ok-bg: rgba(61, 214, 140, 0.12);
      --fail-bg: rgba(255, 107, 107, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }}
    a {{ color: var(--accent); }}
    .wrap {{ max-width: 1400px; margin: 0 auto; padding: 1.5rem; }}
    header.page {{
      margin-bottom: 1.5rem;
      padding-bottom: 1rem;
      border-bottom: 1px solid var(--border);
    }}
    header.page h1 {{ margin: 0 0 0.35rem; font-size: 1.6rem; }}
    header.page p {{ margin: 0.2rem 0; color: var(--muted); font-size: 0.95rem; }}
    .hero {{ margin-bottom: 2rem; }}
    .hero-note {{ color: var(--muted); max-width: 72ch; }}
    .montage {{
      width: 100%;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: #000;
    }}
    .zoomable {{
      cursor: zoom-in;
    }}
    .zoomable:hover {{
      outline: 2px solid var(--accent);
      outline-offset: -2px;
    }}
    .lightbox {{
      display: none;
      position: fixed;
      inset: 0;
      z-index: 1000;
      background: rgba(6, 8, 12, 0.94);
      flex-direction: column;
    }}
    .lightbox.open {{ display: flex; }}
    .lightbox-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
      padding: 0.65rem 1rem;
      border-bottom: 1px solid var(--border);
      background: rgba(15, 17, 23, 0.95);
      flex-shrink: 0;
    }}
    .lightbox-caption {{
      margin: 0;
      font-size: 0.9rem;
      color: var(--muted);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .lightbox-controls {{
      display: flex;
      gap: 0.35rem;
      flex-shrink: 0;
    }}
    .lightbox-controls button {{
      background: var(--panel);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 6px;
      min-width: 2.2rem;
      height: 2rem;
      cursor: pointer;
      font-size: 0.95rem;
    }}
    .lightbox-controls button:hover {{ border-color: var(--accent); }}
    .lightbox-viewport {{
      flex: 1;
      overflow: hidden;
      position: relative;
      cursor: grab;
      touch-action: none;
    }}
    .lightbox-viewport.dragging {{ cursor: grabbing; }}
    .lightbox-stage {{
      position: absolute;
      inset: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      transform-origin: center center;
      will-change: transform;
    }}
    .lightbox-stage img {{
      max-width: none;
      max-height: none;
      user-select: none;
      -webkit-user-drag: none;
      pointer-events: none;
    }}
    .lightbox-hint {{
      padding: 0.4rem 1rem 0.65rem;
      font-size: 0.78rem;
      color: var(--muted);
      text-align: center;
      flex-shrink: 0;
    }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      align-items: center;
      margin-bottom: 1.25rem;
    }}
    .toolbar button {{
      background: var(--panel);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 0.45rem 0.9rem;
      cursor: pointer;
      font-size: 0.9rem;
    }}
    .toolbar button.active {{
      background: var(--accent);
      color: #0b1020;
      border-color: var(--accent);
      font-weight: 600;
    }}
    .grid {{ display: grid; gap: 1.25rem; }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
    }}
    .card.hidden {{ display: none; }}
    .card-header {{ padding: 1rem 1.1rem 0.75rem; border-bottom: 1px solid var(--border); }}
    .card-header h2 {{ margin: 0.35rem 0 0.6rem; font-size: 1.05rem; word-break: break-all; }}
    .badge {{
      display: inline-block;
      font-size: 0.75rem;
      font-weight: 700;
      letter-spacing: 0.03em;
      text-transform: uppercase;
      padding: 0.2rem 0.55rem;
      border-radius: 6px;
    }}
    .badge-ok {{ background: var(--ok-bg); color: var(--ok); }}
    .badge-fail {{ background: var(--fail-bg); color: var(--fail); }}
    .meta {{ display: grid; gap: 0.45rem; margin: 0; }}
    .meta div {{ display: grid; grid-template-columns: 7rem 1fr; gap: 0.5rem; }}
    .meta dt {{ margin: 0; color: var(--muted); font-size: 0.82rem; }}
    .meta dd {{ margin: 0; font-size: 0.92rem; }}
    .meta .gt {{ color: var(--ok); font-weight: 600; }}
    .meta .pred {{ font-weight: 600; }}
    .card-body {{
      display: grid;
      grid-template-columns: minmax(180px, 240px) 1fr;
      gap: 1rem;
      padding: 1rem;
    }}
    @media (max-width: 900px) {{
      .card-body {{ grid-template-columns: 1fr; }}
    }}
    .source-image img {{
      width: 100%;
      border-radius: 8px;
      border: 1px solid var(--border);
    }}
    .figures {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 0.85rem;
    }}
    .fig-block {{
      margin: 0;
      background: #0c0e14;
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
    }}
    .fig-block figcaption {{
      padding: 0.45rem 0.6rem;
      font-size: 0.8rem;
      color: var(--muted);
      border-bottom: 1px solid var(--border);
    }}
    .fig-block img {{
      display: block;
      width: 100%;
      height: auto;
    }}
    .fig-block.fig-scroll {{
      max-height: min(90vh, 1400px);
      overflow: auto;
    }}
    .fig-block.fig-scroll img {{
      width: auto;
      max-width: 100%;
      min-width: min(100%, 640px);
    }}
    .missing {{ color: var(--muted); margin: 0; }}
    footer {{ margin-top: 2rem; color: var(--muted); font-size: 0.85rem; }}
    .montage-row {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
    }}
    @media (max-width: 900px) {{ .montage-row {{ grid-template-columns: 1fr; }} }}
    .montage-col h3 {{ margin: 0 0 0.5rem; font-size: 0.95rem; }}
    .compare-columns {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
    }}
    @media (max-width: 1000px) {{ .compare-columns {{ grid-template-columns: 1fr; }} }}
    .compare-col {{
      background: #0c0e14;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 0.75rem;
    }}
    .compare-col.tiling-on {{ border-top: 3px solid #6ea8fe; }}
    .compare-col.tiling-off {{ border-top: 3px solid #f0ad4e; }}
    .compare-col h3 {{ margin: 0 0 0.35rem; font-size: 0.92rem; }}
    .col-checkpoint {{ margin: 0 0 0.45rem; font-size: 0.78rem; color: var(--muted); }}
    .col-pred {{ font-size: 0.85rem; color: var(--muted); }}
    .compare-figures {{ grid-template-columns: 1fr; }}
    .badge-diff {{ background: rgba(110, 168, 254, 0.15); color: var(--accent); margin-left: 0.35rem; }}
    .badge-shared {{ background: rgba(154, 163, 181, 0.15); color: var(--muted); margin-left: 0.35rem; }}
  </style>
</head>
<body>
  <div class="wrap">
    <header class="page">
      <h1>{html.escape(title)}</h1>
      <p>Per-token text→image decoder attention · backend <code>{html.escape(backend)}</code></p>
      <p>Checkpoint: <code>{html.escape(checkpoint)}</code></p>
      <p>{len(instances)} VSR instances · frozen (viridis) vs reinspection (inferno)</p>
      <p class="hero-note">Click any image to open · scroll to pan · Ctrl+scroll to zoom · double-click to reset</p>
    </header>

    {montage_section}

    <div class="toolbar" role="tablist" aria-label="Filter instances">
      <button type="button" class="filter-btn active" data-filter="all">All ({len(instances)})</button>
      <button type="button" class="filter-btn" data-filter="correct">Correct ({n_correct})</button>
      <button type="button" class="filter-btn" data-filter="failed">Wrong ({n_failed})</button>
    </div>

    <div class="grid" id="cards">
      {"".join(cards)}
    </div>

    <footer>
      Generated by <code>scripts/analysis/build_token_attention_html.py</code>.
      Open this file from the selection output directory so relative image paths resolve.
    </footer>
  </div>

  <div class="lightbox" id="lightbox" aria-hidden="true">
    <div class="lightbox-header">
      <p class="lightbox-caption" id="lightbox-caption"></p>
      <div class="lightbox-controls">
        <button type="button" id="zoom-out" title="Zoom out (−)">−</button>
        <button type="button" id="zoom-reset" title="Reset (0)">100%</button>
        <button type="button" id="zoom-in" title="Zoom in (+)">+</button>
        <button type="button" id="zoom-close" title="Close (Esc)">✕</button>
      </div>
    </div>
    <div class="lightbox-viewport" id="lightbox-viewport">
      <div class="lightbox-stage" id="lightbox-stage">
        <img id="lightbox-img" alt="">
      </div>
    </div>
    <p class="lightbox-hint">Scroll to pan · Ctrl+scroll (or +/−) to zoom · drag to pan · double-click to fit · Esc to close</p>
  </div>

  <script>
    const buttons = document.querySelectorAll('.filter-btn');
    const cards = document.querySelectorAll('.card');
    buttons.forEach(btn => {{
      btn.addEventListener('click', () => {{
        buttons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const f = btn.dataset.filter;
        cards.forEach(card => {{
          const show = f === 'all' || card.dataset.group === f;
          card.classList.toggle('hidden', !show);
        }});
      }});
    }});

    const lightbox = document.getElementById('lightbox');
    const viewport = document.getElementById('lightbox-viewport');
    const stage = document.getElementById('lightbox-stage');
    const lbImg = document.getElementById('lightbox-img');
    const caption = document.getElementById('lightbox-caption');
    const resetBtn = document.getElementById('zoom-reset');

    let scale = 1;
    let tx = 0;
    let ty = 0;
    let dragging = false;
    let lastX = 0;
    let lastY = 0;

    function clampScale(s) {{
      return Math.min(12, Math.max(0.25, s));
    }}

    function applyTransform() {{
      stage.style.transform = `translate(${{tx}}px, ${{ty}}px) scale(${{scale}})`;
      resetBtn.textContent = `${{Math.round(scale * 100)}}%`;
    }}

    function fitImage() {{
      scale = 1;
      tx = 0;
      ty = 0;
      applyTransform();
    }}

    function openLightbox(img) {{
      lbImg.src = img.currentSrc || img.src;
      caption.textContent = img.dataset.caption || img.alt || '';
      fitImage();
      lightbox.classList.add('open');
      lightbox.setAttribute('aria-hidden', 'false');
      document.body.style.overflow = 'hidden';
    }}

    function closeLightbox() {{
      lightbox.classList.remove('open');
      lightbox.setAttribute('aria-hidden', 'true');
      document.body.style.overflow = '';
      lbImg.removeAttribute('src');
    }}

    document.querySelectorAll('.zoomable').forEach(img => {{
      img.addEventListener('click', () => openLightbox(img));
    }});

    document.getElementById('zoom-in').addEventListener('click', () => {{
      scale = clampScale(scale * 1.25);
      applyTransform();
    }});
    document.getElementById('zoom-out').addEventListener('click', () => {{
      scale = clampScale(scale / 1.25);
      applyTransform();
    }});
    document.getElementById('zoom-reset').addEventListener('click', fitImage);
    document.getElementById('zoom-close').addEventListener('click', closeLightbox);

    viewport.addEventListener('wheel', (e) => {{
      e.preventDefault();
      if (e.ctrlKey || e.metaKey) {{
        const delta = e.deltaY < 0 ? 1.12 : 1 / 1.12;
        const prev = scale;
        scale = clampScale(scale * delta);
        const rect = viewport.getBoundingClientRect();
        const cx = e.clientX - rect.left - rect.width / 2;
        const cy = e.clientY - rect.top - rect.height / 2;
        const ratio = scale / prev;
        tx = cx - (cx - tx) * ratio;
        ty = cy - (cy - ty) * ratio;
      }} else {{
        tx -= e.deltaX;
        ty -= e.deltaY;
      }}
      applyTransform();
    }}, {{ passive: false }});

    viewport.addEventListener('pointerdown', (e) => {{
      dragging = true;
      lastX = e.clientX;
      lastY = e.clientY;
      viewport.classList.add('dragging');
      viewport.setPointerCapture(e.pointerId);
    }});
    viewport.addEventListener('pointermove', (e) => {{
      if (!dragging) return;
      tx += e.clientX - lastX;
      ty += e.clientY - lastY;
      lastX = e.clientX;
      lastY = e.clientY;
      applyTransform();
    }});
    function endDrag(e) {{
      if (!dragging) return;
      dragging = false;
      viewport.classList.remove('dragging');
      try {{ viewport.releasePointerCapture(e.pointerId); }} catch (_) {{}}
    }}
    viewport.addEventListener('pointerup', endDrag);
    viewport.addEventListener('pointercancel', endDrag);
    viewport.addEventListener('dblclick', fitImage);

    lightbox.addEventListener('click', (e) => {{
      if (e.target === lightbox) closeLightbox();
    }});

    document.addEventListener('keydown', (e) => {{
      if (!lightbox.classList.contains('open')) return;
      if (e.key === 'Escape') closeLightbox();
      if (e.key === '+' || e.key === '=') {{
        scale = clampScale(scale * 1.25);
        applyTransform();
      }}
      if (e.key === '-') {{
        scale = clampScale(scale / 1.25);
        applyTransform();
      }}
      if (e.key === '0') fitImage();
    }});
  </script>
</body>
</html>
"""


def build_compare_html(
    dir_on: str,
    manifest_on: dict,
    selected_on: List[dict],
    dir_off: str,
    manifest_off: dict,
    selected_off: List[dict],
    html_base: str,
    image_root: str,
    label_on: str,
    label_off: str,
) -> str:
    title = "Token attention — tiling ON vs OFF"
    merged = _merge_instances(selected_on, selected_off)
    ckpt_on = manifest_on.get("checkpoint", "")
    ckpt_off = manifest_off.get("checkpoint", "")

    n_shared = sum(1 for r in merged if r["a"] and r["b"])
    n_pred_diff = sum(
        1 for r in merged
        if r["a"] and r["b"] and r["a"].get("ri_pred") != r["b"].get("ri_pred")
    )
    n_both_correct = sum(
        1 for r in merged
        if r["a"] and r["b"]
        and r["a"].get("group") == "correct" and r["b"].get("group") == "correct"
    )

    montage_on = os.path.join(dir_on, "montage.png")
    montage_off = os.path.join(dir_off, "montage.png")
    montage_section = ""
    if os.path.isfile(montage_on) or os.path.isfile(montage_off):
        cols = []
        for path, label in ((montage_on, label_on), (montage_off, label_off)):
            if os.path.isfile(path):
                cols.append(
                    f'<div class="montage-col"><h3>{html.escape(label)}</h3>'
                    f'<img class="montage zoomable" src="{html.escape(_rel(path, html_base))}" '
                    f'alt="{html.escape(label)} montage" data-caption="{html.escape(label)} montage"></div>'
                )
        montage_section = f"""
        <section class="hero">
          <h2>Overview montages</h2>
          <p class="hero-note">Same layout in each: top row = reinspection correct, bottom = wrong. Inference uses single-tile (no crop) in both runs; difference is the <em>training</em> checkpoint (tiling on vs off).</p>
          <div class="montage-row">{"".join(cols)}</div>
        </section>
        """

    asset_cache: Dict[Tuple[str, str, str], str] = {}
    cards: List[str] = []
    for idx, row in enumerate(merged):
        img = row["img"]
        item_on, item_off = row.get("a"), row.get("b")
        ref = item_on or item_off
        statement = _strip_vsr_question(ref.get("question", ""))
        gt = ref.get("gt", "")

        in_both = item_on is not None and item_off is not None
        pred_diff = in_both and item_on.get("ri_pred") != item_off.get("ri_pred")
        if item_on and item_off:
            group = item_on["group"] if item_on["group"] == item_off["group"] else "mixed"
        else:
            group = (item_on or item_off)["group"]

        badges = []
        if in_both:
            badges.append('<span class="badge badge-shared">in both runs</span>')
        if pred_diff:
            badges.append('<span class="badge badge-diff">prediction differs</span>')

        pred_on = item_on.get("ri_pred", "—") if item_on else "—"
        pred_off = item_off.get("ri_pred", "—") if item_off else "—"

        col_on = _compare_side_column(label_on, item_on, selected_on, dir_on, html_base, "tiling-on")
        col_off = _compare_side_column(label_off, item_off, selected_off, dir_off, html_base, "tiling-off")
        thumb = _render_source_thumb(
            os.path.join(image_root, img),
            html_base,
            f"vsr/{img}",
            asset_cache,
        )

        cards.append(
            f"""
            <article class="card" id="cmp-{idx}"
              data-group="{html.escape(group)}"
              data-in-both="{'true' if in_both else 'false'}"
              data-pred-diff="{'true' if pred_diff else 'false'}">
              <header class="card-header">
                <h2>{html.escape(img)} {''.join(badges)}</h2>
                <dl class="meta">
                  <div><dt>Statement</dt><dd>{html.escape(statement)}</dd></div>
                  <div><dt>Ground truth</dt><dd class="gt">{html.escape(gt)}</dd></div>
                  <div><dt>{html.escape(label_on)}</dt><dd class="pred">{html.escape(pred_on)}</dd></div>
                  <div><dt>{html.escape(label_off)}</dt><dd class="pred">{html.escape(pred_off)}</dd></div>
                </dl>
              </header>
              <div class="card-body compare-card-body">
                {thumb}
                <div class="compare-columns">{col_on}{col_off}</div>
              </div>
            </article>
            """
        )

    return _compare_page(
        title, ckpt_on, ckpt_off, label_on, label_off, len(merged),
        n_shared, n_pred_diff, n_both_correct, montage_section, cards,
    )


def _compare_page(
    title: str,
    ckpt_on: str,
    ckpt_off: str,
    label_on: str,
    label_off: str,
    n_total: int,
    n_shared: int,
    n_pred_diff: int,
    n_both_correct: int,
    montage_section: str,
    cards: List[str],
) -> str:
    """Assemble compare HTML using the same shell as ``build_html``."""
    # Pull styles/lightbox/script from a minimal single-instance build_html call pattern.
    shell = build_html("", {"checkpoint": "", "backend": "internvl3"}, [], "")
    style_start = shell.index("<style>")
    style_end = shell.index("</style>") + len("</style>")
    script_start = shell.index("<script>")
    page_styles = shell[style_start:style_end]
    page_script = shell[script_start:]

    compare_card_body_css = """
    .compare-card-body { grid-template-columns: minmax(160px, 200px) 1fr; }
    """
    page_styles = page_styles.replace("</style>", compare_card_body_css + "\n  </style>")

    filter_script = """
    const cmpCards = document.querySelectorAll('.card');
    document.querySelectorAll('.filter-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const f = btn.dataset.filter;
        cmpCards.forEach(card => {
          let show = true;
          if (f === 'shared') show = card.dataset.inBoth === 'true';
          else if (f === 'pred-diff') show = card.dataset.predDiff === 'true';
          else if (f === 'correct') show = card.dataset.group === 'correct';
          else if (f === 'failed') show = card.dataset.group === 'failed' || card.dataset.group === 'mixed';
          card.classList.toggle('hidden', !show);
        });
      });
    });
    """
    page_script = page_script.replace(
        "const buttons = document.querySelectorAll('.filter-btn');",
        filter_script + "\n    const buttons = document.querySelectorAll('.filter-btn');",
    )
    # Disable duplicate filter handler in single-mode script
    page_script = page_script.replace(
        "buttons.forEach(btn => {",
        "/* single-mode filter disabled */ if (false) buttons.forEach(btn => {",
        1,
    )

    lightbox_start = shell.index('<div class="lightbox"')
    lightbox_end = shell.index("</div>\n\n  <script>")
    lightbox_html = shell[lightbox_start:lightbox_end]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  {page_styles}
</head>
<body>
  <div class="wrap">
    <header class="page">
      <h1>{html.escape(title)}</h1>
      <p>Side-by-side per-token text→image attention · InternVL3 reinspection</p>
      <p><strong>{html.escape(label_on)}</strong>: <code>{html.escape(ckpt_on)}</code></p>
      <p><strong>{html.escape(label_off)}</strong>: <code>{html.escape(ckpt_off)}</code></p>
      <p>{n_total} unique images · {n_shared} in both runs · {n_pred_diff} with different predictions · {n_both_correct} correct in both</p>
      <p class="hero-note">Click any image to open · scroll to pan · Ctrl+scroll to zoom · double-click to reset</p>
    </header>

    {montage_section}

    <div class="toolbar" role="tablist" aria-label="Filter instances">
      <button type="button" class="filter-btn active" data-filter="all">All ({n_total})</button>
      <button type="button" class="filter-btn" data-filter="shared">In both runs ({n_shared})</button>
      <button type="button" class="filter-btn" data-filter="pred-diff">Pred differs ({n_pred_diff})</button>
      <button type="button" class="filter-btn" data-filter="correct">Correct ({n_both_correct})</button>
      <button type="button" class="filter-btn" data-filter="failed">Wrong / mixed</button>
    </div>

    <div class="grid" id="cards">
      {"".join(cards)}
    </div>

    <footer>
      Generated by <code>scripts/analysis/build_token_attention_html.py --compare_dir …</code>
    </footer>
  </div>

  {lightbox_html}

  {page_script}
</body>
</html>
"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selection_dir", default="outputs/token_attn/selection_s2e2",
                   help="Tiling-ON (or primary) selection output directory")
    p.add_argument("--compare_dir", default=None,
                   help="Tiling-OFF (or secondary) selection directory for side-by-side compare page")
    p.add_argument("--compare_label_on", default="Tiling ON (s2_internvl e2)")
    p.add_argument("--compare_label_off", default="Tiling OFF (s2_internvl_notile e4)")
    p.add_argument("--image_root", default="/data/datasets/vsr/images")
    p.add_argument("--output", default=None, help="HTML output path")
    p.add_argument(
        "--runs_dir",
        default=None,
        help="Flat capture tree (subdirs with summary.png); writes index.html gallery",
    )
    p.add_argument(
        "--run_names",
        default=None,
        help="Comma-separated subdir names under --runs_dir (default: all with summary.png)",
    )
    p.add_argument("--gallery_title", default=None, help="Page title for --runs_dir mode")
    args = p.parse_args()

    if args.runs_dir:
        runs_dir = os.path.abspath(args.runs_dir)
        run_names = [n.strip() for n in args.run_names.split(",") if n.strip()] if args.run_names else None
        runs = discover_runs(runs_dir, run_names)
        if not runs:
            raise SystemExit(f"No captures with summary.png under {runs_dir}")
        out_path = args.output or os.path.join(runs_dir, "index.html")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        decode_note = "1024-token decode" if "1024" in runs_dir else "eval CoT capture"
        ckpt = _load_run_meta(runs[0]["run_dir"]).get("checkpoint", "") if runs else ""
        subtitle = (
            f"Eval CoT suffix · {decode_note} · "
            f"{ckpt or 'checkpoint in per-run meta'} · single-tile input · "
            "static grids show all captured tokens (use interactive.html for full decode)"
        )
        html_text = build_runs_gallery_html(
            runs_dir,
            runs,
            os.path.dirname(os.path.abspath(out_path)),
            title=args.gallery_title,
            subtitle=subtitle,
        )
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html_text)
        print(f"Wrote {out_path} ({len(runs)} runs)")
        return

    selection_dir = os.path.abspath(args.selection_dir)
    try:
        manifest, instances = _load_selection(selection_dir)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(e) from e

    if args.compare_dir:
        compare_dir = os.path.abspath(args.compare_dir)
        try:
            manifest_off, selected_off = _load_selection(compare_dir)
        except (FileNotFoundError, ValueError) as e:
            raise SystemExit(e) from e
        out_path = args.output or os.path.join(
            os.path.dirname(selection_dir), "compare_tiling_on_off", "index.html"
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        html_text = build_compare_html(
            selection_dir, manifest, instances,
            compare_dir, manifest_off, selected_off,
            os.path.dirname(os.path.abspath(out_path)),
            args.image_root,
            args.compare_label_on,
            args.compare_label_off,
        )
        merged = _merge_instances(instances, selected_off)
        print(f"Wrote {out_path} ({len(merged)} unique images, compare mode)")
    else:
        out_path = args.output or os.path.join(selection_dir, "index.html")
        html_text = build_html(selection_dir, manifest, instances, args.image_root)
        print(f"Wrote {out_path} ({len(instances)} instances)")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_text)


if __name__ == "__main__":
    main()
