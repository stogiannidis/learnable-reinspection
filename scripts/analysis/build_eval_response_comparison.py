"""Build an HTML comparison of eval responses across frozen, lora_only, and reinspection.

Reads per-sample JSON files written by ``src/evaluate.py`` (e.g.
``frozen_vsr_zeroshot_samples.json``) and joins rows by ``(benchmark, idx)``.
Each card shows the image, input prompt, ground truth, and the three model
responses with scored answers and correctness badges.

Usage::

    PYTHONPATH=. python scripts/analysis/build_eval_response_comparison.py \\
        --eval_dir outputs/internvl3/s2_internvl_1024 \\
        --benchmarks vsr_zeroshot \\
        --limit 20 \\
        --only-disagreements

    PYTHONPATH=. python scripts/analysis/build_eval_response_comparison.py \\
        --eval_dir outputs/internvl3/s2_internvl_1024 \\
        --output outputs/internvl3/s2_internvl_1024/response_comparison.html
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

BENCHMARK_CONFIGS = {
    "vsr": {"image_root": "vsr/images"},
    "gqa_spatial": {"image_root": "gqa_spatial/images"},
    "whatsup": {"image_root": "whatsup/images"},
    "3dsrbench": {"image_root": "3dsrbench/images"},
    "mindcube": {"image_root": "mindcube/images"},
    "blink": {"image_root": "blink/images"},
    "srbench": {"image_root": "srbench/images"},
    "qspatial": {"image_root": "qspatial/images"},
    "embspatial": {"image_root": "embspatial/images"},
    "realworldqa": {"image_root": "realworldqa/images"},
    "vsr_zeroshot": {"image_root": "vsr_zeroshot/images"},
    "cv_bench": {"image_root": "cv_bench/images"},
    "vstar_bench": {"image_root": "vstar_bench/images"},
    "mmvp": {"image_root": "mmvp/images"},
}

CONDITIONS = ("frozen", "lora_only", "reinspection")
CONDITION_LABELS = {
    "frozen": "Frozen",
    "lora_only": "LoRA only",
    "reinspection": "Re-Inspection",
}
SAMPLE_RE = re.compile(r"^(?P<condition>frozen|lora_only|reinspection)_(?P<benchmark>.+)_samples\.json$")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_data_path(path_like: str, data_root: str) -> str:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return str(path)
    candidates = [
        Path(data_root) / path,
        _repo_root() / path,
        Path.cwd() / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return str(candidates[0].resolve())


def discover_sample_files(eval_dir: str) -> Dict[str, Dict[str, str]]:
    """Return ``{benchmark: {condition: path}}`` for files in *eval_dir*."""
    out: Dict[str, Dict[str, str]] = defaultdict(dict)
    for name in sorted(os.listdir(eval_dir)):
        m = SAMPLE_RE.match(name)
        if not m:
            continue
        cond, bm = m.group("condition"), m.group("benchmark")
        out[bm][cond] = os.path.join(eval_dir, name)
    return dict(out)


def load_samples(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def join_samples(
    files_by_bm: Dict[str, Dict[str, str]],
    benchmarks: Optional[Iterable[str]] = None,
) -> List[dict]:
    """Join per-condition rows by ``(benchmark, idx)``."""
    selected = sorted(files_by_bm)
    if benchmarks:
        want = set(benchmarks)
        selected = [bm for bm in selected if bm in want]

    joined: List[dict] = []
    for bm in selected:
        cond_paths = files_by_bm[bm]
        by_idx: Dict[int, dict] = {}
        for cond in CONDITIONS:
            path = cond_paths.get(cond)
            if not path:
                continue
            for row in load_samples(path):
                idx = int(row["idx"])
                entry = by_idx.setdefault(
                    idx,
                    {
                        "benchmark": bm,
                        "idx": idx,
                        "image": row.get("image", ""),
                        "question": row.get("question", ""),
                        "prompted_question": row.get("prompted_question", ""),
                        "ground_truth": row.get("ground_truth", ""),
                        "conditions": {},
                    },
                )
                entry["conditions"][cond] = {
                    "model_output": row.get("model_output", ""),
                    "scored_answer": row.get("scored_answer", ""),
                    "correct": bool(row.get("correct", False)),
                }
                # Prefer shared metadata from any condition (should match).
                for key in ("image", "question", "prompted_question", "ground_truth"):
                    if row.get(key):
                        entry[key] = row[key]
        joined.extend(by_idx[i] for i in sorted(by_idx))
    return joined


def resolve_image_path(
    benchmark: str,
    image_name: str,
    data_root: str,
) -> str:
    """Absolute path to the benchmark image file."""
    rel = BENCHMARK_CONFIGS.get(benchmark, {}).get("image_root", f"{benchmark}/images")
    image_root = _resolve_data_path(rel, data_root)
    return os.path.join(image_root, image_name)


ASSETS_DIRNAME = "response_comparison_assets"


def _link_image_asset(
    benchmark: str,
    image_name: str,
    image_abs: str,
    html_dir: str,
    cache: Dict[Tuple[str, str], str],
) -> str:
    """Symlink image into *html_dir* so HTTP-served pages can load it."""
    key = (benchmark, image_name)
    if key in cache:
        return cache[key]

    rel_src = ""
    if image_abs and os.path.isfile(image_abs):
        assets_root = os.path.join(html_dir, ASSETS_DIRNAME, benchmark)
        os.makedirs(assets_root, exist_ok=True)
        link_path = os.path.join(assets_root, image_name)
        if not os.path.lexists(link_path):
            try:
                os.symlink(image_abs, link_path)
            except OSError:
                # Cross-device or permission issue: fall back to a same-dir copy attempt.
                try:
                    import shutil
                    shutil.copy2(image_abs, link_path)
                except OSError:
                    link_path = ""
        if link_path and os.path.lexists(link_path):
            rel_src = f"{ASSETS_DIRNAME}/{benchmark}/{image_name}"

    cache[key] = rel_src
    return rel_src


def _badge(correct: bool) -> str:
    if correct:
        return "<span class='badge ok'>correct</span>"
    return "<span class='badge bad'>incorrect</span>"


def _filter_rows(
    rows: List[dict],
    only_disagreements: bool,
    only_reinspection_wins: bool,
) -> List[dict]:
    out: List[dict] = []
    for row in rows:
        conds = row.get("conditions", {})
        correctness = {c: conds[c]["correct"] for c in CONDITIONS if c in conds}
        if not correctness:
            continue
        if only_disagreements:
            vals = list(correctness.values())
            if len(set(vals)) <= 1 and len(vals) > 1:
                continue
            if len(vals) == 1:
                continue
        if only_reinspection_wins:
            ri = correctness.get("reinspection")
            fr = correctness.get("frozen")
            if ri is not True or fr is not False:
                continue
        out.append(row)
    return out


def _summary_stats(rows: List[dict]) -> Dict[str, Tuple[int, int]]:
    stats: Dict[str, Tuple[int, int]] = {}
    for cond in CONDITIONS:
        correct = total = 0
        for row in rows:
            c = row.get("conditions", {}).get(cond)
            if c is None:
                continue
            total += 1
            correct += int(c["correct"])
        stats[cond] = (correct, total)
    return stats


def _stats_by_benchmark(rows: List[dict]) -> Dict[str, Dict[str, Tuple[int, int]]]:
    """Per-benchmark accuracy stats plus an ``all`` aggregate."""
    by_bm: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        by_bm[row["benchmark"]].append(row)
    out: Dict[str, Dict[str, Tuple[int, int]]] = {"all": _summary_stats(rows)}
    for bm in sorted(by_bm):
        out[bm] = _summary_stats(by_bm[bm])
    return out


def _format_stat_lines(stats: Dict[str, Tuple[int, int]]) -> str:
    lines: List[str] = []
    for cond in CONDITIONS:
        correct, total = stats.get(cond, (0, 0))
        acc = (100.0 * correct / total) if total else 0.0
        lines.append(
            f"<span class='stat cond-{cond}'><b>{html.escape(CONDITION_LABELS[cond])}</b>: "
            f"{correct}/{total} ({acc:.1f}%)</span>"
        )
    return "".join(lines)


def render_html(
    rows: List[dict],
    *,
    eval_dir: str,
    data_root: str,
    output: str,
    title: str,
) -> str:
    html_dir = os.path.dirname(os.path.abspath(output)) or "."
    os.makedirs(html_dir, exist_ok=True)

    benchmarks = sorted({row["benchmark"] for row in rows})
    bm_counts = {bm: sum(1 for r in rows if r["benchmark"] == bm) for bm in benchmarks}
    stats_by_bm = _stats_by_benchmark(rows)
    cards: List[str] = []
    image_cache: Dict[Tuple[str, str], str] = {}

    for row in rows:
        bm = row["benchmark"]
        idx = row["idx"]
        image_name = row.get("image", "")
        image_abs = resolve_image_path(bm, image_name, data_root)
        img_src = _link_image_asset(bm, image_name, image_abs, html_dir, image_cache)
        prompt = row.get("prompted_question") or row.get("question", "")

        img_html = (
            f"<img src='{html.escape(img_src)}' alt='sample image' loading='lazy'>"
            if img_src
            else f"<div class='missing-img'>image not found<br><small>{html.escape(image_abs)}</small></div>"
        )

        panels: List[str] = []
        for cond in CONDITIONS:
            cdata = row.get("conditions", {}).get(cond)
            if cdata is None:
                panels.append(
                    f"<div class='panel missing'><h3>{html.escape(CONDITION_LABELS[cond])}</h3>"
                    "<p><em>No data for this condition.</em></p></div>"
                )
                continue
            panels.append(
                f"<div class='panel cond-{cond}'>"
                f"<h3>{html.escape(CONDITION_LABELS[cond])} {_badge(cdata['correct'])}</h3>"
                f"<p class='scored'><b>Scored:</b> {html.escape(str(cdata['scored_answer']))}</p>"
                f"<pre class='response'>{html.escape(cdata['model_output'])}</pre>"
                "</div>"
            )

        cards.append(
            f"<article class='card' data-benchmark='{html.escape(bm, quote=True)}'>"
            f"<header><span class='bm'>{html.escape(bm)}</span> "
            f"<span class='idx'>#{idx}</span> "
            f"<span class='img-name'>{html.escape(row.get('image', ''))}</span></header>"
            "<div class='card-body'>"
            f"<div class='image-col'>{img_html}</div>"
            "<div class='meta-col'>"
            f"<p class='prompt'><b>Prompt:</b> {html.escape(prompt)}</p>"
            f"<p class='gt'><b>Ground truth:</b> {html.escape(str(row.get('ground_truth', '')))}</p>"
            f"<div class='panels'>{''.join(panels)}</div>"
            "</div></div></article>"
        )

    select_opts = [
        f"<option value='all' selected>All benchmarks ({len(rows)})</option>",
    ]
    for bm in benchmarks:
        select_opts.append(
            f"<option value='{html.escape(bm, quote=True)}'>"
            f"{html.escape(bm)} ({bm_counts[bm]})</option>"
        )

    stats_json = json.dumps(
        {bm: {cond: list(pair) for cond, pair in cond_stats.items()}
         for bm, cond_stats in stats_by_bm.items()}
    )

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{
  --ok: #1b7f3a;
  --bad: #b42318;
  --bg: #f7f7f8;
  --card: #fff;
  --border: #e4e4e7;
  --muted: #666;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; padding: 24px;
  font-family: system-ui, -apple-system, Segoe UI, sans-serif;
  background: var(--bg); color: #111;
}}
h1 {{ margin: 0 0 8px 0; font-size: 1.5rem; }}
.subtitle {{ color: var(--muted); margin-bottom: 16px; }}
.toolbar {{
  display: flex; flex-wrap: wrap; align-items: center; gap: 12px;
  margin-bottom: 16px; padding: 12px 14px;
  background: var(--card); border: 1px solid var(--border); border-radius: 10px;
}}
.toolbar label {{ font-weight: 600; font-size: 0.95rem; }}
.toolbar select {{
  min-width: 260px; padding: 8px 10px; font-size: 0.95rem;
  border: 1px solid var(--border); border-radius: 8px; background: #fff;
}}
.sample-count {{ color: var(--muted); font-size: 0.9rem; }}
.stats {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 24px; }}
.card.hidden {{ display: none; }}
.stat {{
  background: var(--card); border: 1px solid var(--border);
  border-radius: 8px; padding: 8px 12px; font-size: 0.95rem;
}}
.card {{
  background: var(--card); border: 1px solid var(--border);
  border-radius: 12px; margin-bottom: 20px; overflow: hidden;
}}
.card header {{
  padding: 10px 14px; border-bottom: 1px solid var(--border);
  background: #fafafa; font-size: 0.9rem;
}}
.bm {{ font-weight: 700; }}
.idx, .img-name {{ color: var(--muted); margin-left: 8px; }}
.card-body {{ display: grid; grid-template-columns: 280px 1fr; gap: 16px; padding: 14px; }}
.image-col img {{
  width: 100%; max-height: 280px; object-fit: contain;
  border: 1px solid var(--border); border-radius: 8px; background: #eee;
}}
.missing-img {{
  width: 100%; min-height: 180px; display: flex; align-items: center;
  justify-content: center; text-align: center; color: var(--muted);
  border: 1px dashed var(--border); border-radius: 8px; padding: 12px;
}}
.prompt, .gt {{ margin: 0 0 10px 0; line-height: 1.45; }}
.panels {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }}
.panel {{
  border: 1px solid var(--border); border-radius: 8px; padding: 10px;
  background: #fcfcfd; min-width: 0;
}}
.panel h3 {{ margin: 0 0 8px 0; font-size: 0.95rem; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}
.badge {{
  font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
  padding: 2px 6px; border-radius: 999px;
}}
.badge.ok {{ background: #dcf5e5; color: var(--ok); }}
.badge.bad {{ background: #fde8e6; color: var(--bad); }}
.scored {{ margin: 0 0 8px 0; font-size: 0.85rem; color: var(--muted); }}
.response {{
  margin: 0; white-space: pre-wrap; word-break: break-word;
  font-size: 0.82rem; line-height: 1.35; max-height: 220px; overflow: auto;
  background: #fff; border: 1px solid var(--border); border-radius: 6px; padding: 8px;
}}
.panel.missing {{ opacity: 0.7; }}
@media (max-width: 1100px) {{
  .card-body {{ grid-template-columns: 1fr; }}
  .panels {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<p class="subtitle">Eval dir: {html.escape(os.path.abspath(eval_dir))}</p>
<div class="toolbar">
  <label for="benchmark-select">Benchmark</label>
  <select id="benchmark-select" aria-label="Choose benchmark">
    {''.join(select_opts)}
  </select>
  <span class="sample-count" id="visible-count">{len(rows)} samples shown</span>
</div>
<div class="stats" id="stats-bar">{_format_stat_lines(stats_by_bm['all'])}</div>
<div id="cards">{''.join(cards)}</div>
<script>
const BENCHMARK_STATS = {stats_json};
const CONDITION_LABELS = {json.dumps(CONDITION_LABELS)};
const select = document.getElementById('benchmark-select');
const cards = document.querySelectorAll('.card');
const statsBar = document.getElementById('stats-bar');
const visibleCount = document.getElementById('visible-count');

function renderStats(bm) {{
  const stats = BENCHMARK_STATS[bm] || BENCHMARK_STATS.all;
  statsBar.innerHTML = Object.entries(CONDITION_LABELS).map(([cond, label]) => {{
    const pair = stats[cond] || [0, 0];
    const correct = pair[0], total = pair[1];
    const acc = total ? (100 * correct / total).toFixed(1) : '0.0';
    return `<span class="stat cond-${{cond}}"><b>${{label}}</b>: ${{correct}}/${{total}} (${{acc}}%)</span>`;
  }}).join('');
}}

function applyBenchmark(bm) {{
  let shown = 0;
  cards.forEach(card => {{
    const show = bm === 'all' || card.dataset.benchmark === bm;
    card.classList.toggle('hidden', !show);
    if (show) shown += 1;
  }});
  visibleCount.textContent = `${{shown}} sample${{shown === 1 ? '' : 's'}} shown`;
  renderStats(bm);
  if (location.hash !== `#${{bm}}`) {{
    history.replaceState(null, '', `#${{bm}}`);
  }}
}}

select.addEventListener('change', () => applyBenchmark(select.value));

const initial = location.hash ? location.hash.slice(1) : 'all';
if ([...select.options].some(o => o.value === initial)) {{
  select.value = initial;
}}
applyBenchmark(select.value);
</script>
</body>
</html>
"""

    with open(output, "w", encoding="utf-8") as f:
        f.write(doc)
    return output


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--eval_dir", required=True,
        help="Directory containing {frozen,lora_only,reinspection}_{benchmark}_samples.json",
    )
    ap.add_argument(
        "--data_root", default="/data/datasets",
        help="Dataset root for resolving benchmark image paths (default: /data/datasets)",
    )
    ap.add_argument(
        "--benchmarks", default=None,
        help="Comma-separated benchmark subset (default: all discovered)",
    )
    ap.add_argument("--limit", type=int, default=-1, help="Max samples after filtering (-1 = all)")
    ap.add_argument("--only-disagreements", action="store_true",
                    help="Keep samples where correctness differs across conditions")
    ap.add_argument("--only-reinspection-wins", action="store_true",
                    help="Keep samples where reinspection is correct and frozen is wrong")
    ap.add_argument("--output", default=None, help="Output HTML path (default: <eval_dir>/response_comparison.html)")
    ap.add_argument("--title", default="Eval response comparison", help="HTML page title")
    args = ap.parse_args()

    eval_dir = os.path.abspath(args.eval_dir)
    if not os.path.isdir(eval_dir):
        raise SystemExit(f"eval_dir not found: {eval_dir}")

    files_by_bm = discover_sample_files(eval_dir)
    if not files_by_bm:
        raise SystemExit(f"No *_samples.json files found in {eval_dir}")

    benchmarks = None
    if args.benchmarks:
        benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]

    rows = join_samples(files_by_bm, benchmarks=benchmarks)
    rows = _filter_rows(rows, args.only_disagreements, args.only_reinspection_wins)
    if args.limit > 0:
        rows = rows[: args.limit]

    output = args.output or os.path.join(eval_dir, "response_comparison.html")
    out_path = render_html(
        rows,
        eval_dir=eval_dir,
        data_root=args.data_root,
        output=output,
        title=args.title,
    )
    print(f"Wrote {len(rows)} samples → {out_path}", flush=True)


if __name__ == "__main__":
    main()
