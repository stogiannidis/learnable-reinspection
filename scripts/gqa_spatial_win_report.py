#!/usr/bin/env python3
"""Build an HTML report for GQA-Spatial samples where reinspection beats frozen."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Dict, List


def classify_win(question: str, ground_truth: str, frozen_output: str) -> str:
    gt = ground_truth.strip().lower()
    frozen = frozen_output.strip().lower()
    frozen_len = len(frozen.split())
    q = question.strip().lower()

    if gt in {"yes", "no"} and frozen in {"true", "false"}:
        return "Boolean canonicalization"
    if gt in {"yes", "no"}:
        return "Boolean reasoning"
    if frozen_len > 5:
        if q.startswith(("what", "who", "which", "how", "where")):
            return "Entity or attribute grounding"
        return "Verbose open-ended miss"
    return "Short-answer mismatch"


def load_wins(reinspection_file: Path, frozen_file: Path, image_root: Path) -> List[Dict]:
    reinspection_samples = json.loads(reinspection_file.read_text(encoding="utf-8"))
    frozen_samples = json.loads(frozen_file.read_text(encoding="utf-8"))

    wins: List[Dict] = []
    for ri, frozen in zip(reinspection_samples, frozen_samples):
        key_ri = (ri["idx"], ri["image"], ri["question"])
        key_frozen = (frozen["idx"], frozen["image"], frozen["question"])
        if key_ri != key_frozen:
            raise ValueError(f"Mismatched sample ordering: {key_ri!r} != {key_frozen!r}")
        if ri["correct"] and not frozen["correct"]:
            image_path = image_root / ri["image"]
            wins.append(
                {
                    "idx": ri["idx"],
                    "image": ri["image"],
                    "image_uri": image_path.resolve().as_uri(),
                    "question": ri["question"],
                    "ground_truth": ri["ground_truth"],
                    "reinspection_output": ri["model_output"],
                    "frozen_output": frozen["model_output"],
                    "category": classify_win(ri["question"], ri["ground_truth"], frozen["model_output"]),
                }
            )
    return wins


def category_summary(wins: List[Dict]) -> List[Dict]:
    counts: Dict[str, int] = {}
    for row in wins:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    return [
        {"category": category, "count": count}
        for category, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def build_html(wins: List[Dict], summary: List[Dict], title: str) -> str:
    summary_rows = "\n".join(
        f"<li><strong>{html.escape(item['category'])}</strong>: {item['count']}</li>"
        for item in summary
    )
    cards = []
    for row in wins:
        cards.append(
            f"""
            <article class="card" data-category="{html.escape(row['category'])}">
              <div class="media">
                <img src="{html.escape(row['image_uri'])}" alt="{html.escape(row['image'])}" loading="lazy" />
              </div>
              <div class="content">
                <div class="meta">
                  <span class="pill">{html.escape(row['category'])}</span>
                  <span class="sample">idx {row['idx']} | {html.escape(row['image'])}</span>
                </div>
                <p><strong>Question:</strong> {html.escape(row['question'])}</p>
                <p><strong>Ground truth:</strong> <code>{html.escape(row['ground_truth'])}</code></p>
                <div class="responses">
                  <section class="resp better">
                    <h3>Reinspection</h3>
                    <p><code>{html.escape(row['reinspection_output'])}</code></p>
                  </section>
                  <section class="resp worse">
                    <h3>Frozen</h3>
                    <p><code>{html.escape(row['frozen_output'])}</code></p>
                  </section>
                </div>
              </div>
            </article>
            """
        )

    category_options = "\n".join(
        f'<option value="{html.escape(item["category"])}">{html.escape(item["category"])} ({item["count"]})</option>'
        for item in summary
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f4efe7;
      --panel: #fffaf2;
      --ink: #1d1b18;
      --muted: #6e665c;
      --border: #d5c5b0;
      --accent: #a6442f;
      --good: #1d6b4f;
      --bad: #8c2f39;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", serif;
      background:
        radial-gradient(circle at top left, rgba(166, 68, 47, 0.10), transparent 30%),
        linear-gradient(180deg, #f8f2e8 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    header {{
      padding: 32px 24px 20px;
      border-bottom: 1px solid var(--border);
      background: rgba(255, 250, 242, 0.85);
      position: sticky;
      top: 0;
      backdrop-filter: blur(8px);
      z-index: 10;
    }}
    h1 {{ margin: 0 0 10px; font-size: 2rem; }}
    p, li {{ line-height: 1.45; }}
    .lede {{ max-width: 1000px; color: var(--muted); margin: 0 0 14px; }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      margin-top: 14px;
    }}
    select, input {{
      font: inherit;
      padding: 8px 10px;
      border: 1px solid var(--border);
      background: var(--panel);
      color: var(--ink);
    }}
    main {{
      padding: 24px;
      max-width: 1500px;
      margin: 0 auto;
    }}
    .summary {{
      margin: 0 0 24px;
      padding: 18px 20px;
      background: var(--panel);
      border: 1px solid var(--border);
    }}
    .cards {{
      display: grid;
      gap: 18px;
    }}
    .card {{
      display: grid;
      grid-template-columns: minmax(280px, 420px) 1fr;
      gap: 18px;
      padding: 18px;
      background: rgba(255, 250, 242, 0.96);
      border: 1px solid var(--border);
      box-shadow: 0 10px 30px rgba(29, 27, 24, 0.08);
    }}
    .media img {{
      width: 100%;
      max-height: 360px;
      object-fit: contain;
      background: #ece3d6;
      border: 1px solid var(--border);
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      margin-bottom: 10px;
    }}
    .pill {{
      display: inline-block;
      padding: 4px 8px;
      background: #f2ddcf;
      color: var(--accent);
      border: 1px solid #e0b79d;
      font-size: 0.9rem;
    }}
    .sample {{ color: var(--muted); font-size: 0.95rem; }}
    .responses {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      margin-top: 12px;
    }}
    .resp {{
      padding: 14px;
      border: 1px solid var(--border);
      background: #fff;
    }}
    .resp h3 {{ margin: 0 0 8px; font-size: 1rem; }}
    .better h3 {{ color: var(--good); }}
    .worse h3 {{ color: var(--bad); }}
    code {{
      font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .hidden {{ display: none; }}
    @media (max-width: 960px) {{
      .card {{ grid-template-columns: 1fr; }}
      .responses {{ grid-template-columns: 1fr; }}
      header {{ position: static; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <p class="lede">
      Samples where the reinspection model is marked correct and the frozen baseline is marked wrong on GQA-Spatial.
      The report uses the existing evaluation artifacts and links directly to local images under the dataset root.
    </p>
    <div class="toolbar">
      <label>Category
        <select id="category">
          <option value="all">All categories ({len(wins)})</option>
          {category_options}
        </select>
      </label>
      <label>Search
        <input id="search" type="search" placeholder="question, answer, image..." />
      </label>
      <span id="count">{len(wins)} samples</span>
    </div>
  </header>
  <main>
    <section class="summary">
      <p><strong>Total wins:</strong> {len(wins)}</p>
      <ul>
        {summary_rows}
      </ul>
    </section>
    <section class="cards" id="cards">
      {''.join(cards)}
    </section>
  </main>
  <script>
    const cards = Array.from(document.querySelectorAll('.card'));
    const category = document.getElementById('category');
    const search = document.getElementById('search');
    const count = document.getElementById('count');

    function applyFilters() {{
      const cat = category.value;
      const needle = search.value.trim().toLowerCase();
      let visible = 0;
      for (const card of cards) {{
        const catOk = cat === 'all' || card.dataset.category === cat;
        const textOk = !needle || card.textContent.toLowerCase().includes(needle);
        const show = catOk && textOk;
        card.classList.toggle('hidden', !show);
        if (show) visible += 1;
      }}
      count.textContent = `${{visible}} samples`;
    }}

    category.addEventListener('change', applyFilters);
    search.addEventListener('input', applyFilters);
    applyFilters();
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reinspection-file",
        type=Path,
        default=Path("outputs/internvl3/reinspection_gqa_spatial_samples.json"),
    )
    parser.add_argument(
        "--frozen-file",
        type=Path,
        default=Path("outputs/internvl3/frozen_gqa_spatial_samples.json"),
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("/data/datasets/gqa_spatial/images"),
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path("outputs/internvl3/gqa_spatial_reinspection_beats_frozen_grounding.html"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/internvl3/gqa_spatial_reinspection_beats_frozen_grounding.json"),
    )
    parser.add_argument(
        "--exclude-categories",
        nargs="*",
        default=[],
        help="Categories to exclude from the output report.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="GQA-Spatial Wins: Reinspection vs Frozen",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wins = load_wins(args.reinspection_file, args.frozen_file, args.image_root)
    if args.exclude_categories:
        excluded = set(args.exclude_categories)
        wins = [row for row in wins if row["category"] not in excluded]
    summary = category_summary(wins)

    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(
        build_html(wins, summary, args.title),
        encoding="utf-8",
    )
    args.output_json.write_text(json.dumps(wins, indent=2), encoding="utf-8")

    print(f"Wrote {len(wins)} winning samples to {args.output_json}")
    print(f"Wrote HTML report to {args.output_html}")


if __name__ == "__main__":
    main()
