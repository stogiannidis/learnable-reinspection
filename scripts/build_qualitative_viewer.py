"""Build a single-file, interactive HTML viewer that lets you scroll through
sample (image, question, GT) tuples and the three condition outputs side by side.

Selects a curated subset (capped per outcome bucket × benchmark) so the resulting
HTML stays under ~25 MB. Thumbnails are inlined as base64 JPEGs so the file is
fully portable — no server, no relative image paths.

Output: outputs/internvl3/grounding/qualitative_viewer.html
"""
import argparse
import base64
import html
import io
import json
import random
import re
from pathlib import Path

from PIL import Image, ImageOps

SAMPLES_DIR = Path("outputs/internvl3/grounding")
DATA_ROOT = Path("/data/datasets")
BENCHES = ["vsr", "gqa_spatial", "whatsup", "3dsrbench", "blink", "srbench"]
CONDS = ["frozen", "lora_only", "reinspection"]
COND_LABELS = {
    "frozen": "Frozen (base)",
    "lora_only": "LoRA-only",
    "reinspection": "Re-inspection (ours)",
}

# Per-(benchmark, bucket) cap on samples included.
# srbench gets the lion's share — it's the benchmark where re-inspection's
# behavior is most interesting (option-text matcher fix, MCQ letter wins).
DEFAULT_CAP = 6
PER_BENCH_CAP = {
    "srbench": 25,
}


# Patched matcher (mirrors src/evaluate.match_answer after the option-text fix).
# Inlined here so we don't depend on the rest of src/evaluate's heavy imports.
_MCQ_LETTERS = frozenset("ABCDEF")


def _norm(s):
    return " ".join((s or "").strip().lower().split())


def _extract_letter(text):
    t = (text or "").strip()
    if t.upper() in _MCQ_LETTERS:
        return t.upper()
    m = re.match(r"^\(?([A-Fa-f])\)?[\.\s:]*$", t)
    if m:
        return m.group(1).upper()
    m = re.match(r"^\(?([A-Fa-f])\)?[\.\)\s:]", t)
    if m:
        return m.group(1).upper()
    return None


def _gt_letter(gt_n):
    m = re.match(r"^\(?([a-f])\)?[\.\s)]", gt_n)
    return m.group(1).upper() if m else None


def is_correct(gen, gt):
    g, t = _norm(gen), _norm(gt)
    if len(t) == 1 and t.upper() in _MCQ_LETTERS:
        if g == t:
            return True
        l = _extract_letter(gen)
        return (l == t.upper()) if l is not None else False
    tl = _gt_letter(t)
    if tl is not None:
        gl = _extract_letter(gen)
        if gl is not None:
            return gl == tl
    return g == t or (bool(t) and t in g)


def load(cond, bench):
    p = SAMPLES_DIR / f"{cond}_{bench}_samples.json"
    return json.load(open(p)) if p.exists() else []


def encode_thumb(image_path: Path, max_side: int = 640) -> str:
    """Return data: URI for a downsized JPEG thumbnail, or empty string on failure."""
    try:
        with Image.open(image_path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((max_side, max_side), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=82, optimize=True)
            return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as exc:  # missing image, unreadable, etc.
        print(f"  WARN thumb {image_path}: {exc}")
        return ""


def bucket_of(r, l, f, old_r):
    """``r/l/f`` are the new patched-matcher booleans; ``old_r`` is whether
    re-inspection was already credited by the old matcher. ``matcher_save``
    surfaces the samples that flipped from wrong → right because of the
    option-text fix."""
    if r and not old_r:
        return "matcher_save"
    if r and not l and not f:
        return "ri_only_win"
    if not r and l and f:
        return "ri_only_loss"
    if r and not l and f:
        return "ri_lora_loss"   # re-inspection saved a case lora_only missed
    if r and l and f:
        return "all_correct"
    if not r and not l and not f:
        return "all_wrong"
    return "mixed"


def collect_samples():
    rng = random.Random(42)
    selected = []
    for bench in BENCHES:
        rs = load("reinspection", bench)
        ls = {s["idx"]: s for s in load("lora_only", bench)}
        fs = {s["idx"]: s for s in load("frozen", bench)}
        if not rs:
            continue

        cap = PER_BENCH_CAP.get(bench, DEFAULT_CAP)
        buckets = {}
        for s in rs:
            l = ls.get(s["idx"])
            f = fs.get(s["idx"])
            if not l or not f:
                continue
            # Recompute correctness with the patched matcher so bucket
            # assignment reflects the same scoring as the next eval run.
            r_ok = is_correct(s.get("model_output"), s.get("ground_truth"))
            l_ok = is_correct(l.get("model_output"), l.get("ground_truth"))
            f_ok = is_correct(f.get("model_output"), f.get("ground_truth"))
            b = bucket_of(r_ok, l_ok, f_ok, bool(s.get("correct")))
            buckets.setdefault(b, []).append((s, l, f, r_ok, l_ok, f_ok))

        for b in ("matcher_save", "ri_only_win", "ri_only_loss", "ri_lora_loss",
                  "all_correct", "all_wrong"):
            items = buckets.get(b, [])
            rng.shuffle(items)
            for s_r, s_l, s_f, r_ok, l_ok, f_ok in items[:cap]:
                selected.append(
                    {
                        "benchmark": bench,
                        "bucket": b,
                        "idx": int(s_r["idx"]),
                        "image_file": s_r.get("image", ""),
                        "question": s_r.get("question", ""),
                        "ground_truth": s_r.get("ground_truth", ""),
                        "outputs": {
                            "frozen":       (s_f.get("model_output") or "", bool(f_ok)),
                            "lora_only":    (s_l.get("model_output") or "", bool(l_ok)),
                            "reinspection": (s_r.get("model_output") or "", bool(r_ok)),
                        },
                    }
                )
    return selected


def build_html(samples, max_side):
    print(f"Encoding {len(samples)} thumbnails…")
    rows = []
    for i, s in enumerate(samples):
        img_path = DATA_ROOT / s["benchmark"] / "images" / s["image_file"]
        thumb = encode_thumb(img_path, max_side=max_side) if s["image_file"] else ""
        s["thumb"] = thumb
        rows.append(s)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(samples)}")
    payload = json.dumps(rows, ensure_ascii=False)

    css = """
    :root {
      --bg:#fafafa; --card:#fff; --border:#e5e5e5;
      --text:#222; --muted:#666;
      --ok:#0f7a35; --okbg:#e6f6ec;
      --bad:#a31616; --badbg:#fbe7e7;
      --accent:#1e4d9e;
    }
    * { box-sizing: border-box; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
           background: var(--bg); color: var(--text); margin: 0; padding: 0 24px 48px; }
    h1 { font-weight: 600; margin: 24px 0 4px; }
    .sub { color: var(--muted); margin-bottom: 20px; font-size: 14px; }
    .controls { position: sticky; top: 0; background: var(--bg); padding: 14px 0; z-index: 10;
                border-bottom: 1px solid var(--border); display: flex; gap: 16px; flex-wrap: wrap;
                align-items: center; }
    .controls label { font-size: 13px; color: var(--muted); margin-right: 4px; }
    .controls select, .controls input[type=text] {
      padding: 6px 10px; font-size: 13px; border: 1px solid var(--border);
      border-radius: 4px; background: white;
    }
    .count { margin-left: auto; color: var(--muted); font-size: 13px; }
    .sample { background: var(--card); border: 1px solid var(--border); border-radius: 6px;
              margin: 16px 0; padding: 16px; display: grid;
              grid-template-columns: 340px 1fr; gap: 20px; }
    .sample img { max-width: 100%; max-height: 360px; border-radius: 4px;
                  border: 1px solid var(--border); display: block; }
    .sample .meta { display: flex; gap: 8px; font-size: 12px; color: var(--muted);
                    margin-bottom: 6px; flex-wrap: wrap; }
    .tag { background: #eef; color: #335; padding: 1px 7px; border-radius: 3px; font-weight: 500; }
    .tag.bucket { background: #fdebd0; color: #7c4a00; }
    .tag.ri_only_win { background: #d6f5df; color: #0f5a25; }
    .tag.ri_only_loss { background: #fcd6d6; color: #8a1212; }
    .tag.all_correct { background: #e5ecf7; color: #234578; }
    .tag.all_wrong { background: #ececec; color: #444; }
    .tag.ri_lora_loss { background: #fff4cc; color: #6b4d00; }
    .tag.matcher_save { background: #f3d6ff; color: #5a1c7b; }
    .question { font-size: 14px; line-height: 1.5; margin-bottom: 10px;
                white-space: pre-wrap; word-wrap: break-word; }
    .gt { font-size: 13px; margin-bottom: 12px; }
    .gt strong { color: var(--muted); font-weight: 600; }
    .gt code { background: #f0f0f0; padding: 1px 6px; border-radius: 3px;
               font-family: ui-monospace, Menlo, monospace; }
    .answers { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; }
    .ans { padding: 10px 12px; border-radius: 4px; border: 1px solid var(--border);
           background: #fafafa; font-size: 13px; }
    .ans .lbl { font-size: 11px; color: var(--muted); margin-bottom: 4px;
                text-transform: uppercase; letter-spacing: 0.04em; }
    .ans .out { font-family: ui-monospace, Menlo, monospace; line-height: 1.4;
                white-space: pre-wrap; word-wrap: break-word; max-height: 12em; overflow-y: auto; }
    .ans.ok { border-left: 3px solid var(--ok); background: var(--okbg); }
    .ans.bad { border-left: 3px solid var(--bad); background: var(--badbg); }
    .ans.ours { box-shadow: 0 0 0 2px #ffd680 inset; }
    @media (max-width: 900px) {
      .sample { grid-template-columns: 1fr; }
      .answers { grid-template-columns: 1fr; }
    }
    """

    js = """
    const DATA = __DATA__;
    const BUCKET_LABELS = {
      'matcher_save': 'Matcher fix flipped re-inspection ✗ → ✓',
      'ri_only_win': 'Re-inspection only win',
      'ri_only_loss': 'Re-inspection only loss',
      'ri_lora_loss': 'Re-inspection right, LoRA wrong (Frozen right)',
      'all_correct': 'All three correct',
      'all_wrong': 'All three wrong',
    };
    const COND_LABELS = __COND_LABELS__;

    function el(tag, attrs={}, ...children) {
      const e = document.createElement(tag);
      for (const [k,v] of Object.entries(attrs)) {
        if (k === 'class') e.className = v;
        else if (k === 'html') e.innerHTML = v;
        else e.setAttribute(k, v);
      }
      for (const c of children) {
        if (c == null) continue;
        e.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
      }
      return e;
    }

    function renderSample(s) {
      const wrap = el('div', {class: 'sample'});

      const left = el('div');
      if (s.thumb) {
        left.appendChild(el('img', {src: s.thumb, alt: s.image_file}));
      } else {
        left.appendChild(el('div', {class: 'sub'}, '[image unavailable: ' + s.image_file + ']'));
      }

      const right = el('div');
      const meta = el('div', {class: 'meta'});
      meta.appendChild(el('span', {class: 'tag'}, s.benchmark));
      meta.appendChild(el('span', {class: 'tag bucket ' + s.bucket}, BUCKET_LABELS[s.bucket] || s.bucket));
      meta.appendChild(el('span', {}, 'idx=' + s.idx));
      meta.appendChild(el('span', {}, s.image_file));
      right.appendChild(meta);

      right.appendChild(el('div', {class: 'question'}, s.question));
      const gt = el('div', {class: 'gt'});
      gt.appendChild(el('strong', {}, 'Ground truth: '));
      gt.appendChild(el('code', {}, s.ground_truth));
      right.appendChild(gt);

      const ans = el('div', {class: 'answers'});
      for (const c of ['frozen','lora_only','reinspection']) {
        const [out, ok] = s.outputs[c];
        const cls = 'ans ' + (ok ? 'ok' : 'bad') + (c === 'reinspection' ? ' ours' : '');
        const box = el('div', {class: cls});
        box.appendChild(el('div', {class: 'lbl'}, COND_LABELS[c] + (ok ? ' ✓' : ' ✗')));
        box.appendChild(el('div', {class: 'out'}, out || '(empty)'));
        ans.appendChild(box);
      }
      right.appendChild(ans);

      wrap.appendChild(left);
      wrap.appendChild(right);
      return wrap;
    }

    function applyFilters() {
      const bench = document.getElementById('f-bench').value;
      const bucket = document.getElementById('f-bucket').value;
      const q = document.getElementById('f-search').value.trim().toLowerCase();
      const root = document.getElementById('rows');
      root.innerHTML = '';
      let n = 0;
      for (const s of DATA) {
        if (bench !== 'all' && s.benchmark !== bench) continue;
        if (bucket !== 'all' && s.bucket !== bucket) continue;
        if (q && !(s.question.toLowerCase().includes(q) || s.ground_truth.toLowerCase().includes(q))) continue;
        root.appendChild(renderSample(s));
        n++;
      }
      document.getElementById('count').textContent = n + ' sample' + (n === 1 ? '' : 's');
    }

    window.addEventListener('DOMContentLoaded', () => {
      const benches = Array.from(new Set(DATA.map(d => d.benchmark)));
      const bench_sel = document.getElementById('f-bench');
      for (const b of benches) bench_sel.appendChild(el('option', {value: b}, b));
      const buckets = Array.from(new Set(DATA.map(d => d.bucket)));
      const bucket_sel = document.getElementById('f-bucket');
      for (const b of buckets) bucket_sel.appendChild(el('option', {value: b}, BUCKET_LABELS[b] || b));
      bench_sel.addEventListener('change', applyFilters);
      bucket_sel.addEventListener('change', applyFilters);
      document.getElementById('f-search').addEventListener('input', applyFilters);
      applyFilters();
    });
    """
    js = js.replace("__DATA__", payload).replace(
        "__COND_LABELS__", json.dumps(COND_LABELS)
    )

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Re-inspection qualitative viewer</title>
  <style>{css}</style>
</head>
<body>
  <h1>Re-inspection qualitative viewer</h1>
  <div class="sub">
    InternVL3-8B · Stage-2 epoch_1 (matched-pair, fixed loader) · {len(samples)} curated samples.
    Yellow ring = "ours" (Re-inspection).
  </div>
  <div class="controls">
    <span><label for="f-bench">Benchmark</label>
      <select id="f-bench"><option value="all">all</option></select></span>
    <span><label for="f-bucket">Outcome</label>
      <select id="f-bucket"><option value="all">all</option></select></span>
    <span><label for="f-search">Search</label>
      <input id="f-search" type="text" placeholder="question or GT…" size="32"></span>
    <span class="count" id="count"></span>
  </div>
  <div id="rows"></div>
  <script>{js}</script>
</body>
</html>"""
    return page


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/internvl3/grounding/qualitative_viewer.html")
    ap.add_argument("--max-side", type=int, default=640, help="Thumbnail max dimension (px).")
    args = ap.parse_args()

    samples = collect_samples()
    print(f"Selected {len(samples)} samples across {len(BENCHES)} benchmarks × buckets.")
    html_text = build_html(samples, max_side=args.max_side)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_text, encoding="utf-8")
    size_mb = out.stat().st_size / 1024 / 1024
    print(f"Wrote {out}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
