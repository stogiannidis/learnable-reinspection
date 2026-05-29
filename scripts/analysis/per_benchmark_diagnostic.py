"""Per-benchmark diagnostic for the Re-Inspection Module.

Run-of-the-mill ``python scripts/analysis/per_benchmark_diagnostic.py``. Pure
Python + matplotlib; no torch / GPU.

Consumes:
  outputs/internvl3/grounding/{frozen,lora_only,reinspection}_{bm}_samples.json
  outputs/internvl3/grounding/eval_stage2_epoch1_compare_fixed.json  (entropies)
  /data/datasets/{bm}/test.{json,jsonl}                              (subtasks)

Writes:
  analysis/per_benchmark_summary.csv     - acc, Wilson 95% CI, n, Δ, paired bootstrap CI
  analysis/srbench_by_cluster.csv        - SRBench Δ by question-prefix cluster
  analysis/subtask_deltas.csv            - per-subtask Δ where category exists
  analysis/mcnemar.json                  - flip-in / flip-out per benchmark
  analysis/format_compliance.json        - lenient vs strict scoring on frozen wrong
  analysis/length_stats.json             - output-length distribution per condition
  analysis/figures/headroom.png          - Δ(RI-Frozen) vs Frozen accuracy
  analysis/figures/length_vs_delta.png   - prompt-length bucket vs per-bucket Δ
  analysis/figures/entropy_vs_delta.png  - mean attention entropy vs Δ
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = REPO_ROOT / "outputs" / "internvl3" / "grounding"
RESULTS_JSON = SAMPLES_DIR / "eval_stage2_epoch1_compare_fixed.json"
DATA_ROOT = Path("/data/datasets")
OUT_DIR = REPO_ROOT / "analysis"
FIG_DIR = OUT_DIR / "figures"

BENCHMARKS = [
    "vsr_zeroshot", "gqa_spatial", "whatsup", "3dsrbench", "blink",
    "srbench", "cv_bench", "mmvp", "realworldqa", "vstar_bench",
]
CONDITIONS = ["frozen", "lora_only", "reinspection"]

random.seed(0)


# ---------------------------------------------------------------------------
# Loading

def load_samples(condition: str, benchmark: str) -> List[dict]:
    path = SAMPLES_DIR / f"{condition}_{benchmark}_samples.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def load_source(benchmark: str) -> List[dict]:
    for ext in (".json", ".jsonl"):
        p = DATA_ROOT / benchmark / f"test{ext}"
        if p.exists():
            if ext == ".jsonl":
                return [json.loads(line) for line in open(p) if line.strip()]
            return json.load(open(p))
    return []


def load_entropies() -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    with open(RESULTS_JSON) as f:
        for r in json.load(f):
            if r["condition"] == "reinspection":
                out[r["benchmark"]] = r.get("mean_attention_entropy")
    return out


# ---------------------------------------------------------------------------
# Stats helpers

def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z2 / (4 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def paired_bootstrap_delta(a: List[int], b: List[int], n_iter: int = 2000) -> Tuple[float, float]:
    """95% CI for mean(b) - mean(a) on paired 0/1 vectors."""
    n = len(a)
    if n == 0:
        return (0.0, 0.0)
    rng = random.Random(0)
    diffs: List[float] = []
    for _ in range(n_iter):
        idx = [rng.randint(0, n - 1) for _ in range(n)]
        s = 0
        for i in idx:
            s += b[i] - a[i]
        diffs.append(s / n)
    diffs.sort()
    return (diffs[int(0.025 * n_iter)], diffs[int(0.975 * n_iter)])


def mcnemar_b_c(a: List[int], b: List[int]) -> Tuple[int, int]:
    """Returns (flip_in = wrong→right, flip_out = right→wrong)."""
    flip_in = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    flip_out = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    return flip_in, flip_out


# ---------------------------------------------------------------------------
# Lenient scoring — relaxes the canonical matcher's "letter must be at start" rule.

_MCQ = frozenset("ABCDEF")

# Patterns that bind a letter to an answer-assertion phrase. Crucially we do NOT
# count bare letter occurrences (the article "a", the pronoun "I") — that
# overcounts catastrophically in verbose chain-of-thought outputs.
_ANSWER_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\b(?:the\s+)?(?:correct\s+)?(?:final\s+)?answer\s*(?:is|:)\s*\**\(?([A-Fa-f])\)?\**\b",
        r"\b(?:the\s+)?correct\s+option\s*(?:is|:)\s*\**\(?([A-Fa-f])\)?\**\b",
        r"\b(?:the\s+)?(?:best\s+)?(?:chosen\s+)?option\s*(?:is\s+)?\**\(?([A-Fa-f])\)?\**[\.\s:,)]",
        r"\bchoice\s*(?:is|:)\s*\**\(?([A-Fa-f])\)?\**\b",
        r"\b(?:i\s+(?:choose|select|pick))\s+\**\(?([A-Fa-f])\)?\**\b",
        r"\bso(?:,)?\s+(?:the\s+answer\s+is\s+|it(?:'s|\s+is)\s+)\**\(?([A-Fa-f])\)?\**\b",
        r"\b(?:therefore|thus|hence)(?:,)?\s+\(?([A-Fa-f])\)?[\.\s:)]",
        r"\b(?:option|answer)\s+\**([A-Fa-f])\**\s+is\s+(?:the\s+)?correct\b",
        r"\*\*([A-Fa-f])\*\*",
        r"^\**\(?([A-Fa-f])\)?\**[\.\s:)]",
    ]
]


_STRICT_LETTER_START = re.compile(r"^\(?([A-Fa-f])\)?[\.\s:)]?\s*$")
_STRICT_LETTER_PREFIX = re.compile(r"^\(?([A-Fa-f])\)?[\.\)\s:]")


def _strict_letter(text: str) -> Optional[str]:
    t = (text or "").strip()
    if len(t) == 1 and t.upper() in _MCQ:
        return t.upper()
    m = _STRICT_LETTER_START.match(t)
    if m:
        return m.group(1).upper()
    m = _STRICT_LETTER_PREFIX.match(t)
    if m:
        return m.group(1).upper()
    return None


def lenient_correct(model_output: str, gt: str) -> bool:
    """Relaxed MCQ scorer = strict-letter-extraction OR answer-assertion phrase
    match (e.g., "the correct option is B", "answer: C", "**D**"). Does NOT
    credit free-floating letters in chain-of-thought (articles "a", pronouns
    "I", or letters the model later contradicts).
    """
    gen = (model_output or "").strip()
    g = (gt or "").strip()
    if not gen or not g:
        return False
    gt_letter: Optional[str] = None
    if len(g) == 1 and g.upper() in _MCQ:
        gt_letter = g.upper()
    else:
        m = re.match(r"^\(?([a-f])\)?[\.\s)]", g.lower())
        if m:
            gt_letter = m.group(1).upper()
    if gt_letter is not None:
        sl = _strict_letter(gen)
        if sl == gt_letter:
            return True
        for pat in _ANSWER_PATTERNS:
            for m in pat.finditer(gen):
                if m.group(1).upper() == gt_letter:
                    return True
        return False
    g_norm = " ".join(g.lower().split())
    return g_norm in " ".join(gen.lower().split())


# ---------------------------------------------------------------------------
# SRBench clustering by question prefix

def srbench_cluster(question: str) -> str:
    q = question.lower()
    if q.startswith("the image depicts a 3d polycube"):
        return "polycube"
    if q.startswith("the image depicts a piece of paper folded"):
        return "paper_fold"
    if q.startswith("the figure represents a maze"):
        return "maze"
    if "facing 'left'" in q or "facing 'right'" in q:
        return "facing_direction"
    if "in which hand" in q or "which hand is" in q:
        return "which_hand"
    return "other"


# ---------------------------------------------------------------------------
# Per-benchmark cell

def per_benchmark_summary() -> List[dict]:
    rows: List[dict] = []
    cache: Dict[Tuple[str, str], List[dict]] = {}
    for bm in BENCHMARKS:
        for cond in CONDITIONS:
            samples = load_samples(cond, bm)
            cache[(cond, bm)] = samples
            n = len(samples)
            k = sum(1 for s in samples if s.get("correct"))
            acc = k / n if n else 0.0
            lo, hi = wilson_ci(k, n)
            rows.append({
                "benchmark": bm, "condition": cond, "n": n,
                "correct": k, "accuracy": acc,
                "wilson_lo": lo, "wilson_hi": hi,
            })
        frozen = cache[("frozen", bm)]
        for cond in ("lora_only", "reinspection"):
            other = cache[(cond, bm)]
            if not frozen or not other or len(frozen) != len(other):
                continue
            a = [int(bool(s.get("correct"))) for s in frozen]
            b = [int(bool(s.get("correct"))) for s in other]
            lo, hi = paired_bootstrap_delta(a, b)
            for r in rows:
                if r["benchmark"] == bm and r["condition"] == cond:
                    r["delta_vs_frozen"] = sum(b) / len(b) - sum(a) / len(a)
                    r["delta_ci_lo"] = lo
                    r["delta_ci_hi"] = hi
                    break
    return rows


# ---------------------------------------------------------------------------
# SRBench by cluster

def srbench_by_cluster() -> List[dict]:
    src = load_source("srbench")
    if not src:
        return []
    cluster_by_idx = {i: srbench_cluster(s.get("question", "")) for i, s in enumerate(src)}
    per_cluster: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    for cond in CONDITIONS:
        for s in load_samples(cond, "srbench"):
            c = cluster_by_idx.get(s["idx"], "other")
            per_cluster[c][cond].append(int(bool(s.get("correct"))))
    rows: List[dict] = []
    for cluster in ("polycube", "paper_fold", "maze", "facing_direction", "which_hand", "other"):
        if cluster not in per_cluster:
            continue
        out = {"cluster": cluster, "n": len(per_cluster[cluster].get("frozen", []))}
        for cond in CONDITIONS:
            v = per_cluster[cluster][cond]
            out[f"acc_{cond}"] = (sum(v) / len(v)) if v else 0.0
        frozen = per_cluster[cluster]["frozen"]
        for cond in ("lora_only", "reinspection"):
            other = per_cluster[cluster].get(cond, [])
            if frozen and other and len(frozen) == len(other):
                lo, hi = paired_bootstrap_delta(frozen, other)
                out[f"delta_{cond}"] = sum(other) / len(other) - sum(frozen) / len(frozen)
                out[f"delta_{cond}_ci_lo"] = lo
                out[f"delta_{cond}_ci_hi"] = hi
        rows.append(out)
    return rows


# ---------------------------------------------------------------------------
# Subtask breakdowns

CATEGORY_FIELD = "category"
HAS_CATEGORY = ["3dsrbench", "blink", "cv_bench", "vstar_bench"]


def subtask_deltas() -> List[dict]:
    rows: List[dict] = []
    for bm in HAS_CATEGORY:
        src = load_source(bm)
        if not src:
            continue
        cat_by_idx = {i: s.get(CATEGORY_FIELD, "") for i, s in enumerate(src)}
        per_cat: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        for cond in CONDITIONS:
            for s in load_samples(cond, bm):
                c = cat_by_idx.get(s["idx"], "")
                per_cat[c][cond].append(int(bool(s.get("correct"))))
        for cat in sorted(per_cat):
            out = {"benchmark": bm, "category": cat, "n": len(per_cat[cat].get("frozen", []))}
            for cond in CONDITIONS:
                v = per_cat[cat][cond]
                out[f"acc_{cond}"] = (sum(v) / len(v)) if v else 0.0
            frozen = per_cat[cat]["frozen"]
            for cond in ("lora_only", "reinspection"):
                other = per_cat[cat].get(cond, [])
                if frozen and other and len(frozen) == len(other):
                    out[f"delta_{cond}"] = sum(other) / len(other) - sum(frozen) / len(frozen)
            rows.append(out)
    return rows


# ---------------------------------------------------------------------------
# McNemar flip table per benchmark

def mcnemar_table() -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for bm in BENCHMARKS:
        frozen = load_samples("frozen", bm)
        reinsp = load_samples("reinspection", bm)
        lora = load_samples("lora_only", bm)
        if not frozen or not reinsp:
            continue
        a = [int(bool(s.get("correct"))) for s in frozen]
        b = [int(bool(s.get("correct"))) for s in reinsp]
        c = [int(bool(s.get("correct"))) for s in lora] if lora else None
        flip_in, flip_out = mcnemar_b_c(a, b)
        both_right = sum(1 for x, y in zip(a, b) if x and y)
        both_wrong = sum(1 for x, y in zip(a, b) if not x and not y)
        entry = {
            "n": len(a),
            "frozen_vs_reinspection": {
                "both_right": both_right,
                "both_wrong": both_wrong,
                "flip_in_wrong_to_right": flip_in,
                "flip_out_right_to_wrong": flip_out,
                "net": flip_in - flip_out,
            },
        }
        if c is not None and len(c) == len(a):
            fin, fout = mcnemar_b_c(c, b)
            entry["lora_vs_reinspection"] = {
                "flip_in_wrong_to_right": fin,
                "flip_out_right_to_wrong": fout,
                "net": fin - fout,
            }
        out[bm] = entry
    return out


# ---------------------------------------------------------------------------
# Format-compliance audit (Phase B)

def format_compliance() -> Dict[str, dict]:
    """For each benchmark, count Frozen-wrong items whose model_output contains the
    GT letter as a standalone token (i.e., relaxed match succeeds). Then check
    how many of those are also Reinspection-correct under strict scoring.
    """
    out: Dict[str, dict] = {}
    for bm in BENCHMARKS:
        frozen = load_samples("frozen", bm)
        reinsp = load_samples("reinspection", bm)
        if not frozen or not reinsp:
            continue
        n = len(frozen)
        frozen_wrong_strict = sum(1 for s in frozen if not s.get("correct"))
        # Lenient on all frozen samples
        frozen_lenient_correct = sum(
            1 for s in frozen if lenient_correct(s.get("model_output", ""), s.get("ground_truth", ""))
        )
        # Of frozen-wrong-strict, how many are lenient-correct? (format-failure rescued)
        format_failures = 0
        rescued_by_reinsp = 0
        for f, r in zip(frozen, reinsp):
            if not f.get("correct") and lenient_correct(f.get("model_output", ""), f.get("ground_truth", "")):
                format_failures += 1
                if r.get("correct"):
                    rescued_by_reinsp += 1
        # ReInsp lenient accuracy too (sanity)
        reinsp_lenient_correct = sum(
            1 for s in reinsp if lenient_correct(s.get("model_output", ""), s.get("ground_truth", ""))
        )
        out[bm] = {
            "n": n,
            "frozen_acc_strict": sum(1 for s in frozen if s.get("correct")) / n,
            "frozen_acc_lenient": frozen_lenient_correct / n,
            "reinsp_acc_strict": sum(1 for s in reinsp if s.get("correct")) / n,
            "reinsp_acc_lenient": reinsp_lenient_correct / n,
            "format_failure_count": format_failures,
            "format_failure_rate": format_failures / n,
            "format_failures_also_correct_by_reinsp": rescued_by_reinsp,
            "frozen_lenient_minus_strict": frozen_lenient_correct / n - sum(1 for s in frozen if s.get("correct")) / n,
        }
    return out


# ---------------------------------------------------------------------------
# Output length distribution

def length_stats() -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for bm in BENCHMARKS:
        per_cond: Dict[str, dict] = {}
        for cond in CONDITIONS:
            samples = load_samples(cond, bm)
            if not samples:
                continue
            lens = [len(s.get("model_output", "")) for s in samples]
            lens.sort()
            n = len(lens)
            per_cond[cond] = {
                "n": n,
                "mean": sum(lens) / n,
                "median": lens[n // 2],
                "p25": lens[n // 4],
                "p75": lens[(3 * n) // 4],
            }
        out[bm] = per_cond
    return out


# ---------------------------------------------------------------------------
# Length-bucket Δ on SRBench specifically (where prompt length varies most)

def srbench_length_bucket_delta() -> List[dict]:
    src = load_source("srbench")
    if not src:
        return []
    qlen_by_idx = {i: len(s.get("question", "")) for i, s in enumerate(src)}
    frozen = load_samples("frozen", "srbench")
    reinsp = load_samples("reinspection", "srbench")
    pairs: List[Tuple[int, int, int]] = []
    for f, r in zip(frozen, reinsp):
        ql = qlen_by_idx.get(f["idx"], 0)
        pairs.append((ql, int(bool(f.get("correct"))), int(bool(r.get("correct")))))
    pairs.sort(key=lambda x: x[0])
    n = len(pairs)
    rows: List[dict] = []
    nbuckets = 5
    for i in range(nbuckets):
        lo = (i * n) // nbuckets
        hi = ((i + 1) * n) // nbuckets
        chunk = pairs[lo:hi]
        if not chunk:
            continue
        a = [c[1] for c in chunk]
        b = [c[2] for c in chunk]
        rows.append({
            "bucket": i + 1,
            "n": len(chunk),
            "qlen_min": chunk[0][0],
            "qlen_max": chunk[-1][0],
            "frozen_acc": sum(a) / len(a),
            "reinsp_acc": sum(b) / len(b),
            "delta": sum(b) / len(b) - sum(a) / len(a),
        })
    return rows


# ---------------------------------------------------------------------------
# Plots

def plot_headroom(summary_rows: List[dict]) -> None:
    pts: List[Tuple[str, float, float]] = []
    for bm in BENCHMARKS:
        f = next((r for r in summary_rows if r["benchmark"] == bm and r["condition"] == "frozen"), None)
        ri = next((r for r in summary_rows if r["benchmark"] == bm and r["condition"] == "reinspection"), None)
        if not f or not ri or f["n"] == 0:
            continue
        pts.append((bm, f["accuracy"], ri.get("delta_vs_frozen", 0.0)))
    if not pts:
        return
    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(0, color="grey", linewidth=0.6, alpha=0.5)
    ax.scatter(xs, ys, s=50, color="#cc3333", zorder=3)
    for bm, x, y in pts:
        ax.annotate(bm, (x, y), xytext=(6, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Frozen accuracy (baseline headroom)")
    ax.set_ylabel("Δ accuracy (Re-Inspection − Frozen)")
    ax.set_title("Per-benchmark headroom vs. Re-Inspection gain")
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "headroom.png", dpi=140)
    plt.close(fig)


def plot_length_vs_delta(length_rows: List[dict]) -> None:
    if not length_rows:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = [r["bucket"] for r in length_rows]
    ys = [r["delta"] for r in length_rows]
    ax.axhline(0, color="grey", linewidth=0.6, alpha=0.5)
    ax.bar(xs, ys, color="#3366cc")
    for r in length_rows:
        ax.annotate(
            f"n={r['n']}\nqlen {r['qlen_min']}–{r['qlen_max']}",
            (r["bucket"], r["delta"]),
            xytext=(0, 4 if r["delta"] >= 0 else -16),
            textcoords="offset points", ha="center", fontsize=7,
        )
    ax.set_xlabel("SRBench question-length quintile (1 = shortest)")
    ax.set_ylabel("Δ accuracy (Re-Inspection − Frozen)")
    ax.set_title("SRBench: Re-Inspection gain by question-length bucket")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "length_vs_delta.png", dpi=140)
    plt.close(fig)


def plot_gain_decomposition(fmt: Dict[str, dict]) -> None:
    """Stacked-bar: per-benchmark strict Δ split into format-rescue + lenient Δ."""
    order = ["srbench", "cv_bench", "whatsup", "vstar_bench", "vsr_zeroshot",
             "gqa_spatial", "blink", "realworldqa", "mmvp", "3dsrbench"]
    rows = [(bm, fmt[bm]) for bm in order if bm in fmt]
    if not rows:
        return
    labels = [bm for bm, _ in rows]
    strict = [r["reinsp_acc_strict"] - r["frozen_acc_strict"] for _, r in rows]
    lenient = [r["reinsp_acc_lenient"] - r["frozen_acc_lenient"] for _, r in rows]
    rescue = [s - l for s, l in zip(strict, lenient)]
    xs = range(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(0, color="grey", linewidth=0.6, alpha=0.6)
    bar_real = ax.bar(xs, lenient, color="#2a7a2a", label="real gain (Δ under lenient scoring)")
    bar_fmt = ax.bar(xs, rescue, bottom=lenient, color="#cc8a33", label="format-rescue (strict−lenient Δ)")
    for i, (s, l, r) in enumerate(zip(strict, lenient, rescue)):
        ax.annotate(f"{s:+.2f}", (i, s if s >= 0 else 0), xytext=(0, 3 if s >= 0 else -12),
                    textcoords="offset points", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Δ accuracy (Re-Inspection − Frozen)")
    ax.set_title("How much of the Re-Inspection gain survives lenient scoring?")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "gain_decomposition.png", dpi=140)
    plt.close(fig)


def plot_entropy_vs_delta(summary_rows: List[dict]) -> None:
    ents = load_entropies()
    pts: List[Tuple[str, float, float]] = []
    for bm in BENCHMARKS:
        ri = next((r for r in summary_rows if r["benchmark"] == bm and r["condition"] == "reinspection"), None)
        e = ents.get(bm)
        if ri is None or e is None or ri["n"] == 0:
            continue
        pts.append((bm, e, ri.get("delta_vs_frozen", 0.0)))
    if not pts:
        return
    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(0, color="grey", linewidth=0.6, alpha=0.5)
    ax.scatter(xs, ys, s=50, color="#339966", zorder=3)
    for bm, x, y in pts:
        ax.annotate(bm, (x, y), xytext=(6, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Mean Re-Inspection attention entropy (over visual tokens)")
    ax.set_ylabel("Δ accuracy (Re-Inspection − Frozen)")
    ax.set_title("Module attention concentration vs. Re-Inspection gain")
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "entropy_vs_delta.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# I/O helpers

def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fields: List[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_json(path: Path, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


# ---------------------------------------------------------------------------
# Main

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Computing per-benchmark summary …")
    summary = per_benchmark_summary()
    write_csv(OUT_DIR / "per_benchmark_summary.csv", summary)

    print("Computing SRBench by cluster …")
    srb_clusters = srbench_by_cluster()
    write_csv(OUT_DIR / "srbench_by_cluster.csv", srb_clusters)

    print("Computing subtask deltas …")
    subt = subtask_deltas()
    write_csv(OUT_DIR / "subtask_deltas.csv", subt)

    print("Computing McNemar flips …")
    mcn = mcnemar_table()
    write_json(OUT_DIR / "mcnemar.json", mcn)

    print("Auditing format compliance …")
    fmt = format_compliance()
    write_json(OUT_DIR / "format_compliance.json", fmt)

    print("Computing length stats …")
    lens = length_stats()
    write_json(OUT_DIR / "length_stats.json", lens)

    print("Computing SRBench length-bucket Δ …")
    srb_len = srbench_length_bucket_delta()
    write_csv(OUT_DIR / "srbench_length_buckets.csv", srb_len)

    print("Plotting headroom …")
    plot_headroom(summary)
    print("Plotting length-vs-Δ …")
    plot_length_vs_delta(srb_len)
    print("Plotting entropy-vs-Δ …")
    plot_entropy_vs_delta(summary)
    print("Plotting gain decomposition …")
    plot_gain_decomposition(fmt)

    # Pretty topline to stdout
    print("\n=== Topline (rescored, canonical matcher) ===")
    print(f"{'Benchmark':18s}  {'n':>5s}  {'Frozen':>8s}  {'LoRA':>8s}  {'ReInsp':>8s}  {'Δ(RI-F)':>10s}  {'Δ95% CI':>20s}")
    for bm in BENCHMARKS:
        rows = {r["condition"]: r for r in summary if r["benchmark"] == bm}
        f = rows.get("frozen")
        l = rows.get("lora_only")
        r = rows.get("reinspection")
        if not f or f["n"] == 0:
            print(f"{bm:18s}  n=0 (no samples)")
            continue
        d = r.get("delta_vs_frozen", 0.0)
        lo = r.get("delta_ci_lo", 0.0)
        hi = r.get("delta_ci_hi", 0.0)
        print(
            f"{bm:18s}  {f['n']:>5d}  {f['accuracy']:>8.4f}  "
            f"{l['accuracy']:>8.4f}  {r['accuracy']:>8.4f}  "
            f"{d:>+10.4f}  [{lo:+.4f}, {hi:+.4f}]"
        )


if __name__ == "__main__":
    main()
