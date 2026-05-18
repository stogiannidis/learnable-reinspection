"""Log current eval baseline (frozen/lora_only/reinspection across 6 benchmarks)
to W&B as a single anchor run. Future runs can be diffed against this.

Logs both the default matcher (current `match_answer` in src/evaluate.py) and a
fair matcher (synonyms + generous MCQ-letter extraction) so the matcher artifact
can be tracked separately from the model artifact.

Usage:
    WANDB_MODE=online python scripts/wandb_log_baseline.py \
        --run-name stage2-epoch1-baseline-fixed-loader \
        --samples-dir outputs/internvl3/grounding
"""
import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from collections import Counter

import wandb


BENCHES = ["vsr", "gqa_spatial", "whatsup", "3dsrbench", "blink", "srbench"]
CONDS = ["frozen", "lora_only", "reinspection"]

SYN_GROUPS = [
    {"true", "yes", "correct", "right", "accurate", "valid", "affirmative"},
    {"false", "no", "incorrect", "wrong", "inaccurate", "invalid", "negative"},
    {"left", "to the left", "to the left of", "on the left", "on the left of",
     "left of", "left side"},
    {"right", "to the right", "to the right of", "on the right",
     "on the right of", "right of", "right side"},
    {"above", "on top of", "on top", "over", "atop", "upon"},
    {"below", "underneath", "under", "beneath"},
    {"behind", "in back of", "at the back of", "back of", "back"},
    {"in front of", "in front", "front of", "front", "ahead of"},
    {"inside", "in", "within", "into"},
    {"outside", "out of", "out"},
    {"next to", "beside", "adjacent to", "alongside", "near"},
]

MCQ_PATTERNS = [
    r"\boption\s+([A-Da-d])\b",
    r"\banswer\s+is\s+\(?([A-Da-d])\)?",
    r"\bcorrect\s+(?:option|answer|choice)\s+is\s+\(?([A-Da-d])\)?",
    r"^\(?([A-Da-d])\)?[\.\)\s:]",
    r"^\(?([A-Da-d])\)?[\.\s:]*$",
    r"\bchoose\s+\(?([A-Da-d])\)?",
    r"\(([A-Da-d])\)",
]


def _norm(s):
    return " ".join((s or "").strip().lower().split())


def _syn(s):
    n = _norm(s)
    for g in SYN_GROUPS:
        if n in g:
            return g
    return {n}


def _extract_mcq(s):
    for p in MCQ_PATTERNS:
        m = re.search(p, s or "", flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return None


def _is_mcq(gt):
    n = _norm(gt)
    return len(n) == 1 and n.upper() in {"A", "B", "C", "D"}


def fair_match(gen, gt):
    g, t = _norm(gen), _norm(gt)
    if not t:
        return False
    if _is_mcq(gt):
        return _extract_mcq(gen) == t.upper()
    if g == t:
        return True
    if _syn(g) & _syn(t):
        return True
    if t in g:
        return True
    return False


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def load_samples(samples_dir, cond, bench):
    p = Path(samples_dir) / f"{cond}_{bench}_samples.json"
    if not p.exists():
        return None
    return json.load(open(p))


def score(samples):
    n = len(samples)
    if n == 0:
        return 0.0, 0.0, 0
    default = sum(1 for s in samples if s["correct"]) / n
    fair = sum(1 for s in samples if fair_match(s["model_output"], s["ground_truth"])) / n
    return default, fair, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--samples-dir", default="outputs/internvl3/grounding")
    ap.add_argument("--project", default="learnable-reinspection")
    ap.add_argument(
        "--checkpoint",
        default="models/internvl3/grounding/stage2/epoch_1",
        help="Anchored Stage-2 checkpoint this baseline corresponds to.",
    )
    ap.add_argument(
        "--dataset-mix",
        default="vsr,gqa_spatial,clevr_spatial,vg_spatial,rel3d,cambrian_spatial",
        help="Stage-2 training mix used to produce this checkpoint.",
    )
    ap.add_argument(
        "--notes",
        default="Baseline anchor after _load_ri_checkpoint fix. "
                "Reinspection module + LoRA paired from same Stage-2 epoch_1 dir.",
    )
    args = ap.parse_args()

    cfg = {
        "backend": "internvl3",
        "stage": 2,
        "checkpoint": args.checkpoint,
        "git_commit": _git_commit(),
        "dataset_mix": args.dataset_mix.split(","),
        "matcher": {
            "default": "src.evaluate.match_answer (current)",
            "fair": "synonyms + generous MCQ letter + substring",
        },
        "benchmarks": BENCHES,
        "conditions": CONDS,
    }

    run = wandb.init(
        project=args.project,
        name=args.run_name,
        config=cfg,
        job_type="eval-baseline",
        tags=["baseline", "internvl3", "stage2", "epoch_1", "fixed-loader"],
        notes=args.notes,
        save_code=True,
    )

    # ---- Score everything ----
    # rows[i] = [bench, cond, n, default_acc, fair_acc]
    rows = []
    per_cond_default = {c: [] for c in CONDS}
    per_cond_fair = {c: [] for c in CONDS}
    sw_default = {c: [0, 0] for c in CONDS}  # (correct, total)
    sw_fair = {c: [0, 0] for c in CONDS}

    summary = {}
    for bench in BENCHES:
        for cond in CONDS:
            samples = load_samples(args.samples_dir, cond, bench)
            if samples is None or len(samples) == 0:
                continue
            d_acc, f_acc, n = score(samples)
            rows.append([bench, cond, n, d_acc, f_acc])
            per_cond_default[cond].append(d_acc)
            per_cond_fair[cond].append(f_acc)
            sw_default[cond][0] += int(round(d_acc * n))
            sw_default[cond][1] += n
            sw_fair[cond][0] += sum(
                1 for s in samples if fair_match(s["model_output"], s["ground_truth"])
            )
            sw_fair[cond][1] += n

            # scalars per (bench, cond) so future runs can hit the same keys
            summary[f"default/{bench}/{cond}"] = d_acc
            summary[f"fair/{bench}/{cond}"] = f_acc
            summary[f"n/{bench}/{cond}"] = n

    # per-cond means (per-benchmark unweighted)
    for cond in CONDS:
        if per_cond_default[cond]:
            summary[f"default/mean_per_bench/{cond}"] = sum(per_cond_default[cond]) / len(per_cond_default[cond])
            summary[f"fair/mean_per_bench/{cond}"] = sum(per_cond_fair[cond]) / len(per_cond_fair[cond])
        if sw_default[cond][1] > 0:
            summary[f"default/mean_sample_weighted/{cond}"] = sw_default[cond][0] / sw_default[cond][1]
            summary[f"fair/mean_sample_weighted/{cond}"] = sw_fair[cond][0] / sw_fair[cond][1]

    # gaps reinspection − lora_only (the headline-of-interest deltas)
    for bench in BENCHES:
        r = summary.get(f"fair/{bench}/reinspection")
        l = summary.get(f"fair/{bench}/lora_only")
        if r is not None and l is not None:
            summary[f"fair_gap_reinsp_minus_lora/{bench}"] = r - l

    # ---- Push to W&B ----
    table = wandb.Table(
        columns=["benchmark", "condition", "n", "default_acc", "fair_acc"]
    )
    for r in rows:
        table.add_data(*r)
    wandb.log({"summary_table": table})

    # gap table for quick visual
    gap_rows = []
    for bench in BENCHES:
        for cond in CONDS:
            f_acc = summary.get(f"fair/{bench}/{cond}")
            if f_acc is None:
                continue
            gap_rows.append([bench, cond, f_acc])
    gap_table = wandb.Table(columns=["benchmark", "condition", "fair_acc"])
    for r in gap_rows:
        gap_table.add_data(*r)
    wandb.log({"fair_acc_grid": gap_table})

    # scalars: log each key once, and also into run.summary so they show up on
    # the project Runs table for cross-run diffing
    wandb.log(summary)
    for k, v in summary.items():
        run.summary[k] = v

    # ---- Attach raw sample dumps as artifact ----
    art = wandb.Artifact(
        f"eval-baseline-{Path(args.checkpoint).name}",
        type="eval-baseline",
        metadata={
            "checkpoint": args.checkpoint,
            "dataset_mix": args.dataset_mix,
            "git_commit": cfg["git_commit"],
        },
    )
    for bench in BENCHES:
        for cond in CONDS:
            p = Path(args.samples_dir) / f"{cond}_{bench}_samples.json"
            if p.exists():
                art.add_file(str(p))
    wandb.log_artifact(art)

    # ---- Console table for sanity ----
    print(f"\nLogged to W&B: {run.url}")
    print(f"\n{'bench':<12} {'cond':<13} {'n':>5}  {'default':>8}  {'fair':>8}")
    print("-" * 55)
    for bench, cond, n, d, f in rows:
        print(f"{bench:<12} {cond:<13} {n:>5}  {d:>7.1%}  {f:>7.1%}")

    print(f"\nPer-cond means (default / fair, per-benchmark unweighted):")
    for cond in CONDS:
        d = summary.get(f"default/mean_per_bench/{cond}")
        f = summary.get(f"fair/mean_per_bench/{cond}")
        if d is not None:
            print(f"  {cond:<13} {d:.1%} / {f:.1%}")

    wandb.finish()


if __name__ == "__main__":
    main()
