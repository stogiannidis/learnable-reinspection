"""Re-score every existing ``{condition}_{benchmark}_samples.json`` under the
patched ``match_answer`` (option-text MCQ fix + bool/direction canonical
equivalence) and update the ``correct`` field in place.

The model outputs themselves (``model_output``) are not touched; only the
matcher verdict is updated. A ``correct_old`` field is preserved alongside the
new ``correct`` value so before/after deltas can be recovered without re-running
inference.
"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path("outputs/internvl3/grounding")
CONDS = ["frozen", "lora_only", "reinspection"]
BENCHES = ["vsr", "gqa_spatial", "whatsup", "3dsrbench", "blink", "srbench"]

# ---- Inlined matcher (mirrors src/evaluate.match_answer post-patch) ----
_MCQ_LETTERS = frozenset("ABCDEF")

_CANONICAL_GROUPS = {}
for _phrases, _key in [
    (["true", "yes", "correct", "affirmative"],              "pos"),
    (["false", "no", "incorrect", "negative"],               "neg"),
    (["left", "to the left", "to the left of",
      "on the left", "on the left of", "left of",
      "left side"],                                          "left"),
    (["right", "to the right", "to the right of",
      "on the right", "on the right of", "right of",
      "right side"],                                         "right"),
    (["above", "on top of", "on top", "over"],               "above"),
    (["below", "underneath", "under", "beneath"],            "below"),
    (["behind", "in back of", "at the back of"],             "behind"),
    (["in front of", "in front", "front of", "ahead of"],    "front"),
    (["inside", "within", "into"],                           "inside"),
    (["outside", "out of"],                                  "outside"),
]:
    for _p in _phrases:
        _CANONICAL_GROUPS[_p] = _key
_CANONICAL_PHRASES = sorted(_CANONICAL_GROUPS.keys(), key=len, reverse=True)


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


def _leading_canonical(text):
    n = _norm(text)
    if not n:
        return None
    head = n[:60]
    for phrase in _CANONICAL_PHRASES:
        if re.match(r"\W*" + re.escape(phrase) + r"\b", head):
            return _CANONICAL_GROUPS[phrase]
    return None


def match(gen, gt):
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
    gt_key = _CANONICAL_GROUPS.get(t)
    if gt_key is not None:
        gen_key = _leading_canonical(gen)
        if gen_key is not None:
            return gen_key == gt_key
    return g == t or (bool(t) and t in g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Report deltas without writing back to disk.")
    args = ap.parse_args()

    print(f"{'bench':<12} {'cond':<13} {'old':>7} {'new':>7} {'Δpp':>7}  +/-")
    print("-" * 56)
    summary = {}
    for bench in BENCHES:
        for cond in CONDS:
            p = ROOT / f"{cond}_{bench}_samples.json"
            if not p.exists():
                continue
            ss = json.load(open(p))
            if not ss:
                continue
            old = sum(1 for s in ss if s["correct"])
            gain = loss = 0
            for s in ss:
                old_c = bool(s["correct"])
                new_c = match(s["model_output"], s["ground_truth"])
                if new_c and not old_c:
                    gain += 1
                if (not new_c) and old_c:
                    loss += 1
                # store legacy verdict for traceability, then overwrite
                if "correct_old" not in s:
                    s["correct_old"] = old_c
                s["correct"] = bool(new_c)
            new = sum(1 for s in ss if s["correct"])
            n = len(ss)
            d = (new - old) / n * 100
            mark = " *" if abs(new - old) >= 5 else ""
            print(f"{bench:<12} {cond:<13} {old/n*100:6.2f} {new/n*100:6.2f} {d:+6.2f}  +{gain}/-{loss}{mark}")
            summary.setdefault(cond, []).append(new / n)
            if not args.dry_run:
                json.dump(ss, open(p, "w"), indent=2)
        print()

    print("Per-bench unweighted mean (patched matcher):")
    for cond in CONDS:
        accs = summary.get(cond, [])
        if accs:
            print(f"  {cond:<13}  {sum(accs)/len(accs)*100:.2f}%")

    if args.dry_run:
        print("\n[dry-run] no files were modified.")


if __name__ == "__main__":
    main()
