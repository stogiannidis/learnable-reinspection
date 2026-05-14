"""Generate spatial minimal pairs from VSR-style data.

For each sample with an invertible spatial relation (e.g. "above"/"below"),
produce a paired sample that swaps subject and object.  The original answer
is preserved for the original question; the flipped question gets the
*opposite* ground-truth answer because the spatial relation is inverted.

Two flavours of pairs are produced:

1. **Subject–object swap** (same relation, swap arguments):
   "The cat is above the dog" (True) → "The dog is above the cat" (False)

2. **Relation inversion** (same arguments, invert relation):
   "The cat is above the dog" (True) → "The cat is below the dog" (False)

Both test the same underlying competence but with different linguistic cues.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Spatial relations that have a clear directional inverse.
RELATION_INVERSES: Dict[str, str] = {
    "above": "below",
    "below": "above",
    "left of": "right of",
    "right of": "left of",
    "in front of": "behind",
    "behind": "in front of",
    "on top of": "under",
    "under": "on top of",
    "beneath": "on top of",
    "over": "under",
    "to the left of": "to the right of",
    "to the right of": "to the left of",
}

# Asymmetric relations where swapping subject/object changes the truth value.
# Only these are valid for swap_args pairs.  Symmetric relations (touching,
# facing, near, beside, ...) stay true/false regardless of argument order.
ASYMMETRIC_RELATIONS: set = set(RELATION_INVERSES.keys()) | {
    "inside", "in", "into", "on", "at", "under", "over",
    "ahead of", "down from", "enclosed by",
}

_STMT_RE = re.compile(
    r'^Is the following statement true or false about the image\? "The (.+?) is (.+?) the (.+?)\."'
)


@dataclass
class MinimalPair:
    """A pair of questions about the same image with expected opposite answers."""
    image: str
    question_a: str
    answer_a: str
    question_b: str
    answer_b: str
    pair_type: str       # "swap_args" | "invert_rel"
    relation: str
    subject: str
    object: str


def _flip_answer(answer: str) -> str:
    return "False" if answer.strip().lower() == "true" else "True"


def _build_vsr_question(subject: str, relation: str, obj: str) -> str:
    return (
        f'Is the following statement true or false about the image? '
        f'"The {subject} is {relation} the {obj}." '
        f'Answer with just True or False.'
    )


def parse_vsr_statement(question: str) -> Optional[Tuple[str, str, str]]:
    """Extract (subject, relation, object) from a VSR question string."""
    m = _STMT_RE.match(question)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None


def generate_pairs_from_vsr(
    data_file: str,
    split: str = "test",
) -> List[MinimalPair]:
    """Read a VSR JSONL file and produce minimal pairs.

    Returns a list of MinimalPair objects.  Each original sample with an
    invertible relation yields up to two pairs (swap_args + invert_rel).
    """
    path = Path(data_file)
    items = []
    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                if item.get("split", split) == split:
                    items.append(item)
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = [s for s in data if s.get("split", split) == split]

    pairs: List[MinimalPair] = []
    for item in items:
        parsed = parse_vsr_statement(item["question"])
        if parsed is None:
            continue
        subj, rel, obj = parsed
        answer = item["answer"]
        image = item["image"]

        # --- Pair type 1: swap subject and object (same relation) ---
        # Only valid for asymmetric relations; for symmetric ones (touching,
        # near, beside, ...) swapping args doesn't change the truth value.
        if rel in ASYMMETRIC_RELATIONS:
            q_swap = _build_vsr_question(obj, rel, subj)
            pairs.append(MinimalPair(
                image=image,
                question_a=item["question"],
                answer_a=answer,
                question_b=q_swap,
                answer_b=_flip_answer(answer),
                pair_type="swap_args",
                relation=rel,
                subject=subj,
                object=obj,
            ))

        # --- Pair type 2: invert the relation (same subject/object) ---
        if rel in RELATION_INVERSES:
            inv_rel = RELATION_INVERSES[rel]
            q_inv = _build_vsr_question(subj, inv_rel, obj)
            pairs.append(MinimalPair(
                image=image,
                question_a=item["question"],
                answer_a=answer,
                question_b=q_inv,
                answer_b=_flip_answer(answer),
                pair_type="invert_rel",
                relation=rel,
                subject=subj,
                object=obj,
            ))

    return pairs


def save_pairs_jsonl(pairs: List[MinimalPair], output_path: str) -> None:
    """Write minimal pairs to a JSONL file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps({
                "image": p.image,
                "question_a": p.question_a,
                "answer_a": p.answer_a,
                "question_b": p.question_b,
                "answer_b": p.answer_b,
                "pair_type": p.pair_type,
                "relation": p.relation,
                "subject": p.subject,
                "object": p.object,
            }) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate spatial minimal pairs from VSR data")
    parser.add_argument("--data_file", type=str, default="/data/datasets/vsr/test.jsonl")
    parser.add_argument("--output", type=str, default="outputs/motivation/minimal_pairs.jsonl")
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    pairs = generate_pairs_from_vsr(args.data_file, split=args.split)
    save_pairs_jsonl(pairs, args.output)

    # Stats
    swap_pairs = [p for p in pairs if p.pair_type == "swap_args"]
    inv_pairs = [p for p in pairs if p.pair_type == "invert_rel"]
    rels = set(p.relation for p in inv_pairs)
    print(f"Generated {len(pairs)} minimal pairs:")
    print(f"  swap_args:  {len(swap_pairs)}")
    print(f"  invert_rel: {len(inv_pairs)} (relations: {sorted(rels)})")
