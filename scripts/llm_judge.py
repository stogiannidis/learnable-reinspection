"""LLM-as-a-Judge evaluation via OpenRouter.

Compares a model's generated response against a ground-truth reference using
semantic understanding rather than exact-string matching.

Library usage::

    from scripts.llm_judge import judge_response

    result = judge_response(
        user_prompt="Is the cat to the left of the dog?",
        model_response="Yes, the cat is on the left.",
        reference_answer="yes",
    )
    # -> {"is_correct": True, "reasoning": "..."}

Batch CLI (mirrors ``scripts/rescore_samples.py`` layout)::

    # Set OPENROUTER_API_KEY in .env or the environment
    python scripts/llm_judge.py --dry-run --limit 10
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from openai import APIError, OpenAI, RateLimitError
from dotenv import load_dotenv
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_REPO_ROOT / ".env")

# ---------------------------------------------------------------------------
# Configuration (edit here to change defaults)
# ---------------------------------------------------------------------------
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
JUDGE_MODEL = "google/gemini-3.5-flash"
API_KEY_ENV = "OPENROUTER_API_KEY"

DEFAULT_ROOT = Path("outputs/internvl3/grounding")
DEFAULT_CONDS = ("frozen", "lora_only", "reinspection")
DEFAULT_BENCHES = ("vsr", "gqa_spatial", "whatsup", "3dsrbench", "blink", "srbench")

SYSTEM_PROMPT = """You are an impartial answer grader for vision-language QA.

Given a user question, a model's answer, and a reference (ground-truth) answer,
decide whether the model answer is correct.

Grading rules:
- Judge factual accuracy and intent relative to the reference, not wording.
- Accept paraphrases, synonyms, and equivalent phrasing.
- Accept MCQ letter answers that match the referenced option text, and vice versa.
- Accept canonical equivalents (e.g. yes/true, left/to the left, above/on top of).
- Mark incorrect if the model contradicts the reference or adds wrong facts.
- If the reference is ambiguous, prefer leniency when the model clearly matches
  the reference's meaning.

Respond with ONLY a JSON object (no markdown, no extra text) in this schema:
{"is_correct": true or false, "reasoning": "one short sentence"}
"""

_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


# ---------------------------------------------------------------------------
# OpenRouter client
# ---------------------------------------------------------------------------
def _build_client() -> OpenAI:
    """Create an OpenAI SDK client pointed at OpenRouter."""
    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise RuntimeError(
            f"Missing API key: set {API_KEY_ENV} in the environment or in "
            f"{_REPO_ROOT / '.env'}."
        )
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)


def _format_user_message(
    user_prompt: str,
    model_response: str,
    reference_answer: str,
) -> str:
    return (
        "Grade the model answer against the reference.\n\n"
        f"Question:\n{user_prompt}\n\n"
        f"Model answer:\n{model_response}\n\n"
        f"Reference answer:\n{reference_answer}"
    )


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Parse JSON from model output, with a regex fallback."""
    text = (text or "").strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = _JSON_OBJECT_RE.search(text)
    if match is None:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _normalize_judge_result(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Validate and coerce the judge JSON into the public return schema."""
    if raw is None:
        return {
            "is_correct": None,
            "reasoning": "judge_parse_error: could not parse JSON from model output",
        }

    if "is_correct" not in raw:
        return {
            "is_correct": None,
            "reasoning": "judge_parse_error: missing is_correct field",
        }

    value = raw["is_correct"]
    if isinstance(value, bool):
        is_correct = value
    elif isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            is_correct = True
        elif lowered in {"false", "no", "0"}:
            is_correct = False
        else:
            return {
                "is_correct": None,
                "reasoning": f"judge_parse_error: invalid is_correct string {value!r}",
            }
    elif isinstance(value, (int, float)):
        is_correct = bool(value)
    else:
        return {
            "is_correct": None,
            "reasoning": f"judge_parse_error: invalid is_correct type {type(value).__name__}",
        }

    reasoning = raw.get("reasoning", "")
    if reasoning is None:
        reasoning = ""
    if not isinstance(reasoning, str):
        reasoning = str(reasoning)

    return {"is_correct": is_correct, "reasoning": reasoning.strip()}


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------
def judge_response(
    user_prompt: str,
    model_response: str,
    reference_answer: str,
    *,
    client: OpenAI | None = None,
    model: str = JUDGE_MODEL,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Judge whether ``model_response`` is semantically correct vs the reference.

    Returns ``{'is_correct': bool, 'reasoning': str}`` on success.
    On irrecoverable failure returns ``{'is_correct': None, 'reasoning': '...'}``
    so batch mode can continue without aborting.
    """
    if client is None:
        client = _build_client()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _format_user_message(
                user_prompt, model_response, reference_answer
            ),
        },
    ]

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            content = completion.choices[0].message.content or ""
            raw = _extract_json_object(content)
            return _normalize_judge_result(raw)
        except RateLimitError as exc:
            last_error = exc
            if attempt + 1 >= max_retries:
                break
            time.sleep(2 ** attempt)
        except APIError as exc:
            last_error = exc
            # Retry only on likely-transient HTTP statuses.
            status = getattr(exc, "status_code", None)
            if status is not None and status not in {408, 429, 500, 502, 503, 504}:
                break
            if attempt + 1 >= max_retries:
                break
            time.sleep(2 ** attempt)
        except Exception as exc:  # noqa: BLE001 — surface unexpected errors per sample
            last_error = exc
            break

    reason = f"judge_api_error: {last_error}" if last_error else "judge_api_error: unknown"
    return {"is_correct": None, "reasoning": reason}


# ---------------------------------------------------------------------------
# Batch CLI
# ---------------------------------------------------------------------------
def _accuracy(samples: list[dict[str, Any]], field: str) -> float | None:
    """Fraction of samples where ``field`` is truthy; None if no valid labels."""
    valid = [s for s in samples if s.get(field) is not None]
    if not valid:
        return None
    return sum(1 for s in valid if s[field]) / len(valid)


def _judge_samples(
    samples: list[dict[str, Any]],
    client: OpenAI,
    *,
    limit: int | None,
    skip_if_present: bool,
) -> tuple[int, int, int]:
    """Annotate samples in place. Returns (judged, skipped, errors)."""
    judged = skipped = errors = 0
    todo = samples[:limit] if limit is not None else samples

    for sample in tqdm(todo, desc="judging", leave=False):
        if skip_if_present and "is_correct_llm" in sample:
            skipped += 1
            continue

        result = judge_response(
            sample.get("question", ""),
            sample.get("model_output", ""),
            sample.get("ground_truth", ""),
            client=client,
        )
        sample["is_correct_llm"] = result["is_correct"]
        sample["llm_judge_reasoning"] = result["reasoning"]

        if result["is_correct"] is None:
            errors += 1
        judged += 1

    return judged, skipped, errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-as-a-Judge semantic rescoring for eval sample JSON files."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Directory containing *_samples.json files (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(DEFAULT_CONDS),
        help="Eval conditions to process.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=list(DEFAULT_BENCHES),
        help="Benchmarks to process.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Judge at most N samples per file (cost cap).",
    )
    parser.add_argument(
        "--skip-if-present",
        action="store_true",
        help="Skip samples that already have is_correct_llm.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report deltas without writing files back to disk.",
    )
    args = parser.parse_args()

    client = _build_client()

    print(
        f"{'bench':<12} {'cond':<13} {'det%':>7} {'llm%':>7} {'Δpp':>7}  "
        f"judged  skip  err  +/-"
    )
    print("-" * 72)

    summary_det: dict[str, list[float]] = {}
    summary_llm: dict[str, list[float]] = {}

    for bench in args.benchmarks:
        for cond in args.conditions:
            path = args.root / f"{cond}_{bench}_samples.json"
            if not path.exists():
                continue

            with open(path, encoding="utf-8") as f:
                samples: list[dict[str, Any]] = json.load(f)
            if not samples:
                continue

            judged, skipped, errors = _judge_samples(
                samples,
                client,
                limit=args.limit,
                skip_if_present=args.skip_if_present,
            )

            new_det = _accuracy(samples, "correct")
            new_llm = _accuracy(samples, "is_correct_llm")

            gain = loss = 0
            for s in samples:
                det = bool(s.get("correct"))
                llm = s.get("is_correct_llm")
                if llm is None:
                    continue
                llm_bool = bool(llm)
                if llm_bool and not det:
                    gain += 1
                if (not llm_bool) and det:
                    loss += 1

            n = len(samples)
            det_pct = (new_det or 0.0) * 100
            llm_pct = (new_llm or 0.0) * 100
            delta = llm_pct - det_pct
            mark = " *" if abs(delta) >= 5 else ""

            print(
                f"{bench:<12} {cond:<13} {det_pct:6.2f} {llm_pct:6.2f} {delta:+6.2f}  "
                f"{judged:5d} {skipped:4d} {errors:3d}  +{gain}/-{loss}{mark}"
            )

            if new_det is not None:
                summary_det.setdefault(cond, []).append(new_det)
            if new_llm is not None:
                summary_llm.setdefault(cond, []).append(new_llm)

            if not args.dry_run:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(samples, f, indent=2)

        print()

    print("Per-bench unweighted mean:")
    for cond in args.conditions:
        det_accs = summary_det.get(cond, [])
        llm_accs = summary_llm.get(cond, [])
        if det_accs:
            det_mean = sum(det_accs) / len(det_accs) * 100
            llm_mean = sum(llm_accs) / len(llm_accs) * 100 if llm_accs else float("nan")
            print(f"  {cond:<13}  det={det_mean:.2f}%  llm={llm_mean:.2f}%")

    if args.dry_run:
        print("\n[dry-run] no files were modified.")


if __name__ == "__main__":
    main()
