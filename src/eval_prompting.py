"""Evaluation prompt and scoring helpers (CoT prompting, final-answer extraction)."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from src.config import ReInspectionConfig

_FINAL_ANSWER_MARKERS = (
    "final answer:",
    "answer:",
)


def build_eval_question(question: str, config: ReInspectionConfig) -> str:
    """Append the configured CoT instruction to a benchmark question when enabled."""
    if not config.eval_cot_enabled:
        return question
    cot = config.eval_cot_prompt.strip()
    if not cot:
        return question
    return f"{question.strip()}\n\n{cot}"


def extract_final_answer(generated: str, cot_enabled: bool = True) -> str:
    """Extract the scored answer from a visible-reasoning model output.

    Prefers text after ``Final answer:`` / ``Answer:`` markers, then falls back
    to the last non-empty line when CoT prompting is enabled.
    """
    text = (generated or "").strip()
    if not text or not cot_enabled:
        return text

    lower = text.lower()
    best_idx = -1
    best_marker_len = 0
    for marker in _FINAL_ANSWER_MARKERS:
        idx = lower.rfind(marker)
        if idx != -1 and idx >= best_idx:
            best_idx = idx
            best_marker_len = len(marker)

    if best_idx != -1:
        answer = text[best_idx + best_marker_len :].strip()
        answer = answer.split("\n", 1)[0].strip()
        if answer:
            return answer

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        return lines[-1]
    return text


def frozen_cache_metadata(config: ReInspectionConfig) -> Dict[str, object]:
    """Prompt metadata stored alongside cached frozen-baseline results."""
    return {
        "eval_cot_enabled": config.eval_cot_enabled,
        "eval_cot_prompt": config.eval_cot_prompt if config.eval_cot_enabled else "",
    }


def load_frozen_cache(cache_file: str, config: ReInspectionConfig) -> Optional[List[dict]]:
    """Load cached frozen results when prompt metadata matches the current eval run."""
    with open(cache_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    expected = frozen_cache_metadata(config)
    if isinstance(payload, list):
        if config.eval_cot_enabled:
            print(
                "  Frozen cache uses legacy list format; recomputing frozen baseline "
                "for CoT-enabled evaluation.",
                flush=True,
            )
            return None
        return payload

    if isinstance(payload, dict):
        meta = payload.get("metadata", {})
        if (
            meta.get("eval_cot_enabled") == expected["eval_cot_enabled"]
            and meta.get("eval_cot_prompt", "") == expected["eval_cot_prompt"]
        ):
            results = payload.get("results")
            return results if isinstance(results, list) else None
        print(
            "  Frozen cache prompt metadata mismatch; recomputing frozen baseline.",
            flush=True,
        )
        return None

    print("  Frozen cache format not recognized; recomputing frozen baseline.", flush=True)
    return None


def save_frozen_cache(
    cache_file: str,
    condition_results: List[dict],
    config: ReInspectionConfig,
) -> None:
    """Persist frozen-baseline metrics with prompt metadata for future reuse."""
    cache_records = [{k: v for k, v in r.items() if k != "samples"} for r in condition_results]
    payload = {
        "metadata": frozen_cache_metadata(config),
        "results": cache_records,
    }
    os.makedirs(os.path.dirname(cache_file) or ".", exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
