import json

from src.config import ReInspectionConfig
from src.eval_prompting import (
    build_eval_question,
    extract_final_answer,
    load_frozen_cache,
    save_frozen_cache,
)


def test_build_eval_question_appends_cot_instruction():
    config = ReInspectionConfig(eval_cot_enabled=True, eval_cot_prompt="Think step by step.")
    prompted = build_eval_question("Is the cup above the plate?", config)
    assert prompted.startswith("Is the cup above the plate?")
    assert prompted.endswith("Think step by step.")


def test_build_eval_question_disabled_returns_original():
    config = ReInspectionConfig(eval_cot_enabled=False)
    question = "Is the cup above the plate?"
    assert build_eval_question(question, config) == question


def test_extract_final_answer_prefers_marker():
    text = (
        "The cup is clearly above the plate in the image.\n"
        "Final answer: yes"
    )
    assert extract_final_answer(text) == "yes"


def test_extract_final_answer_uses_last_line_fallback():
    text = "First I look at the left object.\nThen I compare positions.\nabove"
    assert extract_final_answer(text) == "above"


def test_cot_pipeline_extracts_scorable_spatial_answer():
    generated = (
        "The red ball is on the left side of the blue box.\n"
        "Final answer: left"
    )
    assert extract_final_answer(generated) == "left"


def test_cot_pipeline_extracts_scorable_mcq_answer():
    generated = (
        "Option A shows the object on top.\n"
        "Option B shows it below.\n"
        "Final answer: B"
    )
    assert extract_final_answer(generated) == "B"


def test_frozen_cache_roundtrip_with_cot_metadata(tmp_path):
    config = ReInspectionConfig(
        eval_cot_enabled=True,
        eval_cot_prompt="Think step by step. Final answer:",
    )
    cache_file = tmp_path / "frozen_cache.json"
    records = [{"benchmark": "vsr", "accuracy": 0.5, "correct": 1, "total": 2}]
    save_frozen_cache(str(cache_file), records, config)

    loaded = load_frozen_cache(str(cache_file), config)
    assert loaded == records


def test_frozen_cache_legacy_list_rejected_when_cot_enabled(tmp_path):
    config = ReInspectionConfig(eval_cot_enabled=True)
    cache_file = tmp_path / "legacy_cache.json"
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump([{"benchmark": "vsr", "accuracy": 0.5, "correct": 1, "total": 2}], f)

    assert load_frozen_cache(str(cache_file), config) is None


def test_frozen_cache_metadata_mismatch_rejected(tmp_path):
    config = ReInspectionConfig(
        eval_cot_enabled=True,
        eval_cot_prompt="New prompt",
    )
    cache_file = tmp_path / "mismatch_cache.json"
    payload = {
        "metadata": {
            "eval_cot_enabled": True,
            "eval_cot_prompt": "Old prompt",
        },
        "results": [{"benchmark": "vsr", "accuracy": 0.5, "correct": 1, "total": 2}],
    }
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    assert load_frozen_cache(str(cache_file), config) is None
