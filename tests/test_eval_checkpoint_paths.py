import os

from src.config import ReInspectionConfig
from src.eval_checkpoints import (
    find_lora_weights_path,
    find_reinspection_module_path,
    log_eval_checkpoint_plan,
    lora_only_checkpoint_root,
    print_eval_model_paths,
    resolve_eval_model_paths,
)


def test_resolve_eval_model_paths_reinspection_prefers_stage2_module(tmp_path, capsys):
    stage1 = tmp_path / "stage1"
    stage2 = tmp_path / "stage2"
    stage1.mkdir()
    stage2.mkdir()
    (stage1 / "reinspection_module.pt").write_text("s1")
    (stage2 / "reinspection_module.pt").write_text("s2")
    (stage2 / "lora_weights").mkdir()

    paths = resolve_eval_model_paths(
        "reinspection",
        checkpoint_dir=str(stage1),
        lora_checkpoint_dir=str(stage2),
        model_name_or_path="OpenGVLab/InternVL3-8B-hf",
    )

    assert paths["reinspection_module"] == str(stage2 / "reinspection_module.pt")
    assert paths["lora_weights"] == str(stage2 / "lora_weights")


def test_resolve_eval_model_paths_lora_only_uses_lora_checkpoint_dir(tmp_path):
    stage1 = tmp_path / "stage1"
    stage2 = tmp_path / "stage2"
    stage1.mkdir()
    stage2.mkdir()
    (stage2 / "lora_weights").mkdir()

    paths = resolve_eval_model_paths(
        "lora_only",
        checkpoint_dir=str(stage1),
        lora_checkpoint_dir=str(stage2),
        model_name_or_path="OpenGVLab/InternVL3-8B-hf",
    )

    assert paths["checkpoint_root"] == str(stage2)
    assert paths["lora_weights"] == str(stage2 / "lora_weights")
    assert paths["reinspection_module"] is None


def test_log_eval_checkpoint_plan_prints_configured_paths(tmp_path, capsys):
    stage1 = tmp_path / "stage1"
    stage2 = tmp_path / "stage2"
    cache = tmp_path / "frozen.json"
    stage1.mkdir()
    stage2.mkdir()
    (stage2 / "reinspection_module.pt").write_text("s2")
    (stage2 / "lora_weights").mkdir()
    cache.write_text("[]")

    config = ReInspectionConfig(
        checkpoint_dir=str(stage1),
        lora_checkpoint_dir=str(stage2),
        frozen_cache_file=str(cache),
        eval_compare=True,
    )
    log_eval_checkpoint_plan(config, ["frozen", "lora_only", "reinspection"])

    out = capsys.readouterr().out
    assert "Saved model paths (evaluation):" in out
    assert f"checkpoint_dir: {os.path.abspath(stage1)}" in out
    assert f"lora_checkpoint_dir: {os.path.abspath(stage2)}" in out
    assert f"frozen_cache_file: {os.path.abspath(cache)}" in out
    assert "Saved model paths (reinspection):" in out
    assert str(stage2 / "reinspection_module.pt") in out
