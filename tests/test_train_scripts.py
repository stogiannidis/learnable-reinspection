"""Tests for shell training launcher helpers."""

import subprocess


def _bash(command: str) -> str:
    result = subprocess.run(
        ["bash", "-lc", command],
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def test_configured_stage1_checkpoint_reads_stage2_experiment_default():
    checkpoint = _bash(
        "source src/scripts/train_common.sh && "
        "configured_stage1_checkpoint s2_internvl_grounding"
    )

    assert checkpoint == "models/internvl3/stage1/epoch_5/reinspection_module.pt"


def test_stage2_launcher_uses_stage2_config_when_no_stage1_experiment_is_set():
    checkpoint = _bash(
        "source src/scripts/train_common.sh && "
        "unset STAGE1_CHECKPOINT REINSPECTION_EXPERIMENT && "
        "STAGE2_EXPERIMENT=s2_internvl_grounding && "
        "STAGE1_CHECKPOINT=$(configured_stage1_checkpoint \"$STAGE2_EXPERIMENT\") && "
        "printf '%s' \"$STAGE1_CHECKPOINT\""
    )

    assert checkpoint.startswith("models/internvl3/")
