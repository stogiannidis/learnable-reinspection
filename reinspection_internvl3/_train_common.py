"""Compatibility shim — InternVL training API (stage-only) for ``train_stage*.py``."""

from reinspection_vlm.attn_loss import compute_attn_loss_kl as compute_attn_loss
from reinspection_vlm.config import ReInspectionConfig
from reinspection_vlm.train_common import (
    collate_fn,
    is_main_process,
    load_config,
    log,
    run_training as _run_training,
)

_collate_fn = collate_fn


def run_training(stage: int, config: ReInspectionConfig, args) -> None:
    """Delegate to unified trainer with ``backend='internvl3'``."""
    _run_training("internvl3", stage, config, args)
