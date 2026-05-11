"""Tests for Stage-2 auxiliary grounding configuration helpers."""

import torch

from src.config import ReInspectionConfig
from src.training.trainer import _stage2_aux_grounding_enabled, _stage2_optimizer


class _TinyStage2Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.reinspection = torch.nn.Linear(2, 2, bias=False)


def test_stage2_aux_grounding_requires_weight_and_cadence():
    config = ReInspectionConfig(stage=2)
    assert not _stage2_aux_grounding_enabled(config)

    config.stage2_aux_grounding_weight = 0.05
    assert not _stage2_aux_grounding_enabled(config)

    config.stage2_aux_every_n_steps = 4
    assert _stage2_aux_grounding_enabled(config)

    config.stage = 1
    assert not _stage2_aux_grounding_enabled(config)


def test_stage2_optimizer_can_freeze_reinspection_group():
    model = _TinyStage2Model()
    lora_param = torch.nn.Parameter(torch.ones(2, 2))
    config = ReInspectionConfig(stage=2, stage2_train_reinspection=False)
    for param in model.reinspection.parameters():
        param.requires_grad = config.stage2_train_reinspection

    optimizer = _stage2_optimizer(config, model, [lora_param])

    assert [group["name"] for group in optimizer.param_groups] == ["lora"]
    assert optimizer.param_groups[0]["lr"] == config.stage2_lr_lora


def test_stage2_optimizer_keeps_reinspection_group_by_default():
    model = _TinyStage2Model()
    lora_param = torch.nn.Parameter(torch.ones(2, 2))
    config = ReInspectionConfig(stage=2)

    optimizer = _stage2_optimizer(config, model, [lora_param])

    assert [group["name"] for group in optimizer.param_groups] == ["reinspection", "lora"]
    assert optimizer.param_groups[0]["lr"] == config.stage2_lr_module
    assert optimizer.param_groups[1]["lr"] == config.stage2_lr_lora
