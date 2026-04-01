"""Shared model output dataclass for re-inspection backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from transformers.modeling_outputs import ModelOutput


@dataclass
class ReInspectionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[tuple] = None
    hidden_states: Optional[tuple] = None
    attentions: Optional[tuple] = None
    rope_deltas: Optional[torch.LongTensor] = None
    image_hidden_states: Optional[torch.FloatTensor] = None
    attn_task: Optional[torch.FloatTensor] = None
    attn_vis: Optional[torch.FloatTensor] = None
