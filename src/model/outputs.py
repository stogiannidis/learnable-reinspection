"""Shared model output container for re-inspection VLMs.

Subclasses :class:`transformers.modeling_outputs.ModelOutput` so trainers can
consume the same field names as Hugging Face models while attaching auxiliary
tensors used for stage-1 supervision (attention alignment, bounding-box head).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from transformers.modeling_outputs import ModelOutput


@dataclass
class ReInspectionOutput(ModelOutput):
    """Forward pass outputs from a backend wrapped with a re-inspection module.

    Standard language-modeling fields mirror ``CausalLMOutputWithPast``.  The
    additional tensors support analysis and stage-1 losses:

    - ``attn_task`` / ``attn_vis``: aggregated attention from bottleneck queries
      over text and vision tokens (when the backend exposes weights).
    - ``R_bottleneck``: final query states in reduced dimension before the
      up-projection, fed to the optional :class:`~src.model.bbox_head.BboxHead`.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[tuple] = None
    hidden_states: Optional[tuple] = None
    attentions: Optional[tuple] = None
    rope_deltas: Optional[torch.LongTensor] = None
    image_hidden_states: Optional[torch.FloatTensor] = None
    attn_task: Optional[torch.FloatTensor] = None
    attn_vis: Optional[torch.FloatTensor] = None
    R_bottleneck: Optional[torch.FloatTensor] = None
