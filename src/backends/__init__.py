"""Public exports for vision-language backends with re-inspection wrappers.

Each submodule pairs a frozen (or PEFT-wrapped) base VLM with
:class:`~src.model.reinspection_module.ReInspectionModule` and exposes
``load_model`` / ``load_processor`` entry points used by training and evaluation.
"""

from .qwen25vl import Qwen25VLWithReInspection, load_model as load_qwen25_model
from .internvl3 import InternVL3WithReInspection, load_model as load_internvl_model, load_processor
from .gemma4 import Gemma4WithReInspection, load_model as load_gemma4_model, load_processor as load_gemma4_processor

__all__ = [
    "Qwen25VLWithReInspection",
    "InternVL3WithReInspection",
    "Gemma4WithReInspection",
    "load_qwen25_model",
    "load_internvl_model",
    "load_gemma4_model",
    "load_processor",
    "load_gemma4_processor",
]
