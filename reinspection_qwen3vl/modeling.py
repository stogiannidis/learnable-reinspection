"""Compatibility shim — use ``reinspection_vlm.backends.qwen3vl``."""

from reinspection_vlm.backends.qwen3vl import Qwen3VLWithReInspection, load_model
from reinspection_vlm.outputs import ReInspectionOutput

__all__ = ["Qwen3VLWithReInspection", "load_model", "ReInspectionOutput"]
