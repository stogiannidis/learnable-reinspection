"""Compatibility shim."""

from reinspection_vlm.backends.internvl3 import InternVL3WithReInspection, load_model, load_processor
from reinspection_vlm.outputs import ReInspectionOutput

__all__ = ["InternVL3WithReInspection", "load_model", "load_processor", "ReInspectionOutput"]
