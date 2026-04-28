"""Lazy public exports for vision-language backends with re-inspection wrappers."""

from __future__ import annotations

import importlib

_EXPORTS = {
    "Qwen25VLWithReInspection": ("src.backends.qwen25vl", "Qwen25VLWithReInspection"),
    "InternVL3WithReInspection": ("src.backends.internvl3", "InternVL3WithReInspection"),
    "Gemma4WithReInspection": ("src.backends.gemma4", "Gemma4WithReInspection"),
    "load_qwen25_model": ("src.backends.qwen25vl", "load_model"),
    "load_internvl_model": ("src.backends.internvl3", "load_model"),
    "load_gemma4_model": ("src.backends.gemma4", "load_model"),
    "load_processor": ("src.backends.internvl3", "load_processor"),
    "load_gemma4_processor": ("src.backends.gemma4", "load_processor"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(importlib.import_module(module_name), attr_name)
    globals()[name] = value
    return value
