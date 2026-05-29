"""Attention-backend resolution with graceful FlashAttention 2 fallback.

``flash_attention_2`` requires the optional ``flash-attn`` package. When it is
not installed in the current environment, requesting it makes
``transformers.from_pretrained`` raise ``ImportError``. To keep eval/train jobs
robust across containers that may or may not ship ``flash-attn``, this resolver
falls back to ``sdpa`` (PyTorch's built-in fused/memory-efficient attention,
which already dispatches to FlashAttention kernels when available) instead of
crashing.
"""

from __future__ import annotations

import importlib.util
from typing import Optional

_warned = False


def _flash_attn_available() -> bool:
    """Return True if the ``flash_attn`` package can be imported."""
    return importlib.util.find_spec("flash_attn") is not None


def resolve_attn_implementation(requested: Optional[str]) -> Optional[str]:
    """Resolve an attention backend, downgrading FA2 to sdpa when unavailable.

    Args:
        requested: Desired ``attn_implementation`` (``flash_attention_2``,
            ``sdpa``, ``eager``, or ``None`` to let HF pick its default).

    Returns:
        The usable backend string. ``flash_attention_2`` is replaced with
        ``sdpa`` when ``flash-attn`` is not installed.
    """
    global _warned
    if requested == "flash_attention_2" and not _flash_attn_available():
        if not _warned:
            print(
                "[attn] flash_attention_2 requested but flash-attn is not "
                "installed; falling back to sdpa.",
                flush=True,
            )
            _warned = True
        return "sdpa"
    return requested
