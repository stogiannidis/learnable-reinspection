"""Shared utilities: Hydra helpers and attention visualization."""

from .hydra_util import strip_deepspeed_local_rank_argv

__all__ = ["strip_deepspeed_local_rank_argv"]
