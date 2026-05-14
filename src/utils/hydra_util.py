"""Hydra compatibility shims for DeepSpeed-launched Python processes."""

import sys


def strip_deepspeed_local_rank_argv() -> None:
    """Remove ``--local_rank=...`` entries from ``sys.argv`` before Hydra parses.

    DeepSpeed injects a local rank flag that Hydra would otherwise interpret as
    an unknown config override; stripping it keeps composition strict while
    preserving all other launcher arguments.
    """
    sys.argv = [a for a in sys.argv if not a.startswith("--local_rank")]
