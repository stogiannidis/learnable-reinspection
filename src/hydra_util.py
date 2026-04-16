"""Shared Hydra / launcher helpers."""

import sys


def strip_deepspeed_local_rank_argv() -> None:
    """Remove ``--local_rank=`` from ``sys.argv`` so Hydra does not treat it as an override."""
    sys.argv = [a for a in sys.argv if not a.startswith("--local_rank")]
