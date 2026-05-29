"""Optional pbar.io sync for tqdm (https://pbar.io/docs/integrations)."""

from __future__ import annotations

import os
from typing import Any, TYPE_CHECKING, Type

if TYPE_CHECKING:
    from src.config import ReInspectionConfig

_configured = False


def configure_pbar_io(config: ReInspectionConfig) -> None:
    """Apply ``pbar_io.configure`` once when cloud sync is enabled."""
    global _configured
    if not getattr(config, "pbar_enabled", False) or _configured:
        return
    try:
        import pbar_io

        pbar_io.configure(
            api_url=config.pbar_api_url,
            api_key=os.environ.get("PBAR_API_KEY"),
            batch_updates=True,
            update_interval=float(config.pbar_update_interval),
        )
        _configured = True
    except Exception:
        # Tracking must never break training/eval; ignore setup failures.
        pass


def get_tqdm(config: ReInspectionConfig, *, cloud: bool) -> Type[Any]:
    """Return tqdm class: pbar_io drop-in when ``cloud`` and config allow, else stdlib tqdm.

    Use ``cloud=True`` only on rank 0 (or single-process jobs) to avoid duplicate remote bars.
    """
    if cloud and getattr(config, "pbar_enabled", False):
        configure_pbar_io(config)
        try:
            from pbar_io import tqdm as pbar_tqdm  # type: ignore[attr-defined]

            return pbar_tqdm
        except Exception:
            pass
    from tqdm import tqdm as std_tqdm

    return std_tqdm


_devnull = None


def _devnull_file():
    """Lazily open a process-lifetime ``os.devnull`` sink for suppressed bar output."""
    global _devnull
    if _devnull is None:
        _devnull = open(os.devnull, "w")
    return _devnull


def make_eval_tqdm(
    config: ReInspectionConfig,
    iterable,
    *,
    desc: str,
    disable: bool,
    **kwargs,
) -> Any:
    """Create an eval progress bar with optional pbar.io cloud sync.

    pbar_io's tqdm subclass only syncs to the cloud while the bar is *enabled* — a
    ``disable=True`` bar silently drops every remote update, and tqdm's disabled
    ``__iter__`` never calls ``update()`` anyway. So when cloud sync is on we keep the
    bar enabled even when local display is suppressed (e.g. under ``tee``): the local
    rendering is routed to ``os.devnull`` (no per-refresh log spam) while the remote
    bar still advances. We also override the caller's coarse ``miniters`` for the cloud
    path so the remote bar updates on the time-based ``mininterval`` instead of jumping
    in large steps; pbar_io batches the actual network pushes via ``update_interval``.
    """
    if getattr(config, "pbar_enabled", False):
        tqdm_cls = get_tqdm(config, cloud=True)
        if disable:
            kwargs.setdefault("file", _devnull_file())
            kwargs["miniters"] = 1
        return tqdm_cls(iterable, desc=desc, disable=False, **kwargs)

    from tqdm import tqdm as std_tqdm

    return std_tqdm(iterable, desc=desc, disable=disable, **kwargs)
