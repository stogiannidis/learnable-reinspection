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


def maybe_track_tqdm(config: ReInspectionConfig, pbar: Any) -> Any:
    """Attach pbar.io to an existing stdlib tqdm (needed when ``disable=True`` breaks cloud hook).

    pbar_io's tqdm subclass skips remote sync when ``disable`` is True; ``track_tqdm`` patches
    updates regardless of local display, which suits eval under ``tee``.
    """
    if not getattr(config, "pbar_enabled", False):
        return pbar
    configure_pbar_io(config)
    try:
        from pbar_io import track_tqdm

        track_tqdm(pbar)
    except Exception:
        pass
    return pbar
