"""Resolve and log checkpoint paths used during evaluation."""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from src.backends.hf_hub_utils import resolve_pretrained_local_path
from src.config import ReInspectionConfig


def _abs_path(path: Optional[str]) -> Optional[str]:
    return os.path.abspath(path) if path else None


def find_reinspection_module_path(
    checkpoint_dir: Optional[str],
    lora_checkpoint_dir: Optional[str],
) -> Optional[str]:
    for cand_dir in (lora_checkpoint_dir, checkpoint_dir):
        if not cand_dir:
            continue
        cand = os.path.join(cand_dir, "reinspection_module.pt")
        if os.path.exists(cand):
            return cand
    return None


def find_lora_weights_path(
    checkpoint_dir: Optional[str],
    lora_checkpoint_dir: Optional[str],
    *,
    search_order: tuple[Optional[str], ...],
) -> Optional[str]:
    for cand_dir in search_order:
        if not cand_dir:
            continue
        cand = os.path.join(cand_dir, "lora_weights")
        if os.path.exists(cand):
            return cand
    return None


def lora_only_checkpoint_root(
    checkpoint_dir: Optional[str],
    lora_checkpoint_dir: Optional[str],
) -> Optional[str]:
    return lora_checkpoint_dir or checkpoint_dir


def resolve_eval_model_paths(
    condition: str,
    *,
    checkpoint_dir: Optional[str],
    lora_checkpoint_dir: Optional[str],
    model_name_or_path: str,
    frozen_cache_file: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """Resolve checkpoint paths that ``load_condition_model`` will use."""
    paths: Dict[str, Optional[str]] = {
        "base_model": resolve_pretrained_local_path(model_name_or_path),
    }

    if condition == "frozen":
        paths["reinspection_module"] = None
        paths["lora_weights"] = None
        paths["frozen_cache"] = _abs_path(frozen_cache_file)
        return paths

    if condition == "lora_only":
        ckpt_root = lora_only_checkpoint_root(checkpoint_dir, lora_checkpoint_dir)
        paths["checkpoint_root"] = _abs_path(ckpt_root)
        lora_path = os.path.join(ckpt_root, "lora_weights") if ckpt_root else None
        paths["lora_weights"] = _abs_path(lora_path)
        paths["reinspection_module"] = None
        return paths

    paths["checkpoint_dir"] = _abs_path(checkpoint_dir)
    paths["lora_checkpoint_dir"] = _abs_path(lora_checkpoint_dir)
    paths["reinspection_module"] = _abs_path(
        find_reinspection_module_path(checkpoint_dir, lora_checkpoint_dir)
    )
    paths["lora_weights"] = _abs_path(
        find_lora_weights_path(
            checkpoint_dir,
            lora_checkpoint_dir,
            search_order=(checkpoint_dir, lora_checkpoint_dir),
        )
    )
    return paths


def print_eval_model_paths(condition: str, paths: Dict[str, Optional[str]]) -> None:
    print(f"Saved model paths ({condition}):", flush=True)
    for key, value in paths.items():
        if value is None:
            continue
        missing = key != "base_model" and not os.path.exists(value)
        suffix = " [missing]" if missing else ""
        print(f"  {key}: {value}{suffix}", flush=True)


def log_eval_checkpoint_plan(
    config: ReInspectionConfig,
    conditions: List[str],
) -> None:
    print("Saved model paths (evaluation):", flush=True)
    print(f"  checkpoint_dir: {_abs_path(config.checkpoint_dir) or '(none)'}", flush=True)
    print(
        f"  lora_checkpoint_dir: {_abs_path(config.lora_checkpoint_dir) or '(none)'}",
        flush=True,
    )
    if config.frozen_cache_file:
        cache = _abs_path(config.frozen_cache_file)
        cache_note = " [cache hit]" if cache and os.path.exists(cache) else ""
        print(f"  frozen_cache_file: {cache}{cache_note}", flush=True)
    print("", flush=True)
    for condition in conditions:
        paths = resolve_eval_model_paths(
            condition,
            checkpoint_dir=config.checkpoint_dir,
            lora_checkpoint_dir=config.lora_checkpoint_dir,
            model_name_or_path=config.model_name_or_path,
            frozen_cache_file=config.frozen_cache_file,
        )
        print_eval_model_paths(condition, paths)
    print("", flush=True)
