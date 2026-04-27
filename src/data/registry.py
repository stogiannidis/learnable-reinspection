"""Single source of truth for dataset names, stages, paths, and defaults.

All other modules (spatial_dataset, refcoco, trainer, config) import from here
instead of maintaining their own inline lists.
"""

import os
from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class DatasetSpec:
    name: str    # canonical key used everywhere
    stage: int   # 1 = grounding, 2 = spatial VQA
    subdir: str  # relative path under data_root
    default: bool  # included in default training mix


REGISTRY: List[DatasetSpec] = [
    # ------------------------------------------------------------------ #
    # Stage 1 — referring-expression grounding                            #
    # ------------------------------------------------------------------ #
    DatasetSpec("refcoco",      1, "refcoco",      default=True),
    DatasetSpec("refcoco+",     1, "refcoco+",     default=True),
    DatasetSpec("refcocog",     1, "refcocog",     default=True),
    DatasetSpec("grefcoco",     1, "grefcoco",     default=True),
    DatasetSpec("vg_grounding", 1, "vg_grounding", default=True),
    DatasetSpec("grit",         1, "grit",         default=True),
    # ------------------------------------------------------------------ #
    # Stage 2 — spatial VQA                                               #
    # ------------------------------------------------------------------ #
    DatasetSpec("vsr",              2, "vsr",              default=True),
    DatasetSpec("gqa_spatial",      2, "gqa_spatial",      default=True),
    DatasetSpec("clevr_spatial",    2, "clevr_spatial",    default=True),
    DatasetSpec("vg_spatial",       2, "vg_spatial",       default=True),
    # default=False until annotations land on disk
    DatasetSpec("rel3d",            2, "rel3d",            default=False),
    DatasetSpec("cambrian_spatial", 2, "cambrian_spatial", default=False),
]


def stage1_defaults() -> List[str]:
    """Names of all Stage-1 datasets included in the default training mix."""
    return [d.name for d in REGISTRY if d.stage == 1 and d.default]


def stage2_defaults() -> List[str]:
    """Names of all Stage-2 datasets included in the default training mix."""
    return [d.name for d in REGISTRY if d.stage == 2 and d.default]


def all_for_stage(stage: int) -> List[str]:
    """All registered dataset names for a given stage (default and non-default)."""
    return [d.name for d in REGISTRY if d.stage == stage]


def dataset_path_configs(data_root: str, split: str) -> Dict[str, Dict]:
    """Per-dataset path dicts keyed by name, compatible with both loaders.

    Each value has ``data_file`` (split JSON) and ``image_root`` (images dir).
    """
    return {
        d.name: {
            "data_file": os.path.join(data_root, d.subdir, f"{split}.json"),
            "image_root": os.path.join(data_root, d.subdir, "images"),
        }
        for d in REGISTRY
    }
