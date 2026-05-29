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
    # Default mix is the RefCOCO family only. GRIT, grefcoco and          #
    # vg_grounding remain in the registry (data is on disk) so they can be #
    # opted into a future run via ``+stage1_extra_datasets=[grit, ...]``  #
    # or the ``stage1_datasets/all`` Hydra group.                          #
    # ------------------------------------------------------------------ #
    DatasetSpec("refcoco",      1, "refcoco",      default=True),
    DatasetSpec("refcoco+",     1, "refcoco+",     default=True),
    DatasetSpec("refcocog",     1, "refcocog",     default=True),
    DatasetSpec("grefcoco",     1, "grefcoco",     default=False),
    DatasetSpec("vg_grounding", 1, "vg_grounding", default=False),
    DatasetSpec("grit",         1, "grit",         default=False),
    # ------------------------------------------------------------------ #
    # Stage 2 — spatial VQA                                               #
    # Default mix is Visual-CoT only when present on disk.                 #
    # Per-image scene-graph supervision for the aux grounding stream lives #
    # in the gqa_spatial dataset directory. vsr / clevr_spatial /          #
    # vg_spatial / rel3d / cambrian_spatial remain registered for opt-in.  #
    # ------------------------------------------------------------------ #
    DatasetSpec("vsr",              2, "vsr",              default=False),
    DatasetSpec("gqa_spatial",      2, "gqa_spatial",      default=False),
    DatasetSpec("clevr_spatial",    2, "clevr_spatial",    default=False),
    DatasetSpec("vg_spatial",       2, "vg_spatial",       default=False),
    DatasetSpec("rel3d",            2, "rel3d",            default=False),
    DatasetSpec("cambrian_spatial", 2, "cambrian_spatial", default=False),
    DatasetSpec("visual_cot",       2, "visual_cot",       default=True),
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
