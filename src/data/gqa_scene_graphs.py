"""GQA scene-graph–driven referring-expression dataset for Stage-2 auxiliary grounding.

Emits samples in the same shape ``RefCOCODataset`` produces (image path, normalized
bbox, referring expression), so the Stage-2 auxiliary stream can swap it in via
``_build_stage2_aux_grounding_dataset`` without any change to the loss code.

Each sample is one (image, object) pair from the GQA scene graph. The referring
expression is built from the object's name plus a random subset of its
attributes and one of its outgoing relations, e.g.::

    the small yellow banana to the left of the bottle

The attention/ROI target is the *subject* bbox (the object the expression refers
to). The relation provides linguistic context; we deliberately do not extend
the target to the relation's object because that would mean supervising two
distinct attention peaks for one expression — a different loss formulation.
"""
import json
import logging
import os
import random
from typing import Dict, List, Optional

from .refcoco import RefCOCODataset

logger = logging.getLogger(__name__)


# Filter thresholds. Tiny objects are usually annotation noise that explodes
# the attention KL target onto a single patch; we just skip them.
_MIN_BBOX_AREA_PX = 32 * 32          # smaller than 32×32 → drop
_MIN_RELATIVE_AREA = 1.0 / (40 * 40)  # smaller than ~0.06% of the image → drop


def _build_referring_expression(
    obj: Dict,
    scene_objects: Dict[str, Dict],
    rng: random.Random,
    use_attrs_prob: float = 0.7,
    use_relation_prob: float = 0.5,
    max_attrs: int = 3,
) -> str:
    """Construct a natural-language referring expression for ``obj``.

    Pattern: ``"the [attr1 attr2 ...] <name> [<relation> the <other.name>]"``.
    Attributes and relations are sampled probabilistically so that the same
    object yields varied expressions across epochs.
    """
    parts: List[str] = []

    attrs = obj.get("attributes") or []
    if attrs and rng.random() < use_attrs_prob:
        k = rng.randint(1, min(max_attrs, len(attrs)))
        # Preserve dataset order of attributes most of the time but allow a
        # shuffle so the model doesn't latch onto positional cues.
        chosen = rng.sample(attrs, k)
        parts.extend(chosen)

    parts.append(obj["name"])

    rels = obj.get("relations") or []
    if rels and rng.random() < use_relation_prob:
        r = rng.choice(rels)
        target = scene_objects.get(str(r.get("object", "")))
        if target is not None and isinstance(r.get("name"), str):
            parts.extend([r["name"], "the", target["name"]])

    return "the " + " ".join(parts)


class GQASceneGraphGroundingDataset(RefCOCODataset):
    """Stage-2 aux grounding source built from GQA scene graphs.

    Reuses ``RefCOCODataset.__getitem__`` (tokenization, attention-target
    construction, bbox normalization) by populating ``self.samples`` in the
    same schema:

        {"image": <abs path>, "image_w": W, "image_h": H,
         "expression": <str>, "bbox": [x, y, w, h], "dataset": "gqa_scene_graphs"}
    """

    def __init__(
        self,
        scene_graphs_path: str,
        image_root: str,
        processor,
        backend: str,
        *,
        split: str = "train",
        max_pixels: int = 1280 * 28 * 28,
        min_pixels: int = 4 * 28 * 28,
        crop_to_patches: bool = False,
        system_prompt: str = "You are a helpful assistant.",
        answer_ignore_index: int = -100,
        seed: int = 0,
        use_attrs_prob: float = 0.7,
        use_relation_prob: float = 0.5,
        max_attrs: int = 3,
        min_bbox_area_px: int = _MIN_BBOX_AREA_PX,
        min_relative_area: float = _MIN_RELATIVE_AREA,
    ):
        # ---- Mirror RefCOCODataset.__init__ attribute setup, skipping its
        # file-scanning loop. ----
        from torch.utils.data import Dataset
        Dataset.__init__(self)
        if backend not in ("qwen25vl", "internvl3", "gemma4", "llava_next"):
            raise ValueError(f"unsupported backend: {backend}")
        self.processor = processor
        self.backend = backend
        self.split = split
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.crop_to_patches = crop_to_patches
        self.system_prompt = system_prompt
        self.answer_ignore_index = answer_ignore_index
        self.image_seq_length = getattr(processor, "image_seq_length", 256)

        # ---- Load the scene graph file and unfold into per-object samples. ----
        if not os.path.exists(scene_graphs_path):
            raise FileNotFoundError(scene_graphs_path)
        with open(scene_graphs_path, "r", encoding="utf-8") as handle:
            sg = json.load(handle)

        self._rng = random.Random(seed)
        self._use_attrs_prob = use_attrs_prob
        self._use_relation_prob = use_relation_prob
        self._max_attrs = max_attrs

        self.samples: List[Dict] = []
        skipped_tiny = 0
        skipped_missing_img = 0
        for image_id, scene in sg.items():
            W = int(scene.get("width", 0))
            H = int(scene.get("height", 0))
            if W <= 0 or H <= 0:
                continue
            img_area = float(W * H)
            image_path = os.path.join(image_root, f"{image_id}.jpg")
            # Defer the image-presence check to avoid 75k stat calls at init;
            # __getitem__'s open() will surface the missing file if needed.
            objects = scene.get("objects") or {}
            if not objects:
                continue
            for obj_id, obj in objects.items():
                name = obj.get("name")
                if not name:
                    continue
                try:
                    x = float(obj["x"])
                    y = float(obj["y"])
                    w = float(obj["w"])
                    h = float(obj["h"])
                except (KeyError, TypeError, ValueError):
                    continue
                if w <= 0 or h <= 0:
                    continue
                area = w * h
                if area < min_bbox_area_px or area / img_area < min_relative_area:
                    skipped_tiny += 1
                    continue
                # Build the referring expression once at init for reproducibility;
                # downstream caching makes per-epoch variation less important and
                # we avoid paying the sampling cost in every __getitem__.
                expression = _build_referring_expression(
                    obj,
                    scene_objects=objects,
                    rng=self._rng,
                    use_attrs_prob=use_attrs_prob,
                    use_relation_prob=use_relation_prob,
                    max_attrs=max_attrs,
                )
                self.samples.append(
                    {
                        "image": image_path,
                        "image_w": W,
                        "image_h": H,
                        "expression": expression,
                        "bbox": [x, y, w, h],
                        "dataset": "gqa_scene_graphs",
                    }
                )

        logger.info(
            "GQASceneGraphGroundingDataset: %d samples from %d scenes (skipped %d tiny boxes).",
            len(self.samples),
            len(sg),
            skipped_tiny,
        )
