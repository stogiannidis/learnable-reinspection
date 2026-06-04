"""Distributed DeepSpeed training for re-inspection VLMs.

Builds per-backend frozen-base + trainable-module setups, attaches optional
stage-1 supervision (attention KL, bounding-box head), runs the
optimizer loop with cosine warmup scheduling, checkpointing, and optional
Weights & Biases logging.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from typing import Iterable, Optional, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler, Subset
from tqdm import tqdm as tqdm_stdlib
from transformers import AutoProcessor

from src.utils.progress import get_tqdm

from src.backends.hf_hub_utils import resolve_pretrained_local_path
from src.model.attn_loss import compute_attn_loss_kl
from src.model.bbox_head import BboxHead, compute_grounding_loss
from src.model.query_text_infonce import compute_query_text_infonce_loss
from src.model.roi_feature_loss import ROIProjection, compute_roi_feature_loss
from src.config import ReInspectionConfig
from src.data.refcoco import RefCOCODataset
from src.data.spatial_dataset import build_spatial_dataset
from src.data.registry import REGISTRY, stage1_defaults

_CONCAT_KEYS = {"pixel_values", "image_grid_thw", "video_grid_thw", "image_sizes"}
_VARLEN_FLOAT_PAD_KEYS = {"attn_target_mask", "bbox_norm"}
_VARLEN_PAD_KEYS = _VARLEN_FLOAT_PAD_KEYS

# Log once if stage-1 attention targets are resized to match attn_vis width.
_attn_target_vis_mismatch_logged = False


def collate_fn(batch):
    """Merge a list of dataset dicts into a batched tensor dictionary.

    Concatenates image tensors that are listed in ``_CONCAT_KEYS``, pads
    variable-length float tensors (attention targets, boxes), stacks fixed-shape
    tensors, and uses ``-100`` padding for ``labels`` when sequence lengths differ.

    Args:
        batch: List of per-sample dicts from :func:`torch.utils.data.DataLoader`.

    Returns:
        Dict mapping each key to a batched tensor or list value.
    """
    keys = batch[0].keys()
    collated = {}
    for key in keys:
        values = [item[key] for item in batch]
        if not isinstance(values[0], torch.Tensor):
            collated[key] = values
            continue
        if key in _CONCAT_KEYS:
            if key == "pixel_values" and values[0].dim() == 5:
                # LLaVA-Next AnyRes emits per-sample (1, num_patches_i, C, H, W)
                # with a variable patch count. Pad the patch dim to the batch
                # max, then concat along the batch dim -> (B, max_patches, C,
                # H, W). The model re-derives the true patch count from
                # ``image_sizes`` and slices the zero padding back off.
                max_patches = max(v.shape[1] for v in values)
                padded = []
                for v in values:
                    if v.shape[1] < max_patches:
                        pad = v.new_zeros(
                            (v.shape[0], max_patches - v.shape[1], *v.shape[2:])
                        )
                        v = torch.cat([v, pad], dim=1)
                    padded.append(v)
                collated[key] = torch.cat(padded, dim=0)
            else:
                collated[key] = torch.cat(values, dim=0)
        elif key in _VARLEN_FLOAT_PAD_KEYS:
            collated[key] = pad_sequence(values, batch_first=True, padding_value=0.0)
        elif values[0].ndim == 0:
            collated[key] = torch.stack(values)
        elif all(value.shape == values[0].shape for value in values):
            collated[key] = torch.stack(values)
        else:
            pad_value = -100 if key == "labels" else 0
            collated[key] = pad_sequence(values, batch_first=True, padding_value=pad_value)
    return collated



def is_main_process() -> bool:
    """True on rank 0 when distributed is initialized; True if single-process."""
    return not dist.is_initialized() or dist.get_rank() == 0


def log(message: str) -> None:
    """Print ``message`` only on the main process (rank 0)."""
    if is_main_process():
        print(message, flush=True)


def _setup_distributed() -> None:
    """Initialize ``torch.distributed`` NCCL when ``RANK`` is set (DeepSpeed launch)."""
    if dist.is_initialized():
        return
    if "RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device("cuda", local_rank),
        )


def _get_local_rank() -> int:
    """Local GPU index from the launcher environment (defaults to 0)."""
    return int(os.environ.get("LOCAL_RANK", 0))


def _git_info() -> dict:
    """Collect git revision and diff stats for reproducibility."""
    info: dict = {}
    try:
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
        info["dirty"] = bool(dirty)
        info["diff_stat"] = subprocess.check_output(
            ["git", "diff", "--stat"], stderr=subprocess.DEVNULL
        ).decode().strip()[:500]
    except Exception:
        info["commit"] = "unknown"
    return info


def _env_info() -> dict:
    """Collect environment details for reproducibility."""
    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "N/A",
        "cudnn_version": str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "N/A",
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
    try:
        import transformers
        info["transformers_version"] = transformers.__version__
    except ImportError:
        pass
    try:
        import peft as _peft
        info["peft_version"] = _peft.__version__
    except (ImportError, AttributeError):
        pass
    try:
        import deepspeed as _ds
        info["deepspeed_version"] = _ds.__version__
    except (ImportError, AttributeError):
        pass
    return info


def _file_hash(path: str, algo: str = "sha256") -> str:
    """Streaming file digest for dataset reproducibility artifacts.

    Args:
        path: Readable file path.
        algo: Hash algorithm name accepted by :func:`hashlib.new`.

    Returns:
        Lowercase hex digest string.
    """
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _log_dataset_artifact(
    dataset, stage: int, backend: str, data_root: str
) -> None:
    """Log dataset metadata and annotation file checksums as a W&B artifact."""
    try:
        import wandb

        if wandb.run is None:
            return

        meta = {
            "data_root": data_root,
            "num_samples": len(dataset),
            "stage": stage,
            "backend": backend,
        }

        art = wandb.Artifact(
            f"dataset-stage{stage}-{backend}",
            type="dataset",
            metadata=meta,
        )

        ann_files = set()
        if hasattr(dataset, "samples") and dataset.samples:
            sample_keys = list(dataset.samples[0].keys()) if dataset.samples else []
            meta["sample_keys"] = sample_keys
            meta["first_sample"] = {
                k: str(v)[:200] for k, v in dataset.samples[0].items()
            }

        if hasattr(dataset, "datasets"):
            for sub in dataset.datasets:
                if hasattr(sub, "samples") and sub.samples:
                    meta[f"subdataset_{type(sub).__name__}_size"] = len(sub.samples)

        for ds in REGISTRY:
            for split in ["train", "val", "test"]:
                for ext in ["json", "jsonl"]:
                    p = os.path.join(data_root, ds.subdir, f"{split}.{ext}")
                    if os.path.exists(p):
                        ann_files.add(p)

        checksums = {}
        for p in sorted(ann_files):
            try:
                checksums[os.path.relpath(p, data_root)] = _file_hash(p)
                art.add_file(p, name=os.path.relpath(p, data_root))
            except Exception:
                pass

        meta["annotation_checksums"] = checksums
        art.metadata = meta
        wandb.log_artifact(art)
    except Exception:
        pass


def _log_code_artifact() -> None:
    """Log the source tree as a W&B artifact for exact code reproducibility."""
    try:
        import wandb

        if wandb.run is None:
            return

        art = wandb.Artifact("source-code", type="code", metadata=_git_info())
        code_dir = os.path.join(os.path.dirname(__file__), "..")
        if os.path.isdir(code_dir):
            art.add_dir(code_dir, name="src")
        wandb.log_artifact(art)
    except Exception:
        pass


def _log_config_artifact(config: ReInspectionConfig) -> None:
    """Log the resolved config as a JSON artifact."""
    try:
        import wandb

        if wandb.run is None:
            return

        config_path = os.path.join(wandb.run.dir, "resolved_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(asdict(config), f, indent=2, default=str)
        wandb.save(config_path, policy="now")
    except Exception:
        pass


def _init_wandb(config: ReInspectionConfig) -> None:
    """Start a W&B run on rank 0 with git/env metadata and config/code artifacts."""
    if not is_main_process():
        return
    try:
        import wandb

        os.environ["WANDB_SILENT"] = "true"
        resolved = asdict(config)
        resolved.update({
            "_git": _git_info(),
            "_env": _env_info(),
        })
        name = config.wandb_run_name
        if name:
            name = f"{name}_stage{config.stage}"

        tags = [config.backend, f"stage{config.stage}"]
        wandb.init(
            project=config.wandb_project,
            name=name,
            config=resolved,
            tags=tags,
            save_code=True,
        )

        _log_config_artifact(config)
        _log_code_artifact()
    except Exception:
        pass


def _log_wandb(metrics: dict, step: int) -> None:
    """Log a metrics dict to the active W&B run (rank 0 only; silently no-op if absent)."""
    if not is_main_process():
        return
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except Exception:
        pass


def _log_model_artifact(
    save_dir: str, stage: int, backend: str, epoch: int, is_best: bool = False
) -> None:
    """Log a model checkpoint as a W&B artifact."""
    if not is_main_process():
        return
    try:
        import wandb

        if wandb.run is None:
            return

        name = f"model-stage{stage}-{backend}"
        aliases = [f"epoch-{epoch}", "latest"]
        if is_best:
            aliases.append("best")

        art = wandb.Artifact(
            name, type="model",
            metadata={"stage": stage, "backend": backend, "epoch": epoch, "save_dir": save_dir},
        )
        if os.path.isdir(save_dir):
            art.add_dir(save_dir)
        wandb.log_artifact(art, aliases=aliases)
    except Exception:
        pass


def _finish_wandb() -> None:
    if not is_main_process():
        return
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass


def _load_stage1_weights(model, checkpoint_path: str) -> None:
    state_path = checkpoint_path
    if os.path.isdir(state_path):
        state_path = os.path.join(state_path, "reinspection_module.pt")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"Stage 1 checkpoint not found: {state_path}")
    state_dict = torch.load(state_path, map_location="cpu", weights_only=True)
    model.reinspection.load_state_dict(state_dict)


def _stage1_checkpoint_dir(checkpoint_path: Optional[str]) -> Optional[str]:
    if not checkpoint_path:
        return None
    return checkpoint_path if os.path.isdir(checkpoint_path) else os.path.dirname(checkpoint_path)


def _attach_bbox_head(model, config: ReInspectionConfig) -> None:
    """Create and attach a BboxHead for Stage 1 grounding supervision."""
    model.bbox_head = BboxHead(config.d_bottleneck, dtype=torch.float32)


def _attach_roi_projection(model, config: ReInspectionConfig) -> None:
    """Create and attach the ROI feature projection for Stage 1 grounding."""
    model.roi_proj = ROIProjection(config.d_bottleneck, config.d_model, dtype=torch.float32)


def _stage2_aux_grounding_enabled(config: ReInspectionConfig) -> bool:
    """Whether Stage 2 should run the optional ROI-labeled grounding stream."""
    return config.stage == 2 and config.stage2_aux_grounding_weight > 0.0 and config.stage2_aux_every_n_steps > 0


def _attach_stage2_aux_modules(model, config: ReInspectionConfig, stage1_checkpoint: Optional[str]) -> None:
    """Attach fixed Stage-1 sidecar modules needed by Stage-2 auxiliary losses."""
    if not _stage2_aux_grounding_enabled(config) or config.stage2_aux_roi_feature_loss_weight <= 0.0:
        return

    checkpoint_dir = _stage1_checkpoint_dir(stage1_checkpoint)
    roi_path = os.path.join(checkpoint_dir, "roi_proj.pt") if checkpoint_dir else None
    if roi_path and os.path.exists(roi_path):
        _attach_roi_projection(model, config)
        state_dict = torch.load(roi_path, map_location="cpu", weights_only=True)
        model.roi_proj.load_state_dict(state_dict)
        for parameter in model.roi_proj.parameters():
            parameter.requires_grad = False
        log(f"Loaded Stage 1 ROI projection for Stage 2 auxiliary loss: {roi_path}")
    else:
        log(
            "[WARNING] Stage 2 ROI auxiliary loss requested but roi_proj.pt was not "
            "found next to the Stage 1 checkpoint; ROI auxiliary term will be zero."
        )


def _stage2_optimizer(config: ReInspectionConfig, model, lora_params):
    """Build Stage-2 AdamW groups with configurable re-inspection training."""
    groups = []
    reinspection_params = [p for p in model.reinspection.parameters() if p.requires_grad]
    if config.stage2_train_reinspection and reinspection_params:
        groups.append(
            {
                "params": reinspection_params,
                "lr": config.stage2_lr_module,
                "name": "reinspection",
            }
        )
    groups.append({"params": lora_params, "lr": config.stage2_lr_lora, "name": "lora"})
    return torch.optim.AdamW(groups, weight_decay=0.01)


def _setup_model_intern_stage1(config: ReInspectionConfig, processor):
    from src.backends.internvl3 import load_model

    model = load_model(config, device_map=None, processor=processor)
    for parameter in model.base_model.parameters():
        parameter.requires_grad = False
    for parameter in model.reinspection.parameters():
        parameter.requires_grad = True
    optim_groups = [
        {
            "params": list(model.reinspection.parameters()),
            "lr": config.stage1_lr_module,
            "name": "reinspection",
        }
    ]
    if config.stage1_use_grounding_loss:
        _attach_bbox_head(model, config)
        optim_groups.append(
            {
                "params": list(model.bbox_head.parameters()),
                "lr": config.stage1_lr_module,
                "name": "bbox_head",
            }
        )
    if config.stage1_use_roi_feature_loss:
        _attach_roi_projection(model, config)
        optim_groups.append(
            {
                "params": list(model.roi_proj.parameters()),
                "lr": config.stage1_lr_module,
                "name": "roi_proj",
            }
        )
    if config.stage1_train_projector:
        for parameter in model.base_model.model.multi_modal_projector.parameters():
            parameter.requires_grad = True
        optim_groups.append(
            {
                "params": list(model.base_model.model.multi_modal_projector.parameters()),
                "lr": config.stage1_lr_projector,
                "name": "projector",
            }
        )
    optimizer = torch.optim.AdamW(optim_groups, weight_decay=0.01)
    return model, optimizer


def _setup_model_intern_stage2(config: ReInspectionConfig, processor, stage1_checkpoint: Optional[str]):
    from src.backends.internvl3 import load_model

    model = load_model(config, device_map=None, processor=processor)
    if stage1_checkpoint:
        _load_stage1_weights(model, stage1_checkpoint)
    for parameter in model.base_model.parameters():
        parameter.requires_grad = False
    for parameter in model.reinspection.parameters():
        parameter.requires_grad = config.stage2_train_reinspection
    _attach_stage2_aux_modules(model, config, stage1_checkpoint)
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    model.base_model.model.language_model = get_peft_model(
        model.base_model.model.language_model,
        lora_config,
    )
    lora_params = [
        param for _, param in model.base_model.model.language_model.named_parameters() if param.requires_grad
    ]
    optimizer = _stage2_optimizer(config, model, lora_params)
    return model, optimizer


# ---- Qwen2.5-VL setup ----

def _setup_model_qwen25_stage1(config: ReInspectionConfig):
    from src.backends.qwen25vl import load_model

    model = load_model(config, device_map=None)
    for param in model.base_model.parameters():
        param.requires_grad = False
    for param in model.reinspection.parameters():
        param.requires_grad = True
    optim_groups = [
        {
            "params": list(model.reinspection.parameters()),
            "lr": config.stage1_lr_module,
            "name": "reinspection",
        }
    ]
    if config.stage1_use_grounding_loss:
        _attach_bbox_head(model, config)
        optim_groups.append(
            {
                "params": list(model.bbox_head.parameters()),
                "lr": config.stage1_lr_module,
                "name": "bbox_head",
            }
        )
    if config.stage1_use_roi_feature_loss:
        _attach_roi_projection(model, config)
        optim_groups.append(
            {
                "params": list(model.roi_proj.parameters()),
                "lr": config.stage1_lr_module,
                "name": "roi_proj",
            }
        )
    optimizer = torch.optim.AdamW(optim_groups, weight_decay=0.01)
    return model, optimizer


def _setup_model_qwen25_stage2(config: ReInspectionConfig):
    from src.backends.qwen25vl import load_model

    model = load_model(config, device_map=None)
    if config.stage1_checkpoint:
        log(f"Loading Stage 1 checkpoint: {config.stage1_checkpoint}")
        _load_stage1_weights(model, config.stage1_checkpoint)
    for param in model.base_model.parameters():
        param.requires_grad = False
    for param in model.reinspection.parameters():
        param.requires_grad = config.stage2_train_reinspection
    _attach_stage2_aux_modules(model, config, config.stage1_checkpoint)
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    model.base_model.model.language_model = get_peft_model(
        model.base_model.model.language_model, lora_config
    )
    lora_params = [p for _, p in model.base_model.named_parameters() if p.requires_grad]
    optimizer = _stage2_optimizer(config, model, lora_params)
    return model, optimizer


# ---- Gemma4 setup ----

def _setup_model_gemma4_stage1(config: ReInspectionConfig, processor):
    from src.backends.gemma4 import load_model

    model = load_model(config, device_map=None, processor=processor)
    for parameter in model.base_model.parameters():
        parameter.requires_grad = False
    for parameter in model.reinspection.parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.AdamW(
        [{"params": list(model.reinspection.parameters()), "lr": config.stage1_lr_module, "name": "reinspection"}],
        lr=config.stage1_lr_module,
        weight_decay=0.01,
    )
    return model, optimizer


def _setup_model_gemma4_stage2(config: ReInspectionConfig, processor, stage1_checkpoint: Optional[str]):
    from src.backends.gemma4 import load_model

    model = load_model(config, device_map=None, processor=processor)
    if stage1_checkpoint:
        _load_stage1_weights(model, stage1_checkpoint)
    for parameter in model.base_model.parameters():
        parameter.requires_grad = False
    for parameter in model.reinspection.parameters():
        parameter.requires_grad = config.stage2_train_reinspection
    _attach_stage2_aux_modules(model, config, stage1_checkpoint)
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    model.base_model.model.language_model = get_peft_model(
        model.base_model.model.language_model,
        lora_config,
    )
    lora_params = [
        param for _, param in model.base_model.model.language_model.named_parameters() if param.requires_grad
    ]
    optimizer = _stage2_optimizer(config, model, lora_params)
    return model, optimizer


# ---- LLaVA-Next (Mistral-7B) setup ----

def _setup_model_llava_stage1(config: ReInspectionConfig, processor):
    from src.backends.llava_next import load_model

    model = load_model(config, device_map=None, processor=processor)
    for parameter in model.base_model.parameters():
        parameter.requires_grad = False
    for parameter in model.reinspection.parameters():
        parameter.requires_grad = True
    optim_groups = [
        {
            "params": list(model.reinspection.parameters()),
            "lr": config.stage1_lr_module,
            "name": "reinspection",
        }
    ]
    # bbox_head and roi_proj are only attached when their loss flags are set,
    # mirroring the InternVL3 stage-1 setup.
    if config.stage1_use_grounding_loss:
        _attach_bbox_head(model, config)
        optim_groups.append(
            {
                "params": list(model.bbox_head.parameters()),
                "lr": config.stage1_lr_module,
                "name": "bbox_head",
            }
        )
    if config.stage1_use_roi_feature_loss:
        _attach_roi_projection(model, config)
        optim_groups.append(
            {
                "params": list(model.roi_proj.parameters()),
                "lr": config.stage1_lr_module,
                "name": "roi_proj",
            }
        )
    optimizer = torch.optim.AdamW(optim_groups, weight_decay=0.01)
    return model, optimizer


def _setup_model_llava_stage2(config: ReInspectionConfig, processor, stage1_checkpoint: Optional[str]):
    from src.backends.llava_next import load_model

    model = load_model(config, device_map=None, processor=processor)
    if stage1_checkpoint:
        _load_stage1_weights(model, stage1_checkpoint)
    for parameter in model.base_model.parameters():
        parameter.requires_grad = False
    for parameter in model.reinspection.parameters():
        parameter.requires_grad = config.stage2_train_reinspection
    _attach_stage2_aux_modules(model, config, stage1_checkpoint)
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    model.base_model.model.language_model = get_peft_model(
        model.base_model.model.language_model,
        lora_config,
    )
    lora_params = [
        param for _, param in model.base_model.model.language_model.named_parameters() if param.requires_grad
    ]
    optimizer = _stage2_optimizer(config, model, lora_params)
    return model, optimizer


def _build_dataset(
    backend: str,
    stage: int,
    config: ReInspectionConfig,
    data_root: str,
    processor,
):
    if stage == 1:
        stage1_names = list(config.stage1_dataset_names or stage1_defaults())
        if config.stage1_extra_datasets:
            for name in config.stage1_extra_datasets:
                if name not in stage1_names:
                    stage1_names.append(name)
        return RefCOCODataset(
            data_root=data_root,
            processor=processor,
            backend=backend,
            split="train",
            dataset_names=stage1_names,
            max_pixels=config.max_pixels,
            min_pixels=config.min_pixels,
            crop_to_patches=config.crop_to_patches_stage1,
            system_prompt=config.system_prompt,
            answer_ignore_index=config.answer_ignore_index,
            coco_images_dir=config.coco_images_dir,
        )
    return build_spatial_dataset(
        data_root=data_root,
        processor=processor,
        backend=backend,
        split="train",
        datasets=config.stage2_dataset_names,
        max_pixels=config.max_pixels,
        min_pixels=config.min_pixels,
        crop_to_patches=config.crop_to_patches_stage2,
        system_prompt=config.system_prompt,
        answer_ignore_index=config.answer_ignore_index,
    )


def _build_stage2_aux_grounding_dataset(
    backend: str,
    config: ReInspectionConfig,
    data_root: str,
    processor,
):
    """Build the optional Stage-2 grounding stream.

    Default source is the RefCOCO Stage-1 family; setting
    ``stage2_aux_use_gqa_scene_graphs=True`` swaps in GQA scene graphs
    (per-object referring expressions over the gqa_spatial images), keeping
    the rest of the aux machinery (KL attn loss + ROI loss) unchanged.
    """
    if config.stage2_aux_use_gqa_scene_graphs:
        from src.data.gqa_scene_graphs import GQASceneGraphGroundingDataset

        log(
            "Stage-2 aux grounding source: GQA scene graphs "
            f"({config.stage2_aux_gqa_scene_graphs_path})"
        )
        return GQASceneGraphGroundingDataset(
            scene_graphs_path=config.stage2_aux_gqa_scene_graphs_path,
            image_root=config.stage2_aux_gqa_images_dir,
            processor=processor,
            backend=backend,
            split="train",
            max_pixels=config.max_pixels,
            min_pixels=config.min_pixels,
            crop_to_patches=config.crop_to_patches_stage1,
            system_prompt=config.system_prompt,
            answer_ignore_index=config.answer_ignore_index,
            seed=config.seed,
        )

    aux_names = list(config.stage2_aux_stage1_dataset_names or stage1_defaults())
    if config.stage2_aux_stage1_extra_datasets:
        for name in config.stage2_aux_stage1_extra_datasets:
            if name not in aux_names:
                aux_names.append(name)
    return RefCOCODataset(
        data_root=data_root,
        processor=processor,
        backend=backend,
        split="train",
        dataset_names=aux_names,
        max_pixels=config.max_pixels,
        min_pixels=config.min_pixels,
        crop_to_patches=config.crop_to_patches_stage1,
        system_prompt=config.system_prompt,
        answer_ignore_index=config.answer_ignore_index,
        coco_images_dir=config.coco_images_dir,
    )


def _split_train_val(dataset, val_ratio: float, seed: int):
    """Deterministic train/val holdout. Returns (train_subset, val_subset_or_None)."""
    n = len(dataset)
    n_val = int(round(n * val_ratio))
    if n_val == 0 or n_val >= n:
        return dataset, None
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def _cycle_loader(loader):
    """Yield batches from a dataloader forever, restarting at epoch boundaries."""
    while True:
        for batch in loader:
            yield batch


def _run_validation(
    ds_engine,
    val_loader,
    backend: str,
    is_stage1: bool,
    use_attn: bool,
    use_grounding: bool,
    use_qt_infonce: bool,
    use_roi_feat: bool,
    use_lm_ce: bool,
    config: ReInspectionConfig,
    device,
) -> dict:
    """Run one validation epoch. All ranks must participate (collective all_reduce).

    Returns averaged loss components as a dict of floats. Non-finite batches are
    skipped (all_reduced MIN guard) and excluded from the average.
    """
    was_training = ds_engine.training
    ds_engine.eval()

    # Accumulators on device for cheap all_reduce.
    ce_sum = torch.zeros((), device=device)
    attn_sum = torch.zeros((), device=device)
    ground_sum = torch.zeros((), device=device)
    qt_sum = torch.zeros((), device=device)
    roi_sum = torch.zeros((), device=device)
    total_sum = torch.zeros((), device=device)
    n_batches = torch.zeros((), device=device)

    try:
        with torch.inference_mode():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = ds_engine(
                    **_train_forward_kwargs(
                        batch, backend, is_stage1,
                        return_query_text_tensors=use_qt_infonce,
                        use_lm_ce=use_lm_ce,
                    )
                )

                ce_loss = outputs.loss
                if ce_loss is None:
                    ce_loss = torch.zeros((), device=device)

                attn_loss = torch.zeros((), device=device)
                grounding_loss = torch.zeros((), device=device)
                qt_loss = torch.zeros((), device=device)
                roi_loss = torch.zeros((), device=device)
                if use_attn:
                    attn_loss = _stage1_attn_loss(config, outputs, batch, device)
                if use_grounding:
                    grounding_loss = _stage1_grounding_loss(
                        config, ds_engine, outputs, batch, device, global_step=10**9
                    )
                if use_qt_infonce:
                    qt_loss = _stage1_query_text_infonce_loss(config, outputs, device)
                if use_roi_feat:
                    roi_loss = _stage1_roi_feature_loss(config, ds_engine, outputs, batch, device)

                loss = (ce_loss if use_lm_ce else torch.zeros((), device=device)) \
                    + (config.stage1_attn_loss_weight * attn_loss if use_attn else 0.0) \
                    + (config.stage1_grounding_loss_weight * grounding_loss if use_grounding else 0.0) \
                    + (config.stage1_roi_feature_loss_weight * roi_loss if use_roi_feat else 0.0) \
                    + (
                        config.stage1_query_text_infonce_weight * qt_loss
                        if use_qt_infonce
                        else 0.0
                    )

                _finite = torch.tensor(float(torch.isfinite(loss)), device=device)
                if dist.is_initialized():
                    dist.all_reduce(_finite, op=dist.ReduceOp.MIN)
                if _finite.item() < 0.5:
                    continue

                ce_sum = ce_sum + ce_loss.detach()
                attn_sum = attn_sum + attn_loss.detach()
                ground_sum = ground_sum + grounding_loss.detach()
                qt_sum = qt_sum + qt_loss.detach()
                roi_sum = roi_sum + roi_loss.detach()
                total_sum = total_sum + loss.detach()
                n_batches = n_batches + 1
    finally:
        if was_training:
            ds_engine.train()
        _drain_zero3_prefetches(ds_engine)

    packed = torch.stack([ce_sum, attn_sum, ground_sum, qt_sum, roi_sum, total_sum, n_batches])
    if dist.is_initialized():
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    ce_s, attn_s, ground_s, qt_s, roi_s, total_s, n = packed.tolist()
    denom = max(n, 1.0)
    return {
        "loss": total_s / denom,
        "ce_loss": ce_s / denom,
        "attn_loss": attn_s / denom,
        "grounding_loss": ground_s / denom,
        "query_text_infonce_loss": qt_s / denom,
        "roi_feature_loss": roi_s / denom,
        "num_batches": int(n),
    }


def _drain_zero3_prefetches(ds_engine) -> None:
    """Drain any in-flight ZeRO-3 async all-gathers before opening
    GatheredParameters. Without a backward pass (e.g. after eval), prefetched
    params can stay INFLIGHT and trip an assert in partition_parameters.py."""
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
    for fn_name in ("empty_partition_cache",):
        fn = getattr(ds_engine, fn_name, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass
            break
        opt = getattr(ds_engine, "optimizer", None)
        fn = getattr(opt, fn_name, None) if opt is not None else None
        if callable(fn):
            try:
                fn()
            except Exception:
                pass
            break


def _save_checkpoint(ds_engine, save_dir: str, save_lora: bool = False) -> None:
    import deepspeed

    _drain_zero3_prefetches(ds_engine)

    if is_main_process():
        os.makedirs(save_dir, exist_ok=True)

    unwrapped = ds_engine.module

    reinsp_params = list(unwrapped.reinspection.parameters())
    with deepspeed.zero.GatheredParameters(reinsp_params, modifier_rank=0):
        if is_main_process():
            torch.save(
                unwrapped.reinspection.state_dict(),
                os.path.join(save_dir, "reinspection_module.pt"),
            )
    if hasattr(unwrapped, "bbox_head"):
        bbox_params = list(unwrapped.bbox_head.parameters())
        with deepspeed.zero.GatheredParameters(bbox_params, modifier_rank=0):
            if is_main_process():
                torch.save(
                    unwrapped.bbox_head.state_dict(),
                    os.path.join(save_dir, "bbox_head.pt"),
                )
    if hasattr(unwrapped, "roi_proj"):
        roi_params = list(unwrapped.roi_proj.parameters())
        with deepspeed.zero.GatheredParameters(roi_params, modifier_rank=0):
            if is_main_process():
                torch.save(
                    unwrapped.roi_proj.state_dict(),
                    os.path.join(save_dir, "roi_proj.pt"),
                )

    if save_lora:
        lora_params = [
            p
            for p in unwrapped.base_model.model.language_model.parameters()
            if p.requires_grad
        ]
        with deepspeed.zero.GatheredParameters(lora_params, modifier_rank=0):
            if is_main_process():
                unwrapped.base_model.model.language_model.save_pretrained(
                    os.path.join(save_dir, "lora_weights")
                )

    log(f"Saved checkpoint to {save_dir}")


def _resolve_deepspeed_auto_batch(
    ds_config: dict, micro_batch_per_gpu: int, grad_accum_steps: int
) -> dict:
    """Replace string 'auto' batch fields; required when not using HF Trainer + TrainingArguments."""
    cfg = copy.deepcopy(ds_config)
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if cfg.get("train_micro_batch_size_per_gpu") == "auto":
        cfg["train_micro_batch_size_per_gpu"] = micro_batch_per_gpu
    if cfg.get("gradient_accumulation_steps") == "auto":
        cfg["gradient_accumulation_steps"] = grad_accum_steps
    if cfg.get("train_batch_size") == "auto":
        cfg["train_batch_size"] = (
            micro_batch_per_gpu * grad_accum_steps * world_size
        )
    return cfg


def _init_deepspeed(
    model,
    optimizer,
    deepspeed_config: Union[str, dict],
    micro_batch_per_gpu: int,
    grad_accum_steps: int,
):
    import deepspeed

    if not deepspeed_config:
        raise ValueError("--deepspeed config is required")

    if isinstance(deepspeed_config, str):
        with open(deepspeed_config, encoding="utf-8") as f:
            ds_dict = json.load(f)
    else:
        ds_dict = copy.deepcopy(deepspeed_config)
    ds_dict = _resolve_deepspeed_auto_batch(
        ds_dict, micro_batch_per_gpu, grad_accum_steps
    )

    engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_dict,
    )
    return engine, optimizer


def _train_forward_kwargs(
    batch: dict,
    backend: str,
    is_stage1: bool,
    return_query_text_tensors: bool = False,
    use_lm_ce: bool = True,
    return_attn_maps: Optional[bool] = None,
) -> dict:
    if return_attn_maps is None:
        return_attn_maps = is_stage1
    fwd = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "pixel_values": batch.get("pixel_values"),
        # Stage 1 is pure grounding when ``use_lm_ce`` is False — pass labels=None
        # so the backend skips computing the LM cross-entropy.
        "labels": batch["labels"] if use_lm_ce else None,
        "return_attn_maps": return_attn_maps,
        "return_query_text_tensors": return_query_text_tensors,
        # masked_answer_cross_entropy already applies lm_head only to supervised
        # positions; skip the full-sequence lm_head matmul that produces logits.
        "return_logits": False,
    }
    if backend == "qwen25vl":
        fwd["image_grid_thw"] = batch.get("image_grid_thw")
        fwd["video_grid_thw"] = batch.get("video_grid_thw")
    elif backend == "gemma4":
        fwd["image_position_ids"] = batch.get("image_position_ids")
        fwd["mm_token_type_ids"] = batch.get("mm_token_type_ids")
    elif backend == "llava_next":
        fwd["image_sizes"] = batch.get("image_sizes")
    return fwd


def _stage1_attn_loss(config: ReInspectionConfig, outputs, batch, device):
    global _attn_target_vis_mismatch_logged
    if outputs.attn_vis is None or "attn_target_mask" not in batch:
        return torch.zeros((), device=device)
    target = batch["attn_target_mask"]
    n_v = outputs.attn_vis.shape[-1]
    if target.shape[-1] != n_v:
        if is_main_process() and not _attn_target_vis_mismatch_logged:
            log(
                f"[WARNING] attn_target_mask width {target.shape[-1]} != attn_vis {n_v}; "
                "truncating/padding for loss. Check image patch counts vs ViT tokens if this persists."
            )
            _attn_target_vis_mismatch_logged = True
        target = F.pad(target[:, :n_v], (0, max(0, n_v - target.shape[-1])))

    k_sel = config.n_selector_queries
    selector_attn = outputs.attn_vis[:, :k_sel]

    bbox_areas = None
    if config.stage1_attn_small_box_weight and "bbox_norm" in batch:
        b = batch["bbox_norm"].to(device).float()
        bbox_areas = ((b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0))

    return compute_attn_loss_kl(
        selector_attn,
        target,
        bbox_areas=bbox_areas,
        small_box_weight=config.stage1_attn_small_box_weight and bbox_areas is not None,
    )


def _stage1_query_text_infonce_loss(config: ReInspectionConfig, outputs, device):
    """Symmetric in-batch InfoNCE between pooled text-path queries and text tokens."""
    if (
        outputs.Q_text_bottleneck is None
        or outputs.text_bottleneck is None
        or outputs.text_bottleneck_mask is None
    ):
        return torch.zeros((), device=device)
    return compute_query_text_infonce_loss(
        outputs.Q_text_bottleneck,
        outputs.text_bottleneck,
        outputs.text_bottleneck_mask,
        temperature=config.stage1_query_text_infonce_temperature,
    )


def _stage1_grounding_loss(config, model, outputs, batch, device, global_step):
    unwrapped = model.module if hasattr(model, "module") else model
    if not hasattr(unwrapped, "bbox_head"):
        return torch.zeros((), device=device)
    if outputs.R_bottleneck is None or "bbox_norm" not in batch:
        return torch.zeros((), device=device)
    bbox_gt = batch["bbox_norm"].to(device)
    selectors = outputs.R_bottleneck[:, :config.n_selector_queries]
    bbox_pred = unwrapped.bbox_head(selectors)
    warmup = min(1.0, global_step / max(1, config.stage1_grounding_warmup_steps))
    loss = compute_grounding_loss(
        bbox_pred, bbox_gt,
        l1_weight=config.stage1_grounding_l1_weight,
        giou_weight=config.stage1_grounding_giou_weight,
    )
    return loss * warmup


def _stage1_roi_feature_loss(config, model, outputs, batch, device):
    """Cosine ROI feature loss between pooled content tokens and bbox-pooled V."""
    unwrapped = model.module if hasattr(model, "module") else model
    if not hasattr(unwrapped, "roi_proj"):
        return torch.zeros((), device=device)
    if (
        outputs.R_bottleneck is None
        or outputs.vision_hidden_states is None
        or "attn_target_mask" not in batch
    ):
        return torch.zeros((), device=device)

    V_frozen = outputs.vision_hidden_states.detach()
    target = batch["attn_target_mask"]
    n_v = V_frozen.shape[1]
    if target.shape[-1] != n_v:
        target = F.pad(target[:, :n_v], (0, max(0, n_v - target.shape[-1])))

    R_content = outputs.R_bottleneck[:, config.n_selector_queries:]
    if R_content.shape[1] == 0:
        return torch.zeros((), device=device)
    loss, _cos = compute_roi_feature_loss(
        V_frozen=V_frozen,
        R_content=R_content,
        attn_target_mask=target.to(device),
        projection=unwrapped.roi_proj,
    )
    return loss


def _stage2_aux_grounding_loss(config, model, outputs, batch, device):
    """Weak Stage-2 grounding regularizer on ROI-labeled auxiliary batches."""
    attn_loss = torch.zeros((), device=device)
    roi_loss = torch.zeros((), device=device)

    if config.stage2_aux_attn_loss_weight > 0.0:
        attn_loss = _stage1_attn_loss(config, outputs, batch, device)
    if config.stage2_aux_roi_feature_loss_weight > 0.0:
        roi_loss = _stage1_roi_feature_loss(config, model, outputs, batch, device)

    total = (
        config.stage2_aux_grounding_weight
        * (
            config.stage2_aux_attn_loss_weight * attn_loss
            + config.stage2_aux_roi_feature_loss_weight * roi_loss
        )
    )
    return total, attn_loss, roi_loss


def _compute_grad_norm(model) -> float:
    """Compute the total L2 gradient norm across all trainable parameters."""
    total = 0.0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            total += p.grad.data.float().norm(2).item() ** 2
    return total ** 0.5


def _global_grad_norm_for_log(engine) -> Optional[float]:
    """Gradient norm for logging.

    Under DeepSpeed ZeRO, gradients are not stored on ``param.grad``; the engine
    records the global L2 norm during ``step()`` (used for clipping). Call this
    only **after** ``engine.step()``.
    """
    if hasattr(engine, "get_global_grad_norm"):
        raw = engine.get_global_grad_norm()
        if raw is None:
            return None
        return float(raw)
    return _compute_grad_norm(engine)


def _metric_safe_name(name: str) -> str:
    """Convert a free-form optimizer group name into a stable metric segment."""
    safe = []
    for char in name:
        if char.isalnum() or char in ("_", "-"):
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "group"


def _local_grad_for_log(param) -> Optional[torch.Tensor]:
    """Return the local gradient shard used for logging.

    For plain PyTorch training this is ``param.grad``. Under DeepSpeed ZeRO-3 we
    query the optimizer's local fp32 gradient shard to avoid gathering full
    tensors just for logging.
    """
    grad = param.grad
    if grad is not None:
        return grad.detach()
    if hasattr(param, "ds_id") and hasattr(param, "_z3_optimizer"):
        try:
            return param._z3_optimizer.get_local_fp32_grad_for_param(param)
        except Exception:
            return None
    if hasattr(param, "_hp_mapping"):
        try:
            return param.get_full_hp_grad()
        except Exception:
            return None
    return None


def _local_param_for_log(param) -> Optional[torch.Tensor]:
    """Return the local parameter shard used for logging."""
    if hasattr(param, "ds_id") and hasattr(param, "_z3_optimizer"):
        try:
            return param._z3_optimizer.get_local_fp32_param(param)
        except Exception:
            return None
    if hasattr(param, "_hp_mapping"):
        try:
            return param.get_full_hp_param()
        except Exception:
            return None
    return param.detach()


def _collect_param_stats(params: Iterable[torch.nn.Parameter]) -> dict[str, float]:
    """Collect distributed-safe gradient and parameter statistics for ``params``."""
    params = list(params)
    device = None
    for param in params:
        local_param = _local_param_for_log(param)
        if local_param is not None:
            device = local_param.device
            break
        local_grad = _local_grad_for_log(param)
        if local_grad is not None:
            device = local_grad.device
            break
        device = param.device
    if device is None:
        device = torch.device("cpu")
    # NCCL can't all_reduce CPU tensors. Under ZeRO-3, some ranks may hold no
    # local shard of these params and the fallback device above can be CPU.
    if dist.is_initialized() and device.type == "cpu" and torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())

    sums = torch.zeros(6, device=device, dtype=torch.float64)
    max_abs_grad = torch.zeros((), device=device, dtype=torch.float32)

    for param in params:
        local_param = _local_param_for_log(param)
        if local_param is not None:
            local_param = local_param.detach().float()
            sums[3] += local_param.square().sum(dtype=torch.float64)
            sums[4] += float(local_param.numel())

        local_grad = _local_grad_for_log(param)
        if local_grad is None:
            continue

        local_grad = local_grad.detach().float()
        abs_grad = local_grad.abs()
        sums[0] += local_grad.square().sum(dtype=torch.float64)
        sums[1] += abs_grad.sum(dtype=torch.float64)
        sums[2] += float(local_grad.numel())
        sums[5] += torch.count_nonzero(local_grad).to(dtype=torch.float64)
        if abs_grad.numel() > 0:
            max_abs_grad = torch.maximum(max_abs_grad, abs_grad.max())

    if dist.is_initialized():
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(max_abs_grad, op=dist.ReduceOp.MAX)

    grad_sq_sum, grad_abs_sum, grad_numel, param_sq_sum, param_numel, grad_nonzero = sums.tolist()
    grad_norm = math.sqrt(grad_sq_sum)
    param_norm = math.sqrt(param_sq_sum)
    grad_abs_mean = grad_abs_sum / max(grad_numel, 1.0)
    grad_nonzero_frac = grad_nonzero / max(grad_numel, 1.0)
    grad_coverage = grad_numel / max(param_numel, 1.0)
    grad_param_ratio = grad_norm / max(param_norm, 1e-12)

    return {
        "grad_norm": grad_norm,
        "param_norm": param_norm,
        "grad_abs_mean": grad_abs_mean,
        "grad_abs_max": float(max_abs_grad.item()),
        "grad_nonzero_frac": grad_nonzero_frac,
        "grad_coverage": grad_coverage,
        "grad_param_ratio": grad_param_ratio,
    }


def _collect_optimizer_group_grad_metrics(optimizer, metric_prefix: str) -> dict[str, float]:
    """Collect global and per-group gradient stats for the active optimizer."""
    metrics: dict[str, float] = {}

    for index, group in enumerate(optimizer.param_groups):
        group_name = _metric_safe_name(str(group.get("name", f"group_{index}")))
        stats = _collect_param_stats(group["params"])
        group_prefix = f"{metric_prefix}/groups/{group_name}"
        metrics[f"{group_prefix}/lr"] = float(group["lr"])
        metrics[f"{group_prefix}/grad_norm"] = stats["grad_norm"]
        metrics[f"{group_prefix}/param_norm"] = stats["param_norm"]
        metrics[f"{group_prefix}/grad_abs_mean"] = stats["grad_abs_mean"]
        metrics[f"{group_prefix}/grad_abs_max"] = stats["grad_abs_max"]
        metrics[f"{group_prefix}/grad_nonzero_frac"] = stats["grad_nonzero_frac"]
        metrics[f"{group_prefix}/grad_coverage"] = stats["grad_coverage"]
        metrics[f"{group_prefix}/grad_param_ratio"] = stats["grad_param_ratio"]

    # Recompute aggregate stats directly from the flattened trainable set so the
    # global metrics stay exact even if some groups are empty.
    all_params = [
        param
        for group in optimizer.param_groups
        for param in group["params"]
    ]
    total_stats = _collect_param_stats(all_params)
    metrics[f"{metric_prefix}/grad_norm"] = total_stats["grad_norm"]
    metrics[f"{metric_prefix}/param_norm"] = total_stats["param_norm"]
    metrics[f"{metric_prefix}/grad_abs_mean"] = total_stats["grad_abs_mean"]
    metrics[f"{metric_prefix}/grad_abs_max"] = total_stats["grad_abs_max"]
    metrics[f"{metric_prefix}/grad_nonzero_frac"] = total_stats["grad_nonzero_frac"]
    metrics[f"{metric_prefix}/grad_coverage"] = total_stats["grad_coverage"]
    metrics[f"{metric_prefix}/grad_param_ratio"] = total_stats["grad_param_ratio"]
    return metrics


class _CosineWarmupLR:
    """Cosine decay with linear warmup (same schedule as HF ``get_cosine_schedule_with_warmup``, num_cycles=0.5).

    Updates ``optimizer.param_groups`` directly so learning rates stay aligned with the optimizer
    DeepSpeed steps (avoids ``LambdaLR`` / ``optimizer.step()`` ordering issues on wrapped optimizers).
    """

    def __init__(self, optimizer, num_warmup: int, num_training: int) -> None:
        self.optimizer = optimizer
        self.num_warmup = max(1, num_warmup)
        self.num_training = max(1, num_training)
        self.base_lrs = [float(pg["lr"]) for pg in optimizer.param_groups]
        self._step = -1

    def step(self) -> None:
        self._step += 1
        mult = self._lr_mult(self._step)
        for pg, base in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base * mult

    def _lr_mult(self, step: int) -> float:
        if step < self.num_warmup:
            return float(step) / float(self.num_warmup)
        progress = float(step - self.num_warmup) / float(max(1, self.num_training - self.num_warmup))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    def get_last_lr(self) -> list[float]:
        return [float(pg["lr"]) for pg in self.optimizer.param_groups]


def run_training(config: ReInspectionConfig) -> None:
    """End-to-end training for one backend and stage from a resolved config.

    Sets up distributed NCCL, loads the processor (when required), constructs the
    wrapped model and optimizer groups, initializes DeepSpeed, builds train/val
    dataloaders with :func:`collate_fn`, runs epochs with optional auxiliary
    stage-1 losses, checkpoints to ``output_dir``, and tears down the process
    group on exit.

    Args:
        config: Fully populated :class:`~src.config.ReInspectionConfig`.

    Raises:
        RuntimeError: If the training dataset resolves to zero samples.
        ValueError: If DeepSpeed config path is missing.
    """
    backend = config.backend
    stage = config.stage
    is_stage1 = stage == 1
    grad_accum = config.stage1_grad_accum if is_stage1 else config.stage2_grad_accum
    n_epochs = config.stage1_epochs if is_stage1 else config.stage2_epochs
    batch_size = config.stage1_batch_size if is_stage1 else config.stage2_batch_size
    warmup_ratio = config.stage1_warmup_ratio if is_stage1 else config.stage2_warmup_ratio

    _setup_distributed()
    local_rank = _get_local_rank()
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(config.seed)

    checkpoint_root = os.path.join(config.output_dir, backend)
    if is_main_process():
        log(f"Checkpoint root: {checkpoint_root} (layout: .../{backend}/stage{{N}}/epoch_{{E}}/)")

    processor = None
    if backend == "internvl3":
        from src.backends.internvl3 import load_processor

        processor = load_processor(config)
    elif backend == "gemma4":
        from src.backends.gemma4 import load_processor as load_gemma4_proc

        processor = load_gemma4_proc(config)
    elif backend == "llava_next":
        from src.backends.llava_next import load_processor as load_llava_proc

        processor = load_llava_proc(config)
    _init_wandb(config)

    if backend == "qwen25vl":
        if is_stage1:
            model, optimizer = _setup_model_qwen25_stage1(config)
        else:
            model, optimizer = _setup_model_qwen25_stage2(config)
    elif backend == "gemma4":
        if is_stage1:
            model, optimizer = _setup_model_gemma4_stage1(config, processor)
        else:
            model, optimizer = _setup_model_gemma4_stage2(
                config, processor, config.stage1_checkpoint
            )
    elif backend == "llava_next":
        if is_stage1:
            model, optimizer = _setup_model_llava_stage1(config, processor)
        else:
            model, optimizer = _setup_model_llava_stage2(
                config, processor, config.stage1_checkpoint
            )
    else:
        if is_stage1:
            model, optimizer = _setup_model_intern_stage1(config, processor)
        else:
            model, optimizer = _setup_model_intern_stage2(
                config, processor, config.stage1_checkpoint
            )

    if config.gradient_checkpointing:
        model.base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": config.gradient_checkpointing_use_reentrant
            }
        )
        log(
            "Gradient checkpointing enabled "
            f"(use_reentrant={config.gradient_checkpointing_use_reentrant})"
        )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")
    _log_wandb({
        "model/trainable_params": trainable,
        "model/total_params": total,
        "model/trainable_pct": 100 * trainable / total,
    }, step=0)

    ds_engine, optimizer = _init_deepspeed(
        model,
        optimizer,
        config.deepspeed_config,
        batch_size,
        grad_accum,
    )
    model = ds_engine

    if backend == "qwen25vl":
        resolved = resolve_pretrained_local_path(config.model_name_or_path)
        processor = AutoProcessor.from_pretrained(
            resolved,
            max_pixels=config.max_pixels,
            min_pixels=config.min_pixels,
        )

    full_dataset = _build_dataset(backend, stage, config, config.data_root, processor)
    if len(full_dataset) == 0:
        raise RuntimeError(f"Dataset is empty. Check data_root={config.data_root}")
    train_dataset, val_dataset = _split_train_val(full_dataset, config.val_split_ratio, config.seed)
    log(
        f"Training samples: {len(train_dataset)}; "
        f"validation samples: {len(val_dataset) if val_dataset is not None else 0}"
    )

    # Keep all ranks aligned: rank-0-only W&B work must finish before any rank
    # enters the DataLoader/training loop (otherwise NCCL collectives deadlock).
    if dist.is_initialized():
        dist.barrier()
    if is_main_process():
        _log_dataset_artifact(full_dataset, stage, backend, config.data_root)
    if dist.is_initialized():
        dist.barrier()

    num_workers = config.num_workers
    # ``prefetch_factor`` and ``persistent_workers`` are only honored by
    # DataLoader when num_workers > 0.
    extra_loader_kw = {}
    if num_workers > 0:
        if config.dataloader_prefetch_factor is not None:
            extra_loader_kw["prefetch_factor"] = int(config.dataloader_prefetch_factor)
        if config.dataloader_persistent_workers:
            extra_loader_kw["persistent_workers"] = True
    sampler = DistributedSampler(train_dataset) if dist.is_initialized() else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        **extra_loader_kw,
    )

    val_loader = None
    if val_dataset is not None:
        val_batch = config.val_batch_size or batch_size
        val_sampler = (
            DistributedSampler(val_dataset, shuffle=False, drop_last=False)
            if dist.is_initialized()
            else None
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch,
            shuffle=False,
            sampler=val_sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            **extra_loader_kw,
        )

    aux_grounding_loader = None
    aux_grounding_iter = None
    use_stage2_aux_grounding = (not is_stage1) and _stage2_aux_grounding_enabled(config)
    if use_stage2_aux_grounding:
        aux_dataset = _build_stage2_aux_grounding_dataset(
            backend, config, config.data_root, processor
        )
        if len(aux_dataset) == 0:
            log(
                "[WARNING] Stage 2 auxiliary grounding is enabled, but no "
                "Stage 1 grounding samples were found; disabling auxiliary stream."
            )
            use_stage2_aux_grounding = False
        else:
            aux_batch_size = config.stage2_aux_batch_size or batch_size
            aux_sampler = DistributedSampler(aux_dataset) if dist.is_initialized() else None
            aux_grounding_loader = DataLoader(
                aux_dataset,
                batch_size=aux_batch_size,
                shuffle=aux_sampler is None,
                sampler=aux_sampler,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                **extra_loader_kw,
            )
            aux_grounding_iter = _cycle_loader(aux_grounding_loader)
            log(
                f"Stage 2 auxiliary grounding samples: {len(aux_dataset)}; "
                f"batch_size={aux_batch_size}; every_n_steps={config.stage2_aux_every_n_steps}"
            )

    num_updates = max(1, len(train_loader) * n_epochs // grad_accum)
    warmup_override = config.stage1_warmup_steps if is_stage1 else config.stage2_warmup_steps
    num_warmup = warmup_override if warmup_override is not None else int(num_updates * warmup_ratio)
    log(f"Scheduler: {num_updates} update steps, {num_warmup} warmup steps")
    if is_stage1 and is_main_process():
        log(
            f"Stage 1 LR: base lr (first param group)={optimizer.param_groups[0]['lr']:.2e}, "
            f"linear warmup for {num_warmup} optimizer steps then cosine decay."
        )
    scheduler = _CosineWarmupLR(optimizer, num_warmup, num_updates)
    _log_wandb({
        "schedule/num_updates": num_updates,
        "schedule/num_warmup": num_warmup,
        "schedule/num_epochs": n_epochs,
        "schedule/batch_size": batch_size,
        "schedule/grad_accum": grad_accum,
        "schedule/wandb_gradient_log_interval": config.wandb_gradient_log_interval,
        "dataset/num_samples": len(train_dataset),
        "dataset/num_workers": num_workers,
    }, step=0)

    use_attn = is_stage1 and config.stage1_use_attn_loss
    use_grounding = is_stage1 and config.stage1_use_grounding_loss
    use_qt_infonce = is_stage1 and config.stage1_use_query_text_infonce
    use_roi_feat = is_stage1 and config.stage1_use_roi_feature_loss
    # Stage 1 is pure grounding by default — disable LM CE / NTP unless asked.
    use_lm_ce = (not is_stage1) or config.stage1_use_lm_ce

    model.train()
    global_step = 0
    update_step = 0
    best_val_loss = float("inf")
    pfx = f"{backend}_stage{stage}"

    log(
        "Starting training. The first forward/backward on a large VLM can take several minutes."
    )

    for epoch in range(n_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        _early_exit_via_max_steps = False
        epoch_loss = 0.0
        epoch_ce = 0.0
        epoch_attn = 0.0
        epoch_grounding = 0.0
        epoch_qt_infonce = 0.0
        epoch_roi_feat = 0.0
        epoch_stage1_supervision = 0.0
        epoch_stage2_aux_grounding = 0.0
        epoch_stage2_aux_attn = 0.0
        epoch_stage2_aux_roi = 0.0

        tqdm_cls = get_tqdm(config, cloud=is_main_process())
        pbar = tqdm_cls(
            train_loader,
            desc=f"Epoch {epoch + 1}/{n_epochs}",
            disable=not is_main_process(),
            dynamic_ncols=True,
            leave=True,
        )

        # Throughput window: counts reset at every W&B log so samples/sec and
        # tokens/sec reflect the most recent interval. Re-armed per epoch so
        # end-of-epoch validation/checkpoint time never contaminates the rate.
        _tput_t0 = time.perf_counter()
        _tput_samples = 0
        _tput_tokens = 0
        _last_samples_per_sec = None

        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            _tput_samples += int(batch["input_ids"].size(0))
            _tput_tokens += int(
                batch["attention_mask"].sum().item()
                if "attention_mask" in batch
                else batch["input_ids"].numel()
            )

            outputs = model(
                **_train_forward_kwargs(
                    batch, backend, is_stage1,
                    return_query_text_tensors=use_qt_infonce,
                    use_lm_ce=use_lm_ce,
                )
            )

            ce_loss = outputs.loss
            if ce_loss is None:
                if use_lm_ce and backend in ("internvl3", "gemma4", "llava_next"):
                    raise RuntimeError("Model did not return a loss. Check dataset labels.")
                ce_loss = torch.zeros((), device=device)

            attn_loss = torch.zeros((), device=device)
            grounding_loss = torch.zeros((), device=device)
            qt_infonce_loss = torch.zeros((), device=device)
            roi_feat_loss = torch.zeros((), device=device)
            stage2_aux_grounding_loss = torch.zeros((), device=device)
            stage2_aux_attn_loss = torch.zeros((), device=device)
            stage2_aux_roi_loss = torch.zeros((), device=device)
            if use_attn:
                attn_loss = _stage1_attn_loss(config, outputs, batch, device)
            if use_grounding:
                grounding_loss = _stage1_grounding_loss(config, model, outputs, batch, device, global_step)
            if use_qt_infonce:
                qt_infonce_loss = _stage1_query_text_infonce_loss(config, outputs, device)
            if use_roi_feat:
                roi_feat_loss = _stage1_roi_feature_loss(config, model, outputs, batch, device)

            main_loss = (ce_loss if use_lm_ce else torch.zeros((), device=device)) \
                + (config.stage1_attn_loss_weight * attn_loss if use_attn else 0.0) \
                + (config.stage1_grounding_loss_weight * grounding_loss if use_grounding else 0.0) \
                + (config.stage1_roi_feature_loss_weight * roi_feat_loss if use_roi_feat else 0.0) \
                + (
                    config.stage1_query_text_infonce_weight * qt_infonce_loss
                    if use_qt_infonce
                    else 0.0
                )
            loss = main_loss

            _finite = torch.tensor(float(torch.isfinite(main_loss)), device=device)
            if dist.is_initialized():
                dist.all_reduce(_finite, op=dist.ReduceOp.MIN)
            if _finite.item() < 0.5:
                if is_main_process():
                    tqdm_stdlib.write(
                        f"[WARNING] Non-finite loss={main_loss.item():.4f} "
                        f"(ce={ce_loss.item():.4f}, attn={attn_loss.item():.4f}, "
                        f"grounding={grounding_loss.item():.4f}, "
                        f"roi={roi_feat_loss.item():.4f}, "
                        f"qt_infonce={qt_infonce_loss.item():.4f}) "
                        f"at global_step={global_step}, skipping batch"
                    )
                global_step += 1
                continue

            ds_engine.backward(main_loss / grad_accum)
            # Free the main Stage-2 graph before constructing the optional
            # auxiliary grounding graph; otherwise aux steps hold both graphs.
            loss = loss.detach()
            ce_loss = ce_loss.detach()
            attn_loss = attn_loss.detach()
            grounding_loss = grounding_loss.detach()
            qt_infonce_loss = qt_infonce_loss.detach()
            roi_feat_loss = roi_feat_loss.detach()
            del outputs, main_loss

            if (
                use_stage2_aux_grounding
                and aux_grounding_iter is not None
                and global_step % config.stage2_aux_every_n_steps == 0
            ):
                aux_batch = next(aux_grounding_iter)
                aux_batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in aux_batch.items()
                }
                aux_outputs = model(
                    **_train_forward_kwargs(
                        aux_batch,
                        backend,
                        is_stage1=False,
                        use_lm_ce=False,
                        return_attn_maps=True,
                    )
                )
                (
                    stage2_aux_grounding_loss,
                    stage2_aux_attn_loss,
                    stage2_aux_roi_loss,
                ) = _stage2_aux_grounding_loss(config, model, aux_outputs, aux_batch, device)
                aux_finite = torch.tensor(float(torch.isfinite(stage2_aux_grounding_loss)), device=device)
                if dist.is_initialized():
                    dist.all_reduce(aux_finite, op=dist.ReduceOp.MIN)
                if aux_finite.item() < 0.5:
                    if is_main_process():
                        tqdm_stdlib.write(
                            f"[WARNING] Non-finite stage2_aux_grounding_loss="
                            f"{stage2_aux_grounding_loss.item():.4f} "
                            f"at global_step={global_step}, skipping aux gradients"
                        )
                    stage2_aux_grounding_loss = torch.zeros((), device=device)
                    stage2_aux_attn_loss = torch.zeros((), device=device)
                    stage2_aux_roi_loss = torch.zeros((), device=device)
                else:
                    ds_engine.backward(stage2_aux_grounding_loss / grad_accum)
                    loss = loss + stage2_aux_grounding_loss.detach()
                del aux_outputs
            grad_norm = None
            grad_metrics = None
            if (step + 1) % grad_accum == 0:
                next_update_step = update_step + 1
                if (
                    config.wandb_gradient_log_interval > 0
                    and next_update_step % config.wandb_gradient_log_interval == 0
                ):
                    grad_metrics = _collect_optimizer_group_grad_metrics(
                        optimizer,
                        f"{pfx}/train",
                    )
                ds_engine.step()
                scheduler.step()
                update_step = next_update_step
                if grad_metrics is not None:
                    grad_norm = grad_metrics.get(f"{pfx}/train/grad_norm")
                else:
                    grad_norm = _global_grad_norm_for_log(ds_engine)

            epoch_loss += loss.item()
            epoch_ce += ce_loss.item()
            epoch_attn += attn_loss.item()
            epoch_grounding += grounding_loss.item()
            epoch_qt_infonce += qt_infonce_loss.item()
            epoch_roi_feat += roi_feat_loss.item()
            epoch_stage2_aux_grounding += stage2_aux_grounding_loss.item()
            epoch_stage2_aux_attn += stage2_aux_attn_loss.item()
            epoch_stage2_aux_roi += stage2_aux_roi_loss.item()
            global_step += 1

            lr = scheduler.get_last_lr()[0] if update_step > 0 else optimizer.param_groups[0]["lr"]
            metrics = {
                f"{pfx}/train/loss": loss.item(),
                f"{pfx}/train/ce_loss": ce_loss.item(),
                f"{pfx}/train/lr": lr,
                f"{pfx}/train/update_step": update_step,
            }
            if use_attn:
                metrics[f"{pfx}/train/attn_loss"] = attn_loss.item()
            if use_grounding:
                metrics[f"{pfx}/train/grounding_loss"] = grounding_loss.item()
            if use_qt_infonce:
                metrics[f"{pfx}/train/query_text_infonce_loss"] = qt_infonce_loss.item()
            if use_roi_feat:
                metrics[f"{pfx}/train/roi_feature_loss"] = roi_feat_loss.item()
            if use_stage2_aux_grounding:
                metrics[f"{pfx}/train/stage2_aux_grounding_loss"] = stage2_aux_grounding_loss.item()
                metrics[f"{pfx}/train/stage2_aux_attn_loss"] = stage2_aux_attn_loss.item()
                metrics[f"{pfx}/train/stage2_aux_roi_feature_loss"] = stage2_aux_roi_loss.item()
            if is_stage1 and (use_attn or use_grounding or use_qt_infonce or use_roi_feat):
                stage1_sup = 0.0
                if use_attn:
                    stage1_sup += config.stage1_attn_loss_weight * attn_loss.item()
                if use_grounding:
                    stage1_sup += config.stage1_grounding_loss_weight * grounding_loss.item()
                if use_qt_infonce:
                    stage1_sup += config.stage1_query_text_infonce_weight * qt_infonce_loss.item()
                if use_roi_feat:
                    stage1_sup += config.stage1_roi_feature_loss_weight * roi_feat_loss.item()
                metrics[f"{pfx}/train/stage1_supervision_loss"] = stage1_sup
                epoch_stage1_supervision += stage1_sup

            if grad_metrics is not None:
                metrics.update(grad_metrics)
            elif grad_norm is not None:
                metrics[f"{pfx}/train/grad_norm"] = grad_norm

            if torch.cuda.is_available():
                metrics[f"{pfx}/system/gpu_mem_allocated_gb"] = torch.cuda.memory_allocated(device) / (1024 ** 3)
                metrics[f"{pfx}/system/gpu_mem_reserved_gb"] = torch.cuda.memory_reserved(device) / (1024 ** 3)

            if grad_metrics is not None or global_step % config.wandb_log_interval == 0:
                _tput_elapsed = time.perf_counter() - _tput_t0
                if _tput_elapsed > 0:
                    _world = dist.get_world_size() if dist.is_initialized() else 1
                    _sps = _tput_samples / _tput_elapsed
                    _tps = _tput_tokens / _tput_elapsed
                    _last_samples_per_sec = _sps * _world
                    metrics[f"{pfx}/throughput/samples_per_sec"] = _sps * _world
                    metrics[f"{pfx}/throughput/tokens_per_sec"] = _tps * _world
                    metrics[f"{pfx}/throughput/samples_per_sec_per_gpu"] = _sps
                _tput_t0 = time.perf_counter()
                _tput_samples = 0
                _tput_tokens = 0
                _log_wandb(metrics, global_step)

            postfix: dict = {"loss": f"{loss.item():.4f}", "lr": f"{lr:.2e}"}
            if use_lm_ce:
                postfix["ce"] = f"{ce_loss.item():.4f}"
            if use_attn:
                postfix["attn"] = f"{attn_loss.item():.4f}"
            if use_grounding:
                postfix["gnd"] = f"{grounding_loss.item():.4f}"
            if use_roi_feat:
                postfix["roi"] = f"{roi_feat_loss.item():.4f}"
            if use_stage2_aux_grounding:
                postfix["s2_aux"] = f"{stage2_aux_grounding_loss.item():.4f}"
            if use_qt_infonce:
                postfix["qt_nce"] = f"{qt_infonce_loss.item():.4f}"
            if grad_norm is not None:
                postfix["gnorm"] = f"{grad_norm:.2f}"
            if _last_samples_per_sec is not None:
                postfix["sps"] = f"{_last_samples_per_sec:.1f}"
            pbar.set_postfix(postfix)

            if (
                is_stage1
                and config.stage1_max_steps is not None
                and global_step >= config.stage1_max_steps
            ):
                if is_main_process():
                    log(f"Reached stage1_max_steps={config.stage1_max_steps}; stopping early.")
                _early_exit_via_max_steps = True
                break
            if (
                (not is_stage1)
                and config.stage2_max_steps is not None
                and global_step >= config.stage2_max_steps
            ):
                if is_main_process():
                    log(f"Reached stage2_max_steps={config.stage2_max_steps}; stopping early.")
                _early_exit_via_max_steps = True
                break

        pbar.close()
        # When the inner loop tripped a max-steps early exit, skip the rest of
        # the epoch boilerplate (validation, checkpoint save, artifact upload).
        # Smoke runs and short ablation sweeps depend on hard-exiting here.
        if _early_exit_via_max_steps:
            if is_main_process():
                log("Skipping end-of-epoch validation/checkpoint due to max_steps early exit.")
            break

        num_steps = max(1, len(train_loader))
        avg_epoch_loss = epoch_loss / num_steps
        summary = {
            f"{pfx}/epoch/loss": avg_epoch_loss,
            f"{pfx}/epoch/ce_loss": epoch_ce / num_steps,
            f"{pfx}/epoch/epoch": epoch + 1,
        }
        if use_attn:
            summary[f"{pfx}/epoch/attn_loss"] = epoch_attn / num_steps
        if use_grounding:
            summary[f"{pfx}/epoch/grounding_loss"] = epoch_grounding / num_steps
        if use_qt_infonce:
            summary[f"{pfx}/epoch/query_text_infonce_loss"] = epoch_qt_infonce / num_steps
        if use_roi_feat:
            summary[f"{pfx}/epoch/roi_feature_loss"] = epoch_roi_feat / num_steps
        if use_stage2_aux_grounding:
            summary[f"{pfx}/epoch/stage2_aux_grounding_loss"] = epoch_stage2_aux_grounding / num_steps
            summary[f"{pfx}/epoch/stage2_aux_attn_loss"] = epoch_stage2_aux_attn / num_steps
            summary[f"{pfx}/epoch/stage2_aux_roi_feature_loss"] = epoch_stage2_aux_roi / num_steps
        if is_stage1 and (use_attn or use_grounding or use_qt_infonce or use_roi_feat):
            summary[f"{pfx}/epoch/stage1_supervision_loss"] = epoch_stage1_supervision / num_steps
        _log_wandb(summary, global_step)

        aux_msg = ""
        if use_attn:
            aux_msg += f" attn={epoch_attn / num_steps:.4f}"
        if use_grounding:
            aux_msg += f" ground={epoch_grounding / num_steps:.4f}"
        if use_roi_feat:
            aux_msg += f" roi={epoch_roi_feat / num_steps:.4f}"
        if use_stage2_aux_grounding:
            aux_msg += f" s2_aux={epoch_stage2_aux_grounding / num_steps:.4f}"
        if use_qt_infonce:
            aux_msg += f" qt_nce={epoch_qt_infonce / num_steps:.4f}"
        if is_main_process():
            tqdm_stdlib.write(
                f"Epoch {epoch + 1}/{n_epochs}: loss={avg_epoch_loss:.4f} ce={epoch_ce / num_steps:.4f}"
                + aux_msg
            )

        val_loss = None
        if val_loader is not None:
            val_metrics = _run_validation(
                ds_engine, val_loader, backend, is_stage1,
                use_attn, use_grounding, use_qt_infonce, use_roi_feat, use_lm_ce,
                config, device,
            )
            val_loss = val_metrics["loss"]
            val_summary = {
                f"{pfx}/val/loss": val_metrics["loss"],
                f"{pfx}/val/ce_loss": val_metrics["ce_loss"],
                f"{pfx}/val/epoch": epoch + 1,
                f"{pfx}/val/num_batches": val_metrics["num_batches"],
            }
            if use_attn:
                val_summary[f"{pfx}/val/attn_loss"] = val_metrics["attn_loss"]
            if use_grounding:
                val_summary[f"{pfx}/val/grounding_loss"] = val_metrics["grounding_loss"]
            if use_qt_infonce:
                val_summary[f"{pfx}/val/query_text_infonce_loss"] = val_metrics["query_text_infonce_loss"]
            if use_roi_feat:
                val_summary[f"{pfx}/val/roi_feature_loss"] = val_metrics["roi_feature_loss"]
            if is_stage1 and (use_attn or use_grounding or use_qt_infonce or use_roi_feat):
                sup = 0.0
                if use_attn:
                    sup += config.stage1_attn_loss_weight * val_metrics["attn_loss"]
                if use_grounding:
                    sup += config.stage1_grounding_loss_weight * val_metrics["grounding_loss"]
                if use_qt_infonce:
                    sup += config.stage1_query_text_infonce_weight * val_metrics["query_text_infonce_loss"]
                if use_roi_feat:
                    sup += config.stage1_roi_feature_loss_weight * val_metrics["roi_feature_loss"]
                val_summary[f"{pfx}/val/stage1_supervision_loss"] = sup
            _log_wandb(val_summary, global_step)
            if is_main_process():
                tqdm_stdlib.write(
                    f"Epoch {epoch + 1}/{n_epochs} val: loss={val_metrics['loss']:.4f} "
                    f"ce={val_metrics['ce_loss']:.4f}"
                )

        parts = [config.output_dir, backend]
        if config.experiment_name:
            parts.append(config.experiment_name)
        parts.extend([f"stage{stage}", f"epoch_{epoch + 1}"])
        save_dir = os.path.join(*parts)
        _save_checkpoint(
            ds_engine,
            save_dir=save_dir,
            save_lora=not is_stage1,
        )

        tracked_loss = val_loss if val_loss is not None else avg_epoch_loss
        is_best = tracked_loss < best_val_loss
        if is_best:
            best_val_loss = tracked_loss
        _log_model_artifact(save_dir, stage, backend, epoch + 1, is_best=is_best)

        if (
            is_stage1
            and config.stage1_max_steps is not None
            and global_step >= config.stage1_max_steps
        ):
            break
        if (
            (not is_stage1)
            and config.stage2_max_steps is not None
            and global_step >= config.stage2_max_steps
        ):
            break

    _finish_wandb()
    if dist.is_initialized():
        dist.destroy_process_group()
