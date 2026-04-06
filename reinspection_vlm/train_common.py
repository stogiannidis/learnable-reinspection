"""Shared training utilities for all VLM backends."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from typing import Optional, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoProcessor

from reinspection_vlm.attn_loss import compute_attn_loss_focal, compute_attn_loss_kl
from reinspection_vlm.bbox_head import BboxHead, compute_grounding_loss
from reinspection_vlm.config import ReInspectionConfig
from reinspection_vlm.data.refcoco import RefCOCODataset
from reinspection_vlm.data.spatial_dataset import build_spatial_dataset

_CONCAT_KEYS = {"pixel_values", "image_grid_thw", "video_grid_thw"}
_VARLEN_FLOAT_PAD_KEYS = {"attn_target_mask", "bbox_norm"}
_VARLEN_PAD_KEYS = _VARLEN_FLOAT_PAD_KEYS


def collate_fn(batch):
    keys = batch[0].keys()
    collated = {}
    for key in keys:
        values = [item[key] for item in batch]
        if not isinstance(values[0], torch.Tensor):
            collated[key] = values
            continue
        if key in _CONCAT_KEYS:
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
    return not dist.is_initialized() or dist.get_rank() == 0


def log(message: str) -> None:
    if is_main_process():
        print(message, flush=True)


def _setup_distributed() -> None:
    if dist.is_initialized():
        return
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def _get_local_rank() -> int:
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

        for name in ["refcoco", "refcoco+", "refcocog"]:
            for split in ["train", "test"]:
                p = os.path.join(data_root, name, f"{split}.json")
                if os.path.exists(p):
                    ann_files.add(p)
        for name in ["vsr", "whatsup", "gqa_spatial", "spatialbench", "3dsrbench", "mindcube", "blink", "srbench"]:
            for ext in ["json", "jsonl"]:
                for split in ["train", "test"]:
                    p = os.path.join(data_root, name, f"{split}.{ext}")
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
        code_dir = os.path.join(os.path.dirname(__file__))
        if os.path.isdir(code_dir):
            art.add_dir(code_dir, name="reinspection_vlm")
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


def _attach_bbox_head(model, config: ReInspectionConfig) -> None:
    """Create and attach a BboxHead for Stage 1 grounding supervision."""
    dtype = torch.bfloat16 if config.bf16 else torch.float32
    model.bbox_head = BboxHead(config.d_bottleneck, dtype=dtype)


def _setup_model_qwen_stage1(config: ReInspectionConfig):
    from reinspection_vlm.backends.qwen3vl import load_model

    model = load_model(config, device_map=None)
    for param in model.base_model.parameters():
        param.requires_grad = False
    for param in model.reinspection.parameters():
        param.requires_grad = True
    optim_groups = [{"params": list(model.reinspection.parameters()), "lr": config.stage1_lr_module}]
    if config.stage1_aux_loss in ("grounding", "both"):
        _attach_bbox_head(model, config)
        optim_groups.append({"params": list(model.bbox_head.parameters()), "lr": config.stage1_lr_module})
    optimizer = torch.optim.AdamW(optim_groups, weight_decay=0.01)
    return model, optimizer


def _setup_model_qwen_stage2(config: ReInspectionConfig):
    from reinspection_vlm.backends.qwen3vl import load_model

    model = load_model(config, device_map=None)
    if config.stage1_checkpoint:
        log(f"Loading Stage 1 checkpoint: {config.stage1_checkpoint}")
        _load_stage1_weights(model, config.stage1_checkpoint)
    for param in model.base_model.parameters():
        param.requires_grad = False
    for param in model.reinspection.parameters():
        param.requires_grad = True
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
    reinspection_params = list(model.reinspection.parameters())
    lora_params = [p for _, p in model.base_model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": reinspection_params, "lr": config.stage2_lr_module},
            {"params": lora_params, "lr": config.stage2_lr_lora},
        ],
        weight_decay=0.01,
    )
    return model, optimizer


def _setup_model_intern_stage1(config: ReInspectionConfig, processor):
    from reinspection_vlm.backends.internvl3 import load_model

    model = load_model(config, device_map=None, processor=processor)
    for parameter in model.base_model.parameters():
        parameter.requires_grad = False
    for parameter in model.reinspection.parameters():
        parameter.requires_grad = True
    optim_groups = [{"params": list(model.reinspection.parameters()), "lr": config.stage1_lr_module}]
    if config.stage1_aux_loss in ("grounding", "both"):
        _attach_bbox_head(model, config)
        optim_groups.append({"params": list(model.bbox_head.parameters()), "lr": config.stage1_lr_module})
    if config.stage1_train_projector:
        for parameter in model.base_model.model.multi_modal_projector.parameters():
            parameter.requires_grad = True
        optim_groups.append(
            {
                "params": list(model.base_model.model.multi_modal_projector.parameters()),
                "lr": config.stage1_lr_projector,
            }
        )
    optimizer = torch.optim.AdamW(optim_groups, weight_decay=0.01)
    return model, optimizer


def _setup_model_intern_stage2(config: ReInspectionConfig, processor, stage1_checkpoint: Optional[str]):
    from reinspection_vlm.backends.internvl3 import load_model

    model = load_model(config, device_map=None, processor=processor)
    if stage1_checkpoint:
        _load_stage1_weights(model, stage1_checkpoint)
    for parameter in model.base_model.parameters():
        parameter.requires_grad = False
    for parameter in model.reinspection.parameters():
        parameter.requires_grad = True
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
    reinspection_params = list(model.reinspection.parameters())
    lora_params = [
        param for _, param in model.base_model.model.language_model.named_parameters() if param.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": reinspection_params, "lr": config.stage2_lr_module},
            {"params": lora_params, "lr": config.stage2_lr_lora},
        ],
        weight_decay=0.01,
    )
    return model, optimizer


# ---- Qwen2.5-VL setup (same architecture as Qwen3VL) ----

def _setup_model_qwen25_stage1(config: ReInspectionConfig):
    from reinspection_vlm.backends.qwen25vl import load_model

    model = load_model(config, device_map=None)
    for param in model.base_model.parameters():
        param.requires_grad = False
    for param in model.reinspection.parameters():
        param.requires_grad = True
    optimizer = torch.optim.AdamW(
        model.reinspection.parameters(),
        lr=config.stage1_lr_module,
        weight_decay=0.01,
    )
    return model, optimizer


def _setup_model_qwen25_stage2(config: ReInspectionConfig):
    from reinspection_vlm.backends.qwen25vl import load_model

    model = load_model(config, device_map=None)
    if config.stage1_checkpoint:
        log(f"Loading Stage 1 checkpoint: {config.stage1_checkpoint}")
        _load_stage1_weights(model, config.stage1_checkpoint)
    for param in model.base_model.parameters():
        param.requires_grad = False
    for param in model.reinspection.parameters():
        param.requires_grad = True
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
    reinspection_params = list(model.reinspection.parameters())
    lora_params = [p for _, p in model.base_model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": reinspection_params, "lr": config.stage2_lr_module},
            {"params": lora_params, "lr": config.stage2_lr_lora},
        ],
        weight_decay=0.01,
    )
    return model, optimizer


# ---- Gemma4 setup ----

def _setup_model_gemma4_stage1(config: ReInspectionConfig, processor):
    from reinspection_vlm.backends.gemma4 import load_model

    model = load_model(config, device_map=None, processor=processor)
    for parameter in model.base_model.parameters():
        parameter.requires_grad = False
    for parameter in model.reinspection.parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.AdamW(
        model.reinspection.parameters(),
        lr=config.stage1_lr_module,
        weight_decay=0.01,
    )
    return model, optimizer


def _setup_model_gemma4_stage2(config: ReInspectionConfig, processor, stage1_checkpoint: Optional[str]):
    from reinspection_vlm.backends.gemma4 import load_model

    model = load_model(config, device_map=None, processor=processor)
    if stage1_checkpoint:
        _load_stage1_weights(model, stage1_checkpoint)
    for parameter in model.base_model.parameters():
        parameter.requires_grad = False
    for parameter in model.reinspection.parameters():
        parameter.requires_grad = True
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
    reinspection_params = list(model.reinspection.parameters())
    lora_params = [
        param for _, param in model.base_model.model.language_model.named_parameters() if param.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": reinspection_params, "lr": config.stage2_lr_module},
            {"params": lora_params, "lr": config.stage2_lr_lora},
        ],
        weight_decay=0.01,
    )
    return model, optimizer


def _build_dataset(
    backend: str,
    stage: int,
    config: ReInspectionConfig,
    data_root: str,
    processor,
):
    if stage == 1:
        return RefCOCODataset(
            data_root=data_root,
            processor=processor,
            backend=backend,
            split="train",
            max_pixels=config.max_pixels,
            min_pixels=config.min_pixels,
            crop_to_patches=config.crop_to_patches_stage1,
            system_prompt=config.system_prompt,
            answer_ignore_index=config.answer_ignore_index,
        )
    return build_spatial_dataset(
        data_root=data_root,
        processor=processor,
        backend=backend,
        split="train",
        max_pixels=config.max_pixels,
        min_pixels=config.min_pixels,
        crop_to_patches=config.crop_to_patches_stage2,
        system_prompt=config.system_prompt,
        answer_ignore_index=config.answer_ignore_index,
    )


def _save_checkpoint(ds_engine, save_dir: str, save_lora: bool = False) -> None:
    import deepspeed

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

    if save_lora:
        lora_params = list(unwrapped.base_model.model.language_model.parameters())
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


def _train_forward_kwargs(batch: dict, backend: str, is_stage1: bool) -> dict:
    fwd = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "pixel_values": batch.get("pixel_values"),
        "labels": batch["labels"],
        "return_attn_maps": is_stage1,
    }
    if backend in ("qwen3vl", "qwen25vl"):
        fwd["image_grid_thw"] = batch.get("image_grid_thw")
        fwd["video_grid_thw"] = batch.get("video_grid_thw")
    elif backend == "gemma4":
        fwd["image_position_ids"] = batch.get("image_position_ids")
        fwd["mm_token_type_ids"] = batch.get("mm_token_type_ids")
    return fwd


def _stage1_attn_loss(config: ReInspectionConfig, outputs, batch, device):
    if outputs.attn_vis is None or "attn_target_mask" not in batch:
        return torch.zeros((), device=device)
    target = batch["attn_target_mask"]
    n_v = outputs.attn_vis.shape[-1]
    if target.shape[-1] != n_v:
        target = F.pad(target[:, :n_v], (0, max(0, n_v - target.shape[-1])))
    if config.stage1_attn_loss_type == "kl":
        return compute_attn_loss_kl(outputs.attn_vis, target)
    return compute_attn_loss_focal(
        outputs.attn_vis,
        target,
        image_grid_thw=batch.get("image_grid_thw"),
        n_queries=config.n_queries,
    )


def _stage1_grounding_loss(config, model, outputs, batch, device, global_step):
    unwrapped = model.module if hasattr(model, "module") else model
    if not hasattr(unwrapped, "bbox_head"):
        return torch.zeros((), device=device)
    if outputs.R_bottleneck is None or "bbox_norm" not in batch:
        return torch.zeros((), device=device)
    bbox_gt = batch["bbox_norm"].to(device)
    bbox_pred = unwrapped.bbox_head(outputs.R_bottleneck)
    warmup = min(1.0, global_step / max(1, config.stage1_grounding_warmup_steps))
    loss = compute_grounding_loss(
        bbox_pred, bbox_gt,
        l1_weight=config.stage1_grounding_l1_weight,
        giou_weight=config.stage1_grounding_giou_weight,
    )
    return loss * warmup


def _compute_grad_norm(model) -> float:
    """Compute the total L2 gradient norm across all trainable parameters."""
    total = 0.0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            total += p.grad.data.float().norm(2).item() ** 2
    return total ** 0.5


def _compute_param_norm(model) -> float:
    """Compute the total L2 parameter norm across trainable parameters."""
    total = 0.0
    for p in model.parameters():
        if p.requires_grad:
            total += p.data.float().norm(2).item() ** 2
    return total ** 0.5


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
        from reinspection_vlm.backends.internvl3 import load_processor

        processor = load_processor(config)
    elif backend == "gemma4":
        from reinspection_vlm.backends.gemma4 import load_processor as load_gemma4_proc

        processor = load_gemma4_proc(config)
    _init_wandb(config)

    if backend == "qwen3vl":
        if is_stage1:
            model, optimizer = _setup_model_qwen_stage1(config)
        else:
            model, optimizer = _setup_model_qwen_stage2(config)
    elif backend == "qwen25vl":
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
    else:
        if is_stage1:
            model, optimizer = _setup_model_intern_stage1(config, processor)
        else:
            model, optimizer = _setup_model_intern_stage2(
                config, processor, config.stage1_checkpoint
            )

    if config.gradient_checkpointing:
        if backend == "internvl3":
            model.base_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": True}
            )
            log("Gradient checkpointing enabled (reentrant mode for InternVL3)")
        else:
            model.base_model.gradient_checkpointing_enable()
            log("Gradient checkpointing enabled")

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

    if backend in ("qwen3vl", "qwen25vl"):
        processor = AutoProcessor.from_pretrained(
            config.model_name_or_path,
            max_pixels=config.max_pixels,
            min_pixels=config.min_pixels,
        )

    train_dataset = _build_dataset(backend, stage, config, config.data_root, processor)
    if len(train_dataset) == 0:
        raise RuntimeError(f"Dataset is empty. Check data_root={config.data_root}")
    log(f"Training samples: {len(train_dataset)}")

    if is_main_process():
        _log_dataset_artifact(train_dataset, stage, backend, config.data_root)

    num_workers = config.num_workers
    sampler = DistributedSampler(train_dataset) if dist.is_initialized() else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    num_updates = max(1, len(train_loader) * n_epochs // grad_accum)
    warmup_override = config.stage1_warmup_steps if is_stage1 else config.stage2_warmup_steps
    num_warmup = warmup_override if warmup_override is not None else int(num_updates * warmup_ratio)
    log(f"Scheduler: {num_updates} update steps, {num_warmup} warmup steps")
    scheduler = _CosineWarmupLR(optimizer, num_warmup, num_updates)
    _log_wandb({
        "schedule/num_updates": num_updates,
        "schedule/num_warmup": num_warmup,
        "schedule/num_epochs": n_epochs,
        "schedule/batch_size": batch_size,
        "schedule/grad_accum": grad_accum,
        "dataset/num_samples": len(train_dataset),
        "dataset/num_workers": num_workers,
    }, step=0)

    use_attn = is_stage1 and config.stage1_aux_loss in ("attn", "both")
    use_grounding = is_stage1 and config.stage1_aux_loss in ("grounding", "both")

    model.train()
    global_step = 0
    update_step = 0
    best_epoch_loss = float("inf")
    pfx = f"{backend}_stage{stage}"

    for epoch in range(n_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_ce = 0.0
        epoch_attn = 0.0
        epoch_grounding = 0.0

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            outputs = model(**_train_forward_kwargs(batch, backend, is_stage1))

            ce_loss = outputs.loss
            if ce_loss is None:
                if backend in ("internvl3", "gemma4"):
                    raise RuntimeError("Model did not return a loss. Check dataset labels.")
                ce_loss = torch.tensor(0.0, device=device)

            attn_loss = torch.zeros((), device=device)
            grounding_loss = torch.zeros((), device=device)
            if use_attn:
                attn_loss = _stage1_attn_loss(config, outputs, batch, device)
            if use_grounding:
                grounding_loss = _stage1_grounding_loss(config, model, outputs, batch, device, global_step)

            loss = ce_loss \
                + (config.stage1_attn_loss_weight * attn_loss if use_attn else 0.0) \
                + (config.stage1_grounding_loss_weight * grounding_loss if use_grounding else 0.0)
            micro_loss = loss / grad_accum

            _finite = torch.tensor(float(torch.isfinite(loss)), device=device)
            if dist.is_initialized():
                dist.all_reduce(_finite, op=dist.ReduceOp.MIN)
            if _finite.item() < 0.5:
                log(
                    f"[WARNING] Non-finite loss={loss.item():.4f} "
                    f"(ce={ce_loss.item():.4f}, attn={attn_loss.item():.4f}, "
                    f"grounding={grounding_loss.item():.4f}) "
                    f"at global_step={global_step}, skipping batch"
                )
                global_step += 1
                continue

            ds_engine.backward(micro_loss)
            grad_norm = None
            if (step + 1) % grad_accum == 0:
                grad_norm = _compute_grad_norm(model)
                ds_engine.step()
                scheduler.step()
                update_step += 1

            epoch_loss += loss.item()
            epoch_ce += ce_loss.item()
            epoch_attn += attn_loss.item()
            epoch_grounding += grounding_loss.item()
            global_step += 1

            lr = scheduler.get_last_lr()[0] if update_step > 0 else optimizer.param_groups[0]["lr"]
            metrics = {
                f"{pfx}/train/loss": loss.item(),
                f"{pfx}/train/ce_loss": ce_loss.item(),
                f"{pfx}/train/lr": lr,
            }
            if use_attn:
                metrics[f"{pfx}/train/attn_loss"] = attn_loss.item()
            if use_grounding:
                metrics[f"{pfx}/train/grounding_loss"] = grounding_loss.item()

            if grad_norm is not None:
                metrics[f"{pfx}/train/grad_norm"] = grad_norm
                metrics[f"{pfx}/train/param_norm"] = _compute_param_norm(model)

            if torch.cuda.is_available():
                metrics[f"{pfx}/system/gpu_mem_allocated_gb"] = torch.cuda.memory_allocated(device) / (1024 ** 3)
                metrics[f"{pfx}/system/gpu_mem_reserved_gb"] = torch.cuda.memory_reserved(device) / (1024 ** 3)

            _log_wandb(metrics, global_step)

            if global_step % 1000 == 0:
                msg = f"Epoch {epoch + 1} Step {global_step}: loss={loss.item():.4f} ce={ce_loss.item():.4f}"
                if use_attn:
                    msg += f" attn={attn_loss.item():.4f}"
                if use_grounding:
                    msg += f" ground={grounding_loss.item():.4f}"
                msg += f" lr={lr:.2e}"
                log("  " + msg)

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
        _log_wandb(summary, global_step)

        aux_msg = ""
        if use_attn:
            aux_msg += f" attn={epoch_attn / num_steps:.4f}"
        if use_grounding:
            aux_msg += f" ground={epoch_grounding / num_steps:.4f}"
        log(
            f"Epoch {epoch + 1}/{n_epochs}: loss={avg_epoch_loss:.4f} ce={epoch_ce / num_steps:.4f}"
            + aux_msg
        )

        save_dir = os.path.join(config.output_dir, backend, f"stage{stage}", f"epoch_{epoch + 1}")
        _save_checkpoint(
            ds_engine,
            save_dir=save_dir,
            save_lora=not is_stage1,
        )

        is_best = avg_epoch_loss < best_epoch_loss
        if is_best:
            best_epoch_loss = avg_epoch_loss
        _log_model_artifact(save_dir, stage, backend, epoch + 1, is_best=is_best)

    _finish_wandb()
    if dist.is_initialized():
        dist.destroy_process_group()
