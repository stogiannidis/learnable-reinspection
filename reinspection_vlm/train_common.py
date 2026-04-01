"""Shared training utilities for all VLM backends."""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoProcessor, get_cosine_schedule_with_warmup

from reinspection_vlm.attn_loss import compute_attn_loss_focal, compute_attn_loss_kl
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


def load_config(config_path: Optional[str], output_dir: str) -> ReInspectionConfig:
    config = ReInspectionConfig(output_dir=output_dir)
    if not config_path:
        return config
    with open(config_path, "r", encoding="utf-8") as handle:
        overrides = yaml.safe_load(handle) or {}
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    config.__post_init__()
    return config


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


def _init_wandb(config: ReInspectionConfig, args, stage: int, backend: str) -> None:
    if not is_main_process():
        return
    try:
        import wandb

        os.environ["WANDB_SILENT"] = "true"
        resolved = asdict(config)
        resolved.update({"stage": stage, "backend": backend, "_cli": vars(args)})
        name = getattr(args, "wandb_run_name", None)
        if name:
            name = f"{name}_stage{stage}"
        wandb.init(
            project=getattr(args, "wandb_project", None),
            name=name,
            config=resolved,
        )
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


def _load_stage1_weights(model, checkpoint_path: str) -> None:
    state_path = checkpoint_path
    if os.path.isdir(state_path):
        state_path = os.path.join(state_path, "reinspection_module.pt")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"Stage 1 checkpoint not found: {state_path}")
    state_dict = torch.load(state_path, map_location="cpu", weights_only=True)
    model.reinspection.load_state_dict(state_dict)


def _setup_model_qwen_stage1(config: ReInspectionConfig, args):
    from reinspection_vlm.backends.qwen3vl import load_model

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


def _setup_model_qwen_stage2(config: ReInspectionConfig, args):
    from reinspection_vlm.backends.qwen3vl import load_model

    model = load_model(config, device_map=None)
    if args.stage1_checkpoint:
        log(f"Loading Stage 1 checkpoint: {args.stage1_checkpoint}")
        _load_stage1_weights(model, args.stage1_checkpoint)
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


def _save_checkpoint(model, save_dir: str, save_lora: bool = False) -> None:
    if not is_main_process():
        return
    os.makedirs(save_dir, exist_ok=True)
    unwrapped = model.module if hasattr(model, "module") else model
    torch.save(unwrapped.reinspection.state_dict(), os.path.join(save_dir, "reinspection_module.pt"))
    if save_lora:
        unwrapped.base_model.model.language_model.save_pretrained(os.path.join(save_dir, "lora_weights"))
    log(f"Saved checkpoint to {save_dir}")


def _try_deepspeed(model, optimizer, deepspeed_config: Optional[str]):
    if not deepspeed_config:
        return None, model, optimizer
    import deepspeed

    engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=deepspeed_config,
    )
    return engine, engine, optimizer


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


def run_training(backend: str, stage: int, config: ReInspectionConfig, args) -> None:
    is_stage1 = stage == 1
    grad_accum = config.stage1_grad_accum if is_stage1 else config.stage2_grad_accum
    n_epochs = config.stage1_epochs if is_stage1 else config.stage2_epochs
    batch_size = config.stage1_batch_size if is_stage1 else config.stage2_batch_size
    warmup_ratio = config.stage1_warmup_ratio if is_stage1 else config.stage2_warmup_ratio

    _setup_distributed()
    local_rank = _get_local_rank()
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(config.seed)

    processor = None
    if backend == "internvl3":
        from reinspection_vlm.backends.internvl3 import load_processor

        processor = load_processor(config)
    _init_wandb(config, args, stage, backend)

    if backend == "qwen3vl":
        if is_stage1:
            model, optimizer = _setup_model_qwen_stage1(config, args)
        else:
            model, optimizer = _setup_model_qwen_stage2(config, args)
    else:
        if is_stage1:
            model, optimizer = _setup_model_intern_stage1(config, processor)
        else:
            model, optimizer = _setup_model_intern_stage2(
                config, processor, getattr(args, "stage1_checkpoint", None)
            )

    if config.gradient_checkpointing:
        model.base_model.gradient_checkpointing_enable()
        log("Gradient checkpointing enabled")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")

    ds_engine, model, optimizer = _try_deepspeed(model, optimizer, getattr(args, "deepspeed", None))
    use_deepspeed = ds_engine is not None
    if not use_deepspeed:
        model = model.to(device)
        if dist.is_initialized():
            model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    if backend == "qwen3vl":
        processor = AutoProcessor.from_pretrained(
            config.model_name_or_path,
            max_pixels=config.max_pixels,
            min_pixels=config.min_pixels,
        )

    train_dataset = _build_dataset(backend, stage, config, args.data_root, processor)
    if len(train_dataset) == 0:
        raise RuntimeError(f"Dataset is empty. Check --data_root={args.data_root}")
    log(f"Training samples: {len(train_dataset)}")

    num_workers = getattr(args, "num_workers", 4)
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
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup, num_updates)

    if not use_deepspeed:
        optimizer.zero_grad(set_to_none=True)

    model.train()
    global_step = 0
    update_step = 0
    pfx = f"{backend}_stage{stage}"

    for epoch in range(n_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_ce = 0.0
        epoch_attn = 0.0

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            fwd = dict(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch.get("pixel_values"),
                labels=batch["labels"],
                return_attn_maps=is_stage1,
            )
            if backend == "qwen3vl":
                fwd["image_grid_thw"] = batch.get("image_grid_thw")
                fwd["video_grid_thw"] = batch.get("video_grid_thw")
            outputs = model(**fwd)

            ce_loss = outputs.loss
            if ce_loss is None:
                if backend == "internvl3":
                    raise RuntimeError("Model did not return a loss. Check dataset labels.")
                ce_loss = torch.tensor(0.0, device=device)

            attn_loss = torch.zeros((), device=device)
            if is_stage1:
                attn_loss = _stage1_attn_loss(config, outputs, batch, device)

            loss = ce_loss + (config.stage1_attn_loss_weight * attn_loss if is_stage1 else 0.0)
            micro_loss = loss / grad_accum

            _finite = torch.tensor(float(torch.isfinite(loss)), device=device)
            if dist.is_initialized():
                dist.all_reduce(_finite, op=dist.ReduceOp.MIN)
            if _finite.item() < 0.5:
                log(f"[WARNING] Non-finite loss={loss.item():.4f} at global_step={global_step}, skipping batch")
                if not use_deepspeed:
                    optimizer.zero_grad(set_to_none=True)
                global_step += 1
                continue

            if use_deepspeed:
                ds_engine.backward(micro_loss)
                if (step + 1) % grad_accum == 0:
                    ds_engine.step()
                    scheduler.step()
                    update_step += 1
            else:
                micro_loss.backward()
                if (step + 1) % grad_accum == 0:
                    trainable_params = [p for p in model.parameters() if p.requires_grad]
                    _grad_finite = torch.tensor(1.0, device=device)
                    for p in trainable_params:
                        if p.grad is not None and not torch.isfinite(p.grad).all():
                            _grad_finite.fill_(0.0)
                            break
                    if dist.is_initialized():
                        dist.all_reduce(_grad_finite, op=dist.ReduceOp.MIN)
                    if _grad_finite.item() < 0.5:
                        log(f"[WARNING] Non-finite gradient at global_step={global_step}, skipping optimizer step")
                        optimizer.zero_grad(set_to_none=True)
                    else:
                        torch.nn.utils.clip_grad_norm_(trainable_params, config.max_grad_norm)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        update_step += 1

            epoch_loss += loss.item()
            epoch_ce += ce_loss.item()
            epoch_attn += attn_loss.item()
            global_step += 1

            lr = scheduler.get_last_lr()[0] if update_step > 0 else optimizer.param_groups[0]["lr"]
            metrics = {
                f"{pfx}/train/loss": loss.item(),
                f"{pfx}/train/ce_loss": ce_loss.item(),
                f"{pfx}/train/lr": lr,
            }
            if is_stage1:
                metrics[f"{pfx}/train/attn_loss"] = attn_loss.item()
            _log_wandb(metrics, global_step)

            if global_step % 50 == 0:
                msg = f"Epoch {epoch + 1} Step {global_step}: loss={loss.item():.4f} ce={ce_loss.item():.4f}"
                if is_stage1:
                    msg += f" attn={attn_loss.item():.4f}"
                msg += f" lr={lr:.2e}"
                log("  " + msg)

        num_steps = max(1, len(train_loader))
        summary = {
            f"{pfx}/epoch/loss": epoch_loss / num_steps,
            f"{pfx}/epoch/ce_loss": epoch_ce / num_steps,
        }
        if is_stage1:
            summary[f"{pfx}/epoch/attn_loss"] = epoch_attn / num_steps
        _log_wandb(summary, global_step)

        log(
            f"Epoch {epoch + 1}/{n_epochs}: loss={epoch_loss / num_steps:.4f} ce={epoch_ce / num_steps:.4f}"
            + (f" attn={epoch_attn / num_steps:.4f}" if is_stage1 else "")
        )

        _save_checkpoint(
            model,
            save_dir=os.path.join(config.output_dir, f"stage{stage}", f"epoch_{epoch + 1}"),
            save_lora=not is_stage1,
        )

    if dist.is_initialized():
        dist.destroy_process_group()
