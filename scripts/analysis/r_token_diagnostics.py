"""Diagnose whether R-token spatial information survives the LM up-projection.

This script collects grounding examples, compares linear bbox probes trained on
``R_bottleneck`` vs. ``W_up(R_bottleneck)``, and optionally measures how much
decoder self-attention answer tokens pay to the inserted R-token block.

Example
-------
PYTHONPATH=. python scripts/analysis/r_token_diagnostics.py \
  --backend internvl3 \
  --checkpoint_dir models/internvl3/s2_internvl_notile/stage2/epoch_4 \
  --lora_checkpoint_dir models/internvl3/s2_internvl_notile/stage2/epoch_4 \
  --stage1_checkpoint models/internvl3/stage1/epoch_4/reinspection_module.pt \
  --split val \
  --max_samples 128 \
  --output_file outputs/r_token_diagnostics/notile_e4_refcoco_val.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
import yaml
from torch.utils.data import DataLoader, Subset

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import ReInspectionConfig
from src.data.refcoco import RefCOCODataset
from src.eval_checkpoints import find_lora_weights_path, find_reinspection_module_path
from src.model.bbox_head import BboxHead
from src.training.trainer import _train_forward_kwargs, collate_fn
from src.utils.r_token_diagnostics import (
    attention_mass_to_key_mask,
    bbox_metrics,
    fit_ridge_probe,
    r_text_cosine_stats,
    r_token_mask,
    shifted_answer_mask,
    summarize,
)


def _repo_root() -> Path:
    return _REPO_ROOT


def _config_from_backend_yaml(backend: str, overrides: Dict) -> ReInspectionConfig:
    yaml_path = _repo_root() / "src" / "configs" / "backend" / f"{backend}.yaml"
    field_names = {f.name for f in dataclasses.fields(ReInspectionConfig)}
    kw: Dict = {}
    if yaml_path.exists():
        with open(yaml_path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        for key, value in raw.items():
            if key == "processor_path":
                kw["processor_name_or_path"] = value
            elif key in field_names:
                kw[key] = value
    kw.update({k: v for k, v in overrides.items() if v is not None})
    return ReInspectionConfig(**kw)


def _load_processor(backend: str, config: ReInspectionConfig):
    if backend == "internvl3":
        from src.backends.internvl3 import load_processor

        return load_processor(config)
    if backend == "llava_next":
        from src.backends.llava_next import load_processor

        return load_processor(config)
    if backend == "gemma4":
        from src.backends.gemma4 import load_processor

        return load_processor(config)
    if backend == "qwen25vl":
        from transformers import AutoProcessor

        from src.backends.hf_hub_utils import resolve_pretrained_local_path

        return AutoProcessor.from_pretrained(
            resolve_pretrained_local_path(config.processor_name_or_path or config.model_name_or_path)
        )
    raise ValueError(f"Unsupported backend: {backend}")


def _to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _candidate_bbox_head_paths(args, config: ReInspectionConfig) -> List[Path]:
    if args.bbox_head_path and args.bbox_head_path != "auto":
        return [Path(args.bbox_head_path)]

    candidates: List[Path] = []
    for root in (args.checkpoint_dir, args.lora_checkpoint_dir):
        if root:
            candidates.append(Path(root) / "bbox_head.pt")
    stage1 = args.stage1_checkpoint or config.stage1_checkpoint
    if stage1:
        p = Path(stage1)
        candidates.append((p if p.is_dir() else p.parent) / "bbox_head.pt")
    return candidates


def _load_bbox_head(args, config: ReInspectionConfig, device: torch.device) -> Optional[BboxHead]:
    if args.bbox_head_path == "":
        return None
    for path in _candidate_bbox_head_paths(args, config):
        if not path.exists():
            continue
        head = BboxHead(
            config.d_bottleneck,
            dtype=torch.float32,
            head_type=config.stage1_bbox_head_type,
        )
        state = torch.load(path, map_location="cpu", weights_only=True)
        head.load_state_dict(state)
        head.to(device)
        head.eval()
        print(f"[diagnostic] loaded bbox head: {path}", flush=True)
        return head
    print("[diagnostic] no bbox_head.pt found; skipping trained bbox-head metric", flush=True)
    return None


def _load_reinspection_model(args, config: ReInspectionConfig, processor, attn_implementation: Optional[str]):
    """Load only the requested backend to avoid importing unavailable backends."""
    if args.backend == "internvl3":
        from src.backends.internvl3 import load_model

        load_kwargs = {"device_map": "auto", "processor": processor}
    elif args.backend == "llava_next":
        from src.backends.llava_next import load_model

        load_kwargs = {"device_map": "auto", "processor": processor}
    elif args.backend == "gemma4":
        from src.backends.gemma4 import load_model

        load_kwargs = {"device_map": "auto", "processor": processor}
    elif args.backend == "qwen25vl":
        from src.backends.qwen25vl import load_model

        load_kwargs = {"device_map": "auto"}
    else:
        raise ValueError(f"Unsupported backend: {args.backend}")

    if attn_implementation is not None:
        load_kwargs["attn_implementation"] = attn_implementation
    model = load_model(config, **load_kwargs)

    ri_path = find_reinspection_module_path(args.checkpoint_dir, args.lora_checkpoint_dir)
    if ri_path is None:
        raise FileNotFoundError("Could not find reinspection_module.pt in checkpoint/lora dirs")
    state = torch.load(ri_path, map_location="cpu", weights_only=True)
    model.reinspection.load_state_dict(state)
    model.reinspection.to(model.device)
    print(f"[diagnostic] loaded reinspection module: {ri_path}", flush=True)

    lora_path = find_lora_weights_path(
        args.checkpoint_dir,
        args.lora_checkpoint_dir,
        search_order=(args.checkpoint_dir, args.lora_checkpoint_dir),
    )
    if lora_path is not None:
        from peft import PeftModel

        model.base_model.model.language_model = PeftModel.from_pretrained(
            model.base_model.model.language_model,
            lora_path,
        )
    print(f"[diagnostic] loaded LoRA: {lora_path}", flush=True)
    return model


def _feature_probe_report(x: torch.Tensor, y: torch.Tensor, train_frac: float, l2: float, seed: int) -> dict:
    n = x.shape[0]
    if n < 4:
        return {"skipped": f"need at least 4 samples, got {n}"}
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator)
    n_train = int(round(n * train_frac))
    n_train = max(2, min(n - 1, n_train))
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    probe = fit_ridge_probe(x[train_idx], y[train_idx], l2=l2)
    pred_train = probe.predict(x[train_idx])
    pred_test = probe.predict(x[test_idx])
    return {
        "train": bbox_metrics(pred_train, y[train_idx]),
        "test": bbox_metrics(pred_test, y[test_idx]),
        "n_train": int(n_train),
        "n_test": int(test_idx.numel()),
        "l2": float(l2),
    }


def _mean_pool_queries(r_bottleneck: torch.Tensor) -> torch.Tensor:
    return r_bottleneck.float().mean(dim=1)


def _up_project_queries(model, r_bottleneck: torch.Tensor) -> torch.Tensor:
    weight_dtype = model.reinspection.W_up.weight.dtype
    return model.reinspection.W_up(r_bottleneck.to(weight_dtype)).float().mean(dim=1)


def _normal_text_mask(model, batch: dict) -> torch.BoolTensor:
    mask = batch["attention_mask"].bool()
    image_token_id = getattr(model, "_image_token_id", None)
    if image_token_id is not None:
        mask = mask & (batch["input_ids"] != int(image_token_id))
    return mask


def collect(args) -> dict:
    overrides = {
        "backend": args.backend,
        "data_root": args.data_root,
        "checkpoint_dir": args.checkpoint_dir,
        "lora_checkpoint_dir": args.lora_checkpoint_dir,
        "stage1_checkpoint": args.stage1_checkpoint,
        "attn_implementation": "eager" if args.decoder_attention_samples > 0 else args.attn_implementation,
        "bf16": not args.fp32,
    }
    config = _config_from_backend_yaml(args.backend, overrides)
    processor = _load_processor(args.backend, config)

    model = _load_reinspection_model(
        args,
        config,
        processor,
        attn_implementation="eager" if args.decoder_attention_samples > 0 else args.attn_implementation,
    )
    model.eval()
    device = next(model.parameters()).device
    bbox_head = _load_bbox_head(args, config, device)

    dataset_names = [x.strip() for x in args.dataset_names.split(",") if x.strip()] or None
    ds = RefCOCODataset(
        data_root=args.data_root,
        processor=processor,
        backend=args.backend,
        split=args.split,
        dataset_names=dataset_names,
        max_pixels=config.max_pixels,
        min_pixels=config.min_pixels,
        crop_to_patches=config.crop_to_patches_stage1,
        system_prompt=config.system_prompt,
        answer_ignore_index=config.answer_ignore_index,
        coco_images_dir=config.coco_images_dir,
    )
    if len(ds) == 0:
        raise RuntimeError(f"No RefCOCO samples found for split={args.split!r}, datasets={dataset_names}")

    indices = list(range(len(ds)))
    random.Random(args.seed).shuffle(indices)
    if args.max_samples > 0:
        indices = indices[: args.max_samples]
    loader = DataLoader(
        Subset(ds, indices),
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    feats_bottleneck: List[torch.Tensor] = []
    feats_lmspace: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    trained_head_preds: List[torch.Tensor] = []

    r_norms: List[float] = []
    text_norms: List[float] = []
    r_text_norm_ratios: List[float] = []
    r_text_cos_mean: List[float] = []
    r_text_cos_max: List[float] = []
    attn_by_layer: List[List[float]] = []
    attention_batches = 0

    seen = 0
    with torch.no_grad():
        for step, raw_batch in enumerate(loader):
            batch = _to_device(raw_batch, device)
            capture_attn = seen < args.decoder_attention_samples
            fwd = _train_forward_kwargs(
                batch,
                args.backend,
                is_stage1=True,
                return_attn_maps=True,
                use_lm_ce=True,
            )
            fwd["output_attentions"] = bool(capture_attn)
            fwd["return_dict"] = True
            outputs = model(**fwd)
            if outputs.R_bottleneck is None:
                raise RuntimeError("Model did not return R_bottleneck; return_attn_maps path is broken.")

            r_b = outputs.R_bottleneck.detach()
            feats_bottleneck.append(_mean_pool_queries(r_b).cpu())
            feats_lmspace.append(_up_project_queries(model, r_b).cpu())
            targets.append(batch["bbox_norm"].detach().float().cpu())

            if bbox_head is not None:
                trained_head_preds.append(bbox_head(r_b).detach().cpu())

            r_up_tokens = model.reinspection.W_up(r_b.to(model.reinspection.W_up.weight.dtype)).float()
            token_embed = model.base_model.get_input_embeddings()(batch["input_ids"]).float()
            text_mask = _normal_text_mask(model, batch)

            r_norm_batch = r_up_tokens.norm(dim=-1)
            text_norm_batch = token_embed.norm(dim=-1)
            r_norms.extend(r_norm_batch.flatten().detach().cpu().tolist())
            text_norms.extend(text_norm_batch[text_mask].detach().cpu().tolist())
            cos = r_text_cosine_stats(r_up_tokens, token_embed, text_mask)
            r_text_cos_mean.extend(cos["mean"])
            r_text_cos_max.extend(cos["max"])
            for b in range(r_up_tokens.shape[0]):
                valid_text_norms = text_norm_batch[b, text_mask[b]]
                if valid_text_norms.numel() == 0:
                    continue
                ratio = r_norm_batch[b].median() / valid_text_norms.median().clamp(min=1e-8)
                r_text_norm_ratios.append(float(ratio.item()))

            if capture_attn and outputs.attentions is not None:
                insert_positions = model._find_insert_positions(
                    batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch["labels"],
                )
                query_mask = shifted_answer_mask(
                    batch["labels"],
                    insert_positions,
                    config.n_queries,
                    ignore_index=config.answer_ignore_index,
                )
                key_mask = r_token_mask(
                    batch["labels"].shape[0],
                    batch["labels"].shape[1],
                    insert_positions,
                    config.n_queries,
                )
                masses = attention_mass_to_key_mask(outputs.attentions, query_mask, key_mask)
                if masses:
                    while len(attn_by_layer) < len(masses):
                        attn_by_layer.append([])
                    for i, value in enumerate(masses):
                        attn_by_layer[i].append(value)
                    attention_batches += 1

            seen += batch["input_ids"].shape[0]
            if args.log_every > 0 and (step + 1) % args.log_every == 0:
                print(f"[diagnostic] processed {seen}/{len(indices)}", flush=True)

    x_b = torch.cat(feats_bottleneck, dim=0)
    x_lm = torch.cat(feats_lmspace, dim=0)
    y = torch.cat(targets, dim=0)

    report = {
        "config": {
            "backend": args.backend,
            "checkpoint_dir": args.checkpoint_dir,
            "lora_checkpoint_dir": args.lora_checkpoint_dir,
            "stage1_checkpoint": args.stage1_checkpoint,
            "split": args.split,
            "dataset_names": dataset_names,
            "n_samples": int(y.shape[0]),
            "n_queries": int(config.n_queries),
            "d_bottleneck": int(config.d_bottleneck),
            "d_model": int(config.d_model),
        },
        "bbox_linear_probe": {
            "bottleneck_query_mean": _feature_probe_report(
                x_b, y, args.probe_train_frac, args.probe_l2, args.seed
            ),
            "lmspace_query_mean_after_W_up": _feature_probe_report(
                x_lm, y, args.probe_train_frac, args.probe_l2, args.seed
            ),
        },
        "r_token_embedding": {
            "r_up_norm": summarize(r_norms),
            "normal_text_embedding_norm": summarize(text_norms),
            "per_sample_r_median_over_text_median_norm": summarize(r_text_norm_ratios),
            "r_to_text_cosine_mean": summarize(r_text_cos_mean),
            "r_to_text_cosine_max_per_r_mean": summarize(r_text_cos_max),
        },
    }

    if trained_head_preds:
        pred = torch.cat(trained_head_preds, dim=0)
        report["trained_bbox_head_on_bottleneck"] = bbox_metrics(pred, y)

    if attn_by_layer:
        layer_means = [float(torch.tensor(vals).mean().item()) for vals in attn_by_layer]
        report["answer_attention_to_r"] = {
            "n_batches": int(attention_batches),
            "layer_mean": layer_means,
            "all_layers_mean": float(torch.tensor(layer_means).mean().item()),
            "first_layer": layer_means[0],
            "middle_layer": layer_means[len(layer_means) // 2],
            "last_layer": layer_means[-1],
        }
    else:
        report["answer_attention_to_r"] = {
            "skipped": "no attentions captured; set --decoder_attention_samples > 0 and use eager attention"
        }

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", default="internvl3", choices=["internvl3", "llava_next", "gemma4", "qwen25vl"])
    parser.add_argument("--checkpoint_dir", required=True, help="Checkpoint dir with reinspection_module.pt.")
    parser.add_argument("--lora_checkpoint_dir", default=None, help="Optional Stage-2 dir with LoRA and paired module.")
    parser.add_argument("--stage1_checkpoint", default=None, help="Optional Stage-1 module path/dir for bbox_head auto lookup.")
    parser.add_argument("--bbox_head_path", default="auto", help="'auto', explicit bbox_head.pt, or empty string to disable.")
    parser.add_argument("--data_root", default="/data/datasets")
    parser.add_argument("--split", default="val")
    parser.add_argument("--dataset_names", default="refcoco,refcoco+,refcocog")
    parser.add_argument("--max_samples", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--probe_train_frac", type=float, default=0.7)
    parser.add_argument("--probe_l2", type=float, default=1e-3)
    parser.add_argument("--decoder_attention_samples", type=int, default=16)
    parser.add_argument("--attn_implementation", default=None)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--output_file", default="outputs/r_token_diagnostics/report.json")
    args = parser.parse_args()

    report = collect(args)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"[diagnostic] wrote {output_path}", flush=True)
    for name, metrics in report["bbox_linear_probe"].items():
        test = metrics.get("test", {})
        if test:
            print(
                f"[diagnostic] {name}: test IoU={test.get('iou_mean', float('nan')):.4f} "
                f"IoU@0.5={test.get('iou_at_0_5', float('nan')):.4f} L1={test.get('l1', float('nan')):.4f}",
                flush=True,
            )
    if "answer_attention_to_r" in report and "all_layers_mean" in report["answer_attention_to_r"]:
        print(
            "[diagnostic] answer attention to R: "
            f"all_layers={report['answer_attention_to_r']['all_layers_mean']:.4f} "
            f"last={report['answer_attention_to_r']['last_layer']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
