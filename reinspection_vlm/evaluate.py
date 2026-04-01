"""Unified spatial-benchmark evaluation for Qwen3-VL and InternVL3 backends."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import asdict
from typing import Dict, List, Optional

import numpy as np
import torch
from peft import PeftModel
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, InternVLForConditionalGeneration, Qwen3VLForConditionalGeneration

from reinspection_vlm.config import ReInspectionConfig
from reinspection_vlm.data.chat_template import build_chat_messages as intern_build_chat
from reinspection_vlm.data.spatial_dataset import SpatialVQADataset
from reinspection_vlm.data.utils import build_chat_messages as qwen_build_chat
from reinspection_vlm.train_common import load_config

BENCHMARK_CONFIGS = {
    "vsr": {"data_file": "vsr/test.jsonl", "image_root": "vsr/images"},
    "whatsup": {"data_file": "whatsup/test.json", "image_root": "whatsup/images"},
    "gqa_spatial": {"data_file": "gqa_spatial/test.json", "image_root": "gqa_spatial/images"},
    "spatialbench": {"data_file": "spatialbench/test.json", "image_root": "spatialbench/images"},
}


def _init_wandb(args, config: ReInspectionConfig) -> None:
    try:
        import wandb

        os.environ["WANDB_SILENT"] = "true"
        resolved = asdict(config)
        resolved["_cli"] = {k: v for k, v in vars(args).items()}
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=resolved,
            job_type="eval",
        )
    except Exception:
        pass


def _log_wandb(metrics: dict, step: int = 0) -> None:
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except Exception:
        pass


def _finish_wandb() -> None:
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass


def _log_wandb_summary(all_results, prefix: str) -> None:
    try:
        import wandb

        if wandb.run is None:
            return
        columns = ["condition", "benchmark", "accuracy", "correct", "total", "attention_entropy"]
        table = wandb.Table(columns=columns)
        for r in all_results:
            table.add_data(
                r["condition"],
                r["benchmark"],
                r["accuracy"],
                r["correct"],
                r["total"],
                r.get("mean_attention_entropy"),
            )
        wandb.log({f"{prefix}/summary": table})
        cond_accs = defaultdict(list)
        for r in all_results:
            cond_accs[r["condition"]].append(r["accuracy"])
        for cond, accs in cond_accs.items():
            wandb.run.summary[f"{prefix}/{cond}/mean_accuracy"] = float(np.mean(accs))
    except Exception:
        pass


def normalize_answer(text: str) -> str:
    return " ".join(text.strip().lower().split())


def compute_attention_entropy(attn_vis: torch.Tensor) -> float:
    avg_attn = attn_vis.mean(dim=1).clamp(min=1e-8)
    avg_attn = avg_attn / avg_attn.sum(dim=-1, keepdim=True)
    entropy = -(avg_attn * avg_attn.log()).sum(dim=-1)
    return entropy.mean().item()


def model_device(model) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def load_condition_model(
    backend: str,
    condition: str,
    config: ReInspectionConfig,
    processor,
    checkpoint_dir: Optional[str] = None,
    lora_checkpoint_dir: Optional[str] = None,
):
    if backend == "qwen3vl":
        from reinspection_vlm.backends.qwen3vl import load_model as load_qwen_ri

        if condition == "reinspection":
            model = load_qwen_ri(config, device_map="auto")
            if checkpoint_dir:
                path = os.path.join(checkpoint_dir, "reinspection_module.pt")
                if os.path.exists(path):
                    state_dict = torch.load(path, map_location="cpu", weights_only=True)
                    model.reinspection.load_state_dict(state_dict)
                model.reinspection.to(model.device)
                lora_path = os.path.join(checkpoint_dir, "lora_weights")
                if os.path.exists(lora_path):
                    model.base_model.model.language_model = PeftModel.from_pretrained(
                        model.base_model.model.language_model, lora_path
                    )
            return model, True
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            config.model_name_or_path,
            torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
            device_map="auto",
        )
        if condition == "lora_only":
            ckpt = lora_checkpoint_dir or checkpoint_dir
            if ckpt is None:
                raise ValueError("lora_only requires --checkpoint_dir or --lora_checkpoint_dir")
            lora_path = os.path.join(ckpt, "lora_weights")
            if not os.path.exists(lora_path):
                raise FileNotFoundError(f"LoRA weights not found: {lora_path}")
            model.model.language_model = PeftModel.from_pretrained(
                model.model.language_model, lora_path
            )
        return model, False

    from reinspection_vlm.backends.internvl3 import load_model as load_intern_ri

    if condition == "reinspection":
        model = load_intern_ri(config, device_map="auto", processor=processor)
        if checkpoint_dir:
            path = os.path.join(checkpoint_dir, "reinspection_module.pt")
            if os.path.exists(path):
                state_dict = torch.load(path, map_location="cpu", weights_only=True)
                model.reinspection.load_state_dict(state_dict)
                dev = next(model.base_model.parameters()).device
                model.reinspection.to(dev)
            lora_path = os.path.join(checkpoint_dir, "lora_weights")
            if os.path.exists(lora_path):
                model.base_model.model.language_model = PeftModel.from_pretrained(
                    model.base_model.model.language_model, lora_path
                )
        return model, True

    model = InternVLForConditionalGeneration.from_pretrained(
        config.model_name_or_path,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        device_map="auto",
    )
    if condition == "lora_only":
        ckpt = lora_checkpoint_dir or checkpoint_dir
        if ckpt is None:
            raise ValueError("lora_only evaluation requires --checkpoint_dir or --lora_checkpoint_dir")
        lora_path = os.path.join(ckpt, "lora_weights")
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA weights not found: {lora_path}")
        model.model.language_model = PeftModel.from_pretrained(model.model.language_model, lora_path)
    return model, False


@torch.no_grad()
def evaluate_benchmark(
    backend: str,
    model,
    processor,
    data_file: str,
    image_root: str,
    benchmark_name: str,
    config: ReInspectionConfig,
    is_reinspection: bool,
    max_samples: int = -1,
) -> dict:
    ds = SpatialVQADataset(
        data_file=data_file,
        image_root=image_root,
        processor=processor,
        backend=backend,
        split="test",
        max_pixels=config.max_pixels,
        min_pixels=config.min_pixels,
        crop_to_patches=config.crop_to_patches_stage2,
        system_prompt=config.system_prompt,
        answer_ignore_index=config.answer_ignore_index,
    )
    n = len(ds)
    if max_samples > 0:
        n = min(n, max_samples)

    correct = 0
    total = 0
    entropies: List[float] = []
    sample_outputs = []
    skipped = 0
    device = model_device(model)
    model.eval()

    from reinspection_vlm.backends.internvl3 import InternVL3WithReInspection

    for i in tqdm(range(n), desc=benchmark_name):
        sample = ds.samples[i]
        gt_answer = sample["answer"]
        question = sample["question"]
        image_path = os.path.join(image_root, sample["image"])

        if not os.path.isfile(image_path):
            skipped += 1
            continue

        try:
            if backend == "qwen3vl":
                messages = qwen_build_chat(question, image_path=image_path)
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(
                    text=[text],
                    images=[image_path],
                    return_tensors="pt",
                    max_pixels=config.max_pixels,
                    min_pixels=config.min_pixels,
                )
            else:
                messages = intern_build_chat(
                    question=question,
                    image_path=image_path,
                    system_prompt=config.system_prompt,
                )
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(
                    text=[text],
                    images=[image_path],
                    return_tensors="pt",
                    max_pixels=config.max_pixels,
                    min_pixels=config.min_pixels,
                    crop_to_patches=config.crop_to_patches_stage2,
                )
        except Exception as e:
            skipped += 1
            tqdm.write(f"  skip {i}: {e}")
            continue

        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        generated_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)

        if backend == "qwen3vl":
            if is_reinspection:
                input_len = inputs["input_ids"].shape[1] + config.n_queries
            else:
                input_len = inputs["input_ids"].shape[1]
        else:
            if isinstance(model, InternVL3WithReInspection) and model.last_generation_prompt_lengths is not None:
                input_len = int(model.last_generation_prompt_lengths[0].item())
            else:
                input_len = int(inputs["input_ids"].shape[1])

        generated_text = processor.batch_decode(
            generated_ids[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip().lower()

        gt_norm = normalize_answer(gt_answer)
        gen_norm = normalize_answer(generated_text)
        is_ok = gen_norm == gt_norm or gt_norm in gen_norm
        if is_ok:
            correct += 1
        total += 1
        sample_outputs.append({
            "idx": i,
            "image": sample["image"],
            "question": question,
            "ground_truth": gt_answer,
            "model_output": generated_text,
            "correct": is_ok,
        })

        if is_reinspection and hasattr(model, "get_attention_maps"):
            _, attn_vis = model.get_attention_maps()
            if attn_vis is not None:
                entropies.append(compute_attention_entropy(attn_vis))

    acc = correct / total if total else 0.0
    mean_ent = float(np.mean(entropies)) if entropies else None
    print(f"  {benchmark_name}: acc={acc:.4f} ({correct}/{total})")
    return {
        "benchmark": benchmark_name,
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "skipped": skipped,
        "mean_attention_entropy": mean_ent,
        "samples": sample_outputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate spatial reasoning benchmarks")
    parser.add_argument("--backend", type=str, required=True, choices=["qwen3vl", "internvl3"])
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--lora_checkpoint_dir", type=str, default=None)
    parser.add_argument("--output_file", type=str, default="eval_results.json")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["vsr", "whatsup", "gqa_spatial", "spatialbench"],
    )
    parser.add_argument("--condition", type=str, default="reinspection",
                        choices=["frozen", "lora_only", "reinspection"])
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="reinspection-vlm")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    args = parser.parse_args()

    backend = args.backend
    config = ReInspectionConfig()
    if args.config:
        config = load_config(args.config, output_dir="outputs")

    processor = None
    if backend == "internvl3":
        from reinspection_vlm.backends.internvl3 import load_processor

        processor = load_processor(config)
    else:
        processor = AutoProcessor.from_pretrained(
            config.model_name_or_path,
            max_pixels=config.max_pixels,
            min_pixels=config.min_pixels,
        )

    _init_wandb(args, config)
    conditions = [args.condition]
    if args.compare:
        conditions = ["frozen", "lora_only", "reinspection"]

    prefix = f"{backend}_eval"
    all_results = []
    wandb_step = 0

    for condition in conditions:
        if condition == "lora_only":
            ckpt = args.lora_checkpoint_dir or args.checkpoint_dir
            if ckpt is None:
                print(f"Skipping {condition}: no checkpoint directory")
                continue
            if not os.path.exists(os.path.join(ckpt, "lora_weights")):
                print(f"Skipping {condition}: lora_weights not found")
                continue
        if condition == "reinspection" and args.checkpoint_dir is None:
            print(f"Skipping {condition}: no --checkpoint_dir")
            continue

        print(f"\n{'=' * 60}\nCondition: {condition}\n{'=' * 60}")
        model, is_ri = load_condition_model(
            backend, condition, config, processor,
            checkpoint_dir=args.checkpoint_dir,
            lora_checkpoint_dir=args.lora_checkpoint_dir,
        )
        model.eval()

        for bm in args.benchmarks:
            if bm not in BENCHMARK_CONFIGS:
                continue
            rel = BENCHMARK_CONFIGS[bm]
            data_file = os.path.join(args.data_root, rel["data_file"])
            image_root = os.path.join(args.data_root, rel["image_root"])
            if not os.path.exists(data_file):
                print(f"Missing {data_file}, skip")
                continue
            result = evaluate_benchmark(
                backend, model, processor,
                data_file=data_file,
                image_root=image_root,
                benchmark_name=bm,
                config=config,
                is_reinspection=is_ri,
                max_samples=args.max_samples,
            )
            result["condition"] = condition
            all_results.append(result)
            m = {
                f"{prefix}/{condition}/{bm}/accuracy": result["accuracy"],
                f"{prefix}/{condition}/{bm}/correct": result["correct"],
                f"{prefix}/{condition}/{bm}/total": result["total"],
            }
            if result["mean_attention_entropy"] is not None:
                m[f"{prefix}/{condition}/{bm}/attention_entropy"] = result["mean_attention_entropy"]
            _log_wandb(m, wandb_step)
            wandb_step += 1

        del model
        torch.cuda.empty_cache()

    _log_wandb_summary(all_results, prefix)

    out_dir = os.path.dirname(args.output_file) or "."
    for result in all_results:
        samples = result.pop("samples", [])
        if samples:
            sf = os.path.join(out_dir, f"{result['condition']}_{result['benchmark']}_samples.json")
            with open(sf, "w", encoding="utf-8") as f:
                json.dump(samples, f, indent=2)

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output_file}")
    _finish_wandb()


if __name__ == "__main__":
    main()
