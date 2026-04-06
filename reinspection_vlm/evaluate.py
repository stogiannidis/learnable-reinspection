"""Unified spatial-benchmark evaluation for Qwen3-VL and InternVL3 backends.

Entry point is Hydra-only: ``python -m reinspection_vlm.evaluate stage=eval [overrides]``.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from peft import PeftModel
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, InternVLForConditionalGeneration, Qwen3VLForConditionalGeneration

import hydra
from omegaconf import DictConfig, OmegaConf

from reinspection_vlm.config import ReInspectionConfig
from reinspection_vlm.data.chat_template import build_chat_messages as intern_build_chat
from reinspection_vlm.data.spatial_dataset import SpatialVQADataset
from reinspection_vlm.data.utils import build_chat_messages as qwen_build_chat
from reinspection_vlm.hydra_util import strip_deepspeed_local_rank_argv
from reinspection_vlm.train_common import _env_info, _git_info

strip_deepspeed_local_rank_argv()

# Silence noisy third-party loggers (pad_token_id spam is WARNING from transformers.generation.utils)
for _logger_name in ("httpx", "httpcore", "urllib3"):
    logging.getLogger(_logger_name).setLevel(logging.WARNING)
for _logger_name in ("transformers.generation", "transformers.generation.utils"):
    logging.getLogger(_logger_name).setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*pad_token_id.*eos_token_id.*")
warnings.filterwarnings("ignore", message=".*not a valid argument for this processor.*")

BENCHMARK_CONFIGS = {
    "vsr": {"data_file": "vsr/test.jsonl", "image_root": "vsr/images"},
    "whatsup": {"data_file": "whatsup/test.json", "image_root": "whatsup/images"},
    "gqa_spatial": {"data_file": "gqa_spatial/test.json", "image_root": "gqa_spatial/images"},
    "spatialbench": {"data_file": "spatialbench/test.json", "image_root": "spatialbench/images"},
    "3dsrbench": {"data_file": "3dsrbench/test.json", "image_root": "3dsrbench/images"},
    "mindcube": {"data_file": "mindcube/test.json", "image_root": "mindcube/images"},
    "blink": {"data_file": "blink/test.json", "image_root": "blink/images"},
    "srbench": {"data_file": "srbench/test.json", "image_root": "srbench/images"},
}

# tqdm refresh and explicit acc line (both avoid per-instance log spam when tee'd to a file).
EVAL_PROGRESS_LOG_INTERVAL = 200


def _eval_tqdm_disable() -> bool:
    """Skip tqdm bar when stderr is not a TTY (pipes, ``tee``, log files).

    In those cases each refresh becomes a new log line (~once/sec with default
    dynamic miniters). Rely on ``print`` every ``EVAL_PROGRESS_LOG_INTERVAL`` instead.
    """
    return not sys.stderr.isatty()


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _generate_extra_kw(processor) -> Dict[str, int]:
    """Avoid per-step ``Setting pad_token_id to eos_token_id`` logs from ``model.generate``."""
    tok = getattr(processor, "tokenizer", None)
    if tok is None:
        return {}
    pad = tok.pad_token_id
    if pad is None:
        pad = tok.eos_token_id
    if pad is None:
        return {}
    out: Dict[str, int] = {"pad_token_id": pad}
    if tok.eos_token_id is not None:
        out["eos_token_id"] = tok.eos_token_id
    return out


def _resolve_path(path_like: str, base_dir: Optional[Path] = None) -> str:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return str(path)
    base = base_dir or Path.cwd()
    return str((base / path).resolve())


def _resolve_data_path(path_like: str, data_root: str) -> str:
    """Resolve a data path with sensible fallbacks."""
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return str(path)

    candidates = [
        Path(data_root) / path,
        _repo_root() / path,
        Path.cwd() / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    # Default to data_root-relative even if missing (for clear errors/logging).
    return str(candidates[0].resolve())


def _init_wandb(config: ReInspectionConfig) -> None:
    try:
        import wandb

        os.environ["WANDB_SILENT"] = "true"
        resolved = asdict(config)
        resolved["_git"] = _git_info()
        resolved["_env"] = _env_info()

        tags = [config.backend, "eval"]
        if config.eval_condition:
            tags.append(config.eval_condition)
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=resolved,
            job_type="eval",
            tags=tags,
            save_code=True,
        )

        if config.checkpoint_dir and os.path.isdir(config.checkpoint_dir):
            art = wandb.Artifact(
                f"eval-checkpoint-{config.backend}",
                type="model",
                metadata={
                    "checkpoint_dir": config.checkpoint_dir,
                    "backend": config.backend,
                },
            )
            art.add_dir(config.checkpoint_dir)
            wandb.log_artifact(art)
    except Exception:
        pass


def _log_wandb(metrics: dict, step: int = 0) -> None:
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except Exception:
        pass


def _log_eval_results_artifact(output_file: str, all_results: list, backend: str) -> None:
    """Log evaluation results JSON as a W&B artifact."""
    try:
        import wandb

        if wandb.run is None:
            return
        art = wandb.Artifact(
            f"eval-results-{backend}",
            type="eval-results",
            metadata={
                "num_benchmarks": len(all_results),
                "conditions": list({r["condition"] for r in all_results}),
            },
        )
        if os.path.exists(output_file):
            art.add_file(output_file)

        out_dir = os.path.dirname(output_file) or "."
        for r in all_results:
            sf = os.path.join(out_dir, f"{r['condition']}_{r['benchmark']}_samples.json")
            if os.path.exists(sf):
                art.add_file(sf)
        wandb.log_artifact(art)
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


def _extract_mcq_letter(text: str) -> Optional[str]:
    """Extract a single MCQ letter (A-D) from model output, if present."""
    import re

    text = text.strip()
    # Exact single letter
    if text.upper() in {"A", "B", "C", "D"}:
        return text.upper()
    # "(A)" style
    m = re.match(r"^\(?([A-Da-d])\)?[\.\s:]*$", text)
    if m:
        return m.group(1).upper()
    # Leading letter: "A. baseball glove" or "A) ..."
    m = re.match(r"^\(?([A-Da-d])\)?[\.\)\s:]", text)
    if m:
        return m.group(1).upper()
    return None


def match_answer(generated: str, ground_truth: str) -> bool:
    """Match generated answer against ground truth, with MCQ-aware logic."""
    gen_norm = normalize_answer(generated)
    gt_norm = normalize_answer(ground_truth)

    # Direct match or substring
    if gen_norm == gt_norm or gt_norm in gen_norm:
        return True

    # MCQ letter matching: if ground truth is a single letter (A-D),
    # try to extract a letter from the generated text
    if gt_norm.upper() in {"A", "B", "C", "D"}:
        gen_letter = _extract_mcq_letter(generated)
        if gen_letter is not None:
            return gen_letter == gt_norm.upper()

    return False


def print_comparison_table(all_results: list, output_file: Optional[str] = None) -> None:
    """Print (and optionally save) a formatted accuracy table across conditions and benchmarks."""
    if not all_results:
        return

    # {condition: {benchmark: accuracy}}
    table: Dict[str, Dict[str, float]] = defaultdict(dict)
    for r in all_results:
        table[r["condition"]][r["benchmark"]] = r["accuracy"]

    conditions = list(dict.fromkeys(r["condition"] for r in all_results))
    benchmarks = list(dict.fromkeys(r["benchmark"] for r in all_results))

    bm_w = max(len(b) for b in benchmarks + ["Benchmark"])
    col_w = max(max(len(c) for c in conditions), 8)

    def fmt(v: Optional[float]) -> str:
        return f"{v:.1%}" if v is not None else "  —   "

    header = f"{'Benchmark':<{bm_w}}  | " + " | ".join(f"{c:^{col_w}}" for c in conditions)
    sep = "-" * (bm_w + 2) + "+" + "+".join("-" * (col_w + 2) for _ in conditions)

    lines = ["", "=" * len(header), "Comparison Table", "=" * len(header), header, sep]

    for bm in benchmarks:
        row = f"{bm:<{bm_w}}  | " + " | ".join(
            f"{fmt(table[c].get(bm)):^{col_w}}" for c in conditions
        )
        lines.append(row)

    lines.append(sep)

    means: Dict[str, Optional[float]] = {}
    for c in conditions:
        vals = [table[c][b] for b in benchmarks if b in table[c]]
        means[c] = float(np.mean(vals)) if vals else None

    lines.append(
        f"{'Mean':<{bm_w}}  | " + " | ".join(f"{fmt(means[c]):^{col_w}}" for c in conditions)
    )
    lines.append("=" * len(header))

    if "reinspection" in conditions and "frozen" in conditions:
        lines.append("\nDelta (reinspection - frozen):")
        for bm in benchmarks:
            ri = table["reinspection"].get(bm)
            fr = table["frozen"].get(bm)
            if ri is not None and fr is not None:
                d = ri - fr
                lines.append(f"  {bm:<{bm_w}} {'+'if d>=0 else ''}{d:.1%}")
        if means.get("reinspection") is not None and means.get("frozen") is not None:
            d = means["reinspection"] - means["frozen"]  # type: ignore[operator]
            lines.append(f"  {'Mean':<{bm_w}} {'+'if d>=0 else ''}{d:.1%}")

    lines.append("")
    table_str = "\n".join(lines)
    print(table_str)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(table_str + "\n")
        print(f"Comparison table saved to {output_file}")


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
                if not os.path.exists(lora_path) and lora_checkpoint_dir:
                    lora_path = os.path.join(lora_checkpoint_dir, "lora_weights")
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
            model.reinspection.to(model.device)
            lora_path = os.path.join(checkpoint_dir, "lora_weights")
            if not os.path.exists(lora_path) and lora_checkpoint_dir:
                lora_path = os.path.join(lora_checkpoint_dir, "lora_weights")
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

    for i in tqdm(
        range(n),
        desc=benchmark_name,
        miniters=EVAL_PROGRESS_LOG_INTERVAL,
        mininterval=1.0,
        disable=_eval_tqdm_disable(),
    ):
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
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            **_generate_extra_kw(processor),
        )

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

        is_ok = match_answer(generated_text, gt_answer)
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

        if (i + 1) % EVAL_PROGRESS_LOG_INTERVAL == 0:
            print(f"  [{benchmark_name}] {i + 1}/{n}  acc={correct / total:.4f} ({correct}/{total})", flush=True)

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


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    config = ReInspectionConfig(**OmegaConf.to_container(cfg, resolve=True))
    backend = config.backend

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

    available_bm = [bm for bm in config.benchmarks if bm in BENCHMARK_CONFIGS]
    present, missing = [], []
    for bm in available_bm:
        rel = BENCHMARK_CONFIGS[bm]
        path = _resolve_data_path(rel["data_file"], config.data_root)
        (present if os.path.exists(path) else missing).append(bm)
    print(f"\n{'=' * 60}")
    print(f"Evaluation: {backend} | model: {config.model_name_or_path}")
    print(f"Benchmarks ({len(present)} available): {', '.join(present) or '(none)'}")
    if missing:
        print(f"Benchmarks skipped (data not found): {', '.join(missing)}")
    print(f"{'=' * 60}\n")

    _init_wandb(config)
    conditions = [config.eval_condition]
    if config.eval_compare:
        conditions = ["frozen", "lora_only", "reinspection"]

    prefix = f"{backend}_eval"
    all_results = []
    wandb_step = 0

    for condition in conditions:
        if condition == "lora_only":
            ckpt = config.lora_checkpoint_dir or config.checkpoint_dir
            if ckpt is None:
                print(f"Skipping {condition}: no checkpoint directory")
                continue
            if not os.path.exists(os.path.join(ckpt, "lora_weights")):
                print(f"Skipping {condition}: lora_weights not found")
                continue
        if condition == "reinspection" and config.checkpoint_dir is None:
            print(f"Skipping {condition}: no checkpoint_dir")
            continue

        print(f"\n{'=' * 60}\nCondition: {condition}\n{'=' * 60}")
        model, is_ri = load_condition_model(
            backend, condition, config, processor,
            checkpoint_dir=config.checkpoint_dir,
            lora_checkpoint_dir=config.lora_checkpoint_dir,
        )
        model.eval()

        for bm in config.benchmarks:
            if bm not in BENCHMARK_CONFIGS:
                continue
            rel = BENCHMARK_CONFIGS[bm]
            data_file = _resolve_data_path(rel["data_file"], config.data_root)
            image_root = _resolve_data_path(rel["image_root"], config.data_root)
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
                max_samples=config.max_samples,
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

    out_dir = os.path.dirname(config.output_file) or "."
    os.makedirs(out_dir, exist_ok=True)
    for result in all_results:
        samples = result.pop("samples", [])
        if samples:
            sf = os.path.join(out_dir, f"{result['condition']}_{result['benchmark']}_samples.json")
            with open(sf, "w", encoding="utf-8") as f:
                json.dump(samples, f, indent=2)

    with open(config.output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {config.output_file}")

    table_file = config.output_file.replace(".json", "_table.txt")
    print_comparison_table(all_results, output_file=table_file)

    _log_eval_results_artifact(config.output_file, all_results, backend)
    _finish_wandb()


if __name__ == "__main__":
    main()
