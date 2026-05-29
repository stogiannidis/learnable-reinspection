"""Spatial-benchmark evaluation for InternVL3, Qwen2.5-VL, and Gemma 4.

Loads processors and condition-specific checkpoints (frozen base, LoRA-only, or
full re-inspection), runs greedy decoding on JSON/JSONL benchmarks, scores
answers with MCQ-aware matching, optionally logs attention entropy, and writes
JSON plus human-readable comparison tables.

Entry point: ``python -m src.evaluate`` with Hydra overrides (for example
``stage=eval``, ``backend=internvl3``).
"""

from __future__ import annotations

import json
import logging
import os
import random
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
from tqdm import tqdm as tqdm_stdlib
from transformers import (
    AutoProcessor,
    InternVLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Gemma4ForConditionalGeneration,
)

import hydra
from omegaconf import DictConfig, OmegaConf

from src.backends.hf_hub_utils import resolve_pretrained_local_path
from src.config import ReInspectionConfig
from src.eval_checkpoints import (
    find_lora_weights_path,
    find_reinspection_module_path,
    log_eval_checkpoint_plan,
    lora_only_checkpoint_root,
    print_eval_model_paths,
    resolve_eval_model_paths,
)
from src.eval_prompting import (
    build_eval_question,
    extract_final_answer,
    load_frozen_cache as _load_frozen_cache,
    save_frozen_cache as _save_frozen_cache,
)
from src.data.chat_template import build_chat_messages as intern_build_chat
from src.data.gemma4_chat import build_chat_messages as gemma4_build_chat
from src.data.llava_next_chat import build_chat_messages as llava_next_build_chat
from src.data.spatial_dataset import SpatialVQADataset
from src.data.utils import build_chat_messages as qwen_build_chat
from src.utils.attn import resolve_attn_implementation
from src.utils.hydra_util import strip_deepspeed_local_rank_argv
from src.utils.progress import make_eval_tqdm
from src.training.trainer import _env_info, _git_info

strip_deepspeed_local_rank_argv()


def _set_eval_reproducibility(seed: int) -> None:
    """Fix random seeds for eval; cudnn deterministic mode for more stable CUDA results."""
    try:
        from transformers import set_seed as _hf_set_seed

        _hf_set_seed(seed)
    except Exception:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Silence noisy third-party loggers (pad_token_id spam is WARNING from transformers.generation.utils)
for _logger_name in ("httpx", "httpcore", "urllib3"):
    logging.getLogger(_logger_name).setLevel(logging.WARNING)
for _logger_name in ("transformers.generation", "transformers.generation.utils"):
    logging.getLogger(_logger_name).setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*pad_token_id.*eos_token_id.*")
warnings.filterwarnings("ignore", message=".*not a valid argument for this processor.*")

BENCHMARK_CONFIGS = {
    "vsr": {"data_file": "vsr/test.jsonl", "image_root": "vsr/images"},
    "gqa_spatial": {"data_file": "gqa_spatial/test.json", "image_root": "gqa_spatial/images"},
    "whatsup": {"data_file": "whatsup/test.json", "image_root": "whatsup/images"},
    "3dsrbench": {"data_file": "3dsrbench/test.json", "image_root": "3dsrbench/images"},
    "mindcube": {"data_file": "mindcube/test.json", "image_root": "mindcube/images"},
    "blink": {"data_file": "blink/test.json", "image_root": "blink/images"},
    "srbench": {"data_file": "srbench/test.json", "image_root": "srbench/images"},
    "qspatial": {"data_file": "qspatial/test.json", "image_root": "qspatial/images"},
    "embspatial": {"data_file": "embspatial/test.json", "image_root": "embspatial/images"},
    "realworldqa": {"data_file": "realworldqa/test.json", "image_root": "realworldqa/images"},
    "vsr_zeroshot": {"data_file": "vsr_zeroshot/test.jsonl", "image_root": "vsr_zeroshot/images"},
    "cv_bench": {"data_file": "cv_bench/test.json", "image_root": "cv_bench/images"},
    "vstar_bench": {"data_file": "vstar_bench/test.json", "image_root": "vstar_bench/images"},
    "mmvp": {"data_file": "mmvp/test.json", "image_root": "mmvp/images"},
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
    """Return the repository root (parent of the ``src`` package directory)."""
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
    """Resolve a path string against ``base_dir`` when relative.

    Args:
        path_like: User or config path (may be absolute).
        base_dir: Directory used for relative resolution (defaults to ``cwd()``).

    Returns:
        Absolute path as a string.
    """
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return str(path)
    base = base_dir or Path.cwd()
    return str((base / path).resolve())


def _resolve_data_path(path_like: str, data_root: str) -> str:
    """Resolve a benchmark-relative path, preferring ``data_root`` then repo cwd.

    Args:
        path_like: Relative path from a benchmark config (for example ``vsr/test.jsonl``).
        data_root: Root directory containing unpacked benchmark folders.

    Returns:
        Existing file path when found; otherwise the ``data_root``-relative path
        for clearer downstream ``FileNotFoundError`` messages.
    """
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
    """Optionally initialize W&B for eval with config, git/env, and checkpoint artifact."""
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
    """Log metrics to W&B when a run is active (no-op on failure)."""
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
    """End the W&B run if one was started."""
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
    """Lowercase, strip, and collapse internal whitespace for robust string compare."""
    return " ".join(text.strip().lower().split())


_MCQ_LETTERS = frozenset("ABCDEF")


def _extract_mcq_letter(text: str) -> Optional[str]:
    """Extract a single MCQ letter (A-F) from model output, if present.

    Most benchmarks in this suite use 2-4 choices (A-D); CV-Bench's Count
    subtask uses up to 6 choices, so the matcher accepts A-F.
    """
    import re

    text = text.strip()
    if text.upper() in _MCQ_LETTERS:
        return text.upper()
    m = re.match(r"^\(?([A-Fa-f])\)?[\.\s:]*$", text)
    if m:
        return m.group(1).upper()
    m = re.match(r"^\(?([A-Fa-f])\)?[\.\)\s:]", text)
    if m:
        return m.group(1).upper()
    return None


def _gt_mcq_letter(gt_norm: str) -> Optional[str]:
    """If the ground truth itself looks like an MCQ option (``"c. top"`` /
    ``"(b) above"``), return its leading letter. Used to credit terse
    letter-only model outputs against verbose option-text ground truths.
    """
    import re

    m = re.match(r"^\(?([a-f])\)?[\.\s)]", gt_norm)
    if m:
        return m.group(1).upper()
    return None


# Canonical-key mapping for short boolean / direction answers. Lets a verbose
# baseline output ("yes, the bowl is...") credit against a terse GT ("True") and
# vice versa. The match is anchored at the start of the model output so an
# embedded keyword inside a contradiction ("no, it's not above") doesn't
# falsely fire.
_CANONICAL_GROUPS: Dict[str, str] = {}
for _phrases, _key in [
    (["true", "yes", "correct", "affirmative"],              "pos"),
    (["false", "no", "incorrect", "negative"],               "neg"),
    (["left", "to the left", "to the left of",
      "on the left", "on the left of", "left of",
      "left side"],                                          "left"),
    (["right", "to the right", "to the right of",
      "on the right", "on the right of", "right of",
      "right side"],                                         "right"),
    (["above", "on top of", "on top", "over"],               "above"),
    (["below", "underneath", "under", "beneath"],            "below"),
    (["behind", "in back of", "at the back of"],             "behind"),
    (["in front of", "in front", "front of", "ahead of"],    "front"),
    (["inside", "within", "into"],                           "inside"),
    (["outside", "out of"],                                  "outside"),
]:
    for _p in _phrases:
        _CANONICAL_GROUPS[_p] = _key

# Phrases sorted longest-first so "to the left of" matches before "left".
_CANONICAL_PHRASES = sorted(_CANONICAL_GROUPS.keys(), key=len, reverse=True)


def _leading_canonical(text: Optional[str]) -> Optional[str]:
    """Return the canonical key for the first answer-bearing token in ``text``.

    Looks within the first 60 normalized characters and requires a word
    boundary at the end of the phrase, so verbose answers like ``"yes, the
    apple is above..."`` resolve to ``"pos"`` while ``"no, not yes"`` resolves
    to ``"neg"`` (the leading token, not an embedded one).
    """
    import re

    n = " ".join((text or "").strip().lower().split())
    if not n:
        return None
    head = n[:60]
    for phrase in _CANONICAL_PHRASES:
        if re.match(r"\W*" + re.escape(phrase) + r"\b", head):
            return _CANONICAL_GROUPS[phrase]
    return None


def match_answer(generated: str, ground_truth: str) -> bool:
    """Return True if the decoded answer matches the reference label.

    Multiple-choice answers (single letter A–D) use strict letter extraction so
    substring matches cannot succeed on unrelated text; free-form answers allow
    equality or reference contained in the hypothesis after normalization.

    Args:
        generated: Raw decoded model string.
        ground_truth: Dataset reference answer.

    Returns:
        Whether the pair is counted as correct for accuracy metrics.
    """
    gen_norm = normalize_answer(generated)
    gt_norm = normalize_answer(ground_truth)

    # Single-letter MCQ (A-F): never use substring — e.g. gt "A" must not match
    # the character "a" inside "space" / "shape" in long free-form answers.
    if len(gt_norm) == 1 and gt_norm.upper() in _MCQ_LETTERS:
        if gen_norm == gt_norm:
            return True
        gen_letter = _extract_mcq_letter(generated)
        if gen_letter is not None:
            return gen_letter == gt_norm.upper()
        return False

    # MCQ with option-text suffix (e.g. gt="C. top" / "(b) above"): credit a
    # bare-letter model output ("c") when its letter matches gt's letter.
    # Falls through to the free-form path on no match so verbose answers like
    # "c. top" still credit via substring.
    gt_letter = _gt_mcq_letter(gt_norm)
    if gt_letter is not None:
        gen_letter = _extract_mcq_letter(generated)
        if gen_letter is not None:
            return gen_letter == gt_letter
        # gen_letter is None → no clean letter extractable, fall through to
        # substring rule so we don't regress against the previous matcher.

    # Canonical short-answer equivalence: ``"True"`` <-> ``"yes"``, ``"left"``
    # <-> ``"to the left of"``, etc. Symmetric so verbose baselines and terse
    # re-inspection outputs are credited on equal footing.
    gt_key = _CANONICAL_GROUPS.get(gt_norm)
    if gt_key is not None:
        gen_key = _leading_canonical(generated)
        if gen_key is not None:
            return gen_key == gt_key
        # No clean leading canonical word → fall through to substring rule.

    if gen_norm == gt_norm or gt_norm in gen_norm:
        return True

    return False


def print_comparison_table(all_results: list, output_file: Optional[str] = None) -> None:
    """Print (and optionally save) a formatted accuracy table across conditions and benchmarks."""
    if not all_results:
        return

    # {condition: {benchmark: accuracy}} and totals for mean (exclude 0-sample runs)
    table: Dict[str, Dict[str, float]] = defaultdict(dict)
    totals: Dict[str, Dict[str, int]] = defaultdict(dict)
    for r in all_results:
        table[r["condition"]][r["benchmark"]] = r["accuracy"]
        totals[r["condition"]][r["benchmark"]] = int(r.get("total", 0))

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
        vals = [
            table[c][b]
            for b in benchmarks
            if b in table[c] and totals[c].get(b, 0) > 0
        ]
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
    """Mean Shannon entropy of averaged vision attention across queries.

    Args:
        attn_vis: Attention tensor with a final dimension over vision positions.

    Returns:
        Scalar mean entropy (natural log) as a Python float.
    """
    avg_attn = attn_vis.mean(dim=1).clamp(min=1e-8)
    avg_attn = avg_attn / avg_attn.sum(dim=-1, keepdim=True)
    entropy = -(avg_attn * avg_attn.log()).sum(dim=-1)
    return entropy.mean().item()


def model_device(model) -> torch.device:
    """Infer the device hosting model parameters (``model.device`` or first param)."""
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
    attn_implementation: Optional[str] = None,
):
    """Load a base or wrapped model for one evaluation ``condition``.

    Args:
        backend: One of ``internvl3``, ``qwen25vl``, ``gemma4``.
        condition: ``reinspection`` (wrapper + checkpoint), ``frozen`` (base HF
            model), or ``lora_only`` (base + LoRA adapter weights).
        config: Run configuration (paths, dtype, module sizes).
        processor: Loaded processor used by Gemma/InternVL wrapper paths.
        checkpoint_dir: Directory containing ``reinspection_module.pt`` and/or
            ``lora_weights`` for the re-inspection or LoRA-only conditions.
        lora_checkpoint_dir: Optional separate LoRA directory when not colocated.
        attn_implementation: Optional override for the HF attention backend.
            Pass ``"eager"`` for decoder attention-map experiments; otherwise
            falls back to ``config.attn_implementation`` (FlashAttention 2 by
            default in this repo).

    Returns:
        Tuple ``(model, is_reinspection)`` where the boolean flags whether the
        forward pass exposes re-inspection hooks (for example attention maps).

    Raises:
        ValueError: For ``lora_only`` without any checkpoint directory.
        FileNotFoundError: When expected LoRA weights are missing on disk.
    """
    def _load_ri_checkpoint(model, checkpoint_dir, lora_checkpoint_dir):
        """Load reinspection weights and optional LoRA into a *WithReInspection wrapper.

        Reinspection module: prefer ``lora_checkpoint_dir`` when it carries one
        (Stage-2 retrains the module jointly with LoRA; pairing Stage-1's module
        with Stage-2's LoRA produces catastrophic collapse to grounding outputs).
        """
        ri_path = find_reinspection_module_path(checkpoint_dir, lora_checkpoint_dir)
        if ri_path is not None:
            state_dict = torch.load(ri_path, map_location="cpu", weights_only=True)
            model.reinspection.load_state_dict(state_dict)
        model.reinspection.to(model.device)
        print(f"[reinspection] loaded module from: {ri_path}", flush=True)

        lora_path = find_lora_weights_path(
            checkpoint_dir,
            lora_checkpoint_dir,
            search_order=(checkpoint_dir, lora_checkpoint_dir),
        )
        if lora_path is not None:
            model.base_model.model.language_model = PeftModel.from_pretrained(
                model.base_model.model.language_model, lora_path
            )
        print(f"[reinspection] loaded LoRA from: {lora_path}", flush=True)

    def _load_lora_only(base_model, checkpoint_dir, lora_checkpoint_dir):
        ckpt = lora_only_checkpoint_root(checkpoint_dir, lora_checkpoint_dir)
        if ckpt is None:
            raise ValueError("lora_only requires --checkpoint_dir or --lora_checkpoint_dir")
        lora_path = os.path.join(ckpt, "lora_weights")
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA weights not found: {lora_path}")
        base_model.model.language_model = PeftModel.from_pretrained(
            base_model.model.language_model, lora_path
        )
        print(f"[lora_only] loaded LoRA from: {lora_path}", flush=True)

    effective_attn_impl = resolve_attn_implementation(
        attn_implementation
        if attn_implementation is not None
        else config.attn_implementation
    )

    def _hf_load_kwargs() -> dict:
        load_kw = {
            "torch_dtype": torch.bfloat16 if config.bf16 else torch.float32,
            "device_map": "auto",
        }
        if effective_attn_impl is not None:
            load_kw["attn_implementation"] = effective_attn_impl
        return load_kw

    if backend == "qwen25vl":
        from src.backends.qwen25vl import load_model as load_qwen25_ri

        if condition == "reinspection":
            model = load_qwen25_ri(
                config,
                device_map="auto",
                attn_implementation=effective_attn_impl,
            )
            _load_ri_checkpoint(model, checkpoint_dir, lora_checkpoint_dir)
            return model, True
        resolved = resolve_pretrained_local_path(config.model_name_or_path)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            resolved,
            **_hf_load_kwargs(),
        )
        if condition == "lora_only":
            _load_lora_only(model, checkpoint_dir, lora_checkpoint_dir)
        return model, False

    if backend == "gemma4":
        from src.backends.gemma4 import load_model as load_gemma4_ri

        if condition == "reinspection":
            model = load_gemma4_ri(
                config,
                device_map="auto",
                processor=processor,
                attn_implementation=effective_attn_impl,
            )
            _load_ri_checkpoint(model, checkpoint_dir, lora_checkpoint_dir)
            return model, True
        resolved = resolve_pretrained_local_path(config.model_name_or_path)
        model = Gemma4ForConditionalGeneration.from_pretrained(
            resolved,
            **_hf_load_kwargs(),
        )
        if condition == "lora_only":
            _load_lora_only(model, checkpoint_dir, lora_checkpoint_dir)
        return model, False

    if backend == "llava_next":
        from src.backends.llava_next import load_model as load_llava_ri
        from transformers import LlavaNextForConditionalGeneration

        if condition == "reinspection":
            model = load_llava_ri(
                config,
                device_map="auto",
                processor=processor,
                attn_implementation=effective_attn_impl,
            )
            _load_ri_checkpoint(model, checkpoint_dir, lora_checkpoint_dir)
            return model, True
        resolved = resolve_pretrained_local_path(config.model_name_or_path)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            resolved,
            **_hf_load_kwargs(),
        )
        if condition == "lora_only":
            _load_lora_only(model, checkpoint_dir, lora_checkpoint_dir)
        return model, False

    from src.backends.internvl3 import load_model as load_intern_ri

    intern_attn_kw = {}
    if effective_attn_impl is not None:
        intern_attn_kw["attn_implementation"] = effective_attn_impl

    if condition == "reinspection":
        model = load_intern_ri(
            config, device_map="auto", processor=processor, **intern_attn_kw
        )
        _load_ri_checkpoint(model, checkpoint_dir, lora_checkpoint_dir)
        return model, True

    resolved = resolve_pretrained_local_path(config.model_name_or_path)
    model = InternVLForConditionalGeneration.from_pretrained(
        resolved,
        **_hf_load_kwargs(),
    )
    if condition == "lora_only":
        _load_lora_only(model, checkpoint_dir, lora_checkpoint_dir)
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
    condition: str = "",
) -> dict:
    """Evaluate one benchmark split with greedy decoding and per-sample outputs.

    Args:
        backend: VLM backend id controlling chat template and tensor layout.
        model: Loaded HF or custom wrapper model in eval mode (modified in-place).
        processor: Matching processor for tokenization and decoding.
        data_file: Path to JSON or JSONL annotations.
        image_root: Directory containing image files referenced by samples.
        benchmark_name: Short name used in logs and result dicts.
        config: Pixel bounds, crop flags, system prompt, and ignore index for labels.
        is_reinspection: Whether to slice generated tokens after inserted queries
            and optionally collect attention entropy.
        max_samples: Cap on evaluated items; ``<= 0`` means full split.
        condition: Eval condition label for progress-bar titles (e.g. ``frozen``).

    Returns:
        Dict with keys ``benchmark``, ``accuracy``, ``correct``, ``total``,
        ``skipped``, ``mean_attention_entropy`` (or None), and ``samples`` (list
        of per-item records).
    """
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

    from src.backends.internvl3 import InternVL3WithReInspection
    from src.backends.gemma4 import Gemma4WithReInspection

    _QWEN_BACKENDS = ("qwen25vl",)
    _PROMPT_LEN_BACKENDS = ("internvl3", "gemma4", "llava_next")

    progress_desc = f"{condition}/{benchmark_name}" if condition else benchmark_name
    pbar = make_eval_tqdm(
        config,
        range(n),
        desc=progress_desc,
        miniters=EVAL_PROGRESS_LOG_INTERVAL,
        mininterval=1.0,
        disable=_eval_tqdm_disable(),
    )

    for i in pbar:
        sample = ds.samples[i]
        gt_answer = sample["answer"]
        raw_question = sample["question"]
        question = build_eval_question(raw_question, config)
        image_path = os.path.join(image_root, sample["image"])

        if not os.path.isfile(image_path):
            skipped += 1
            continue

        try:
            if backend in _QWEN_BACKENDS:
                messages = qwen_build_chat(question, image_path=image_path)
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(
                    text=[text],
                    images=[image_path],
                    return_tensors="pt",
                    max_pixels=config.max_pixels,
                    min_pixels=config.min_pixels,
                )
            elif backend == "gemma4":
                messages = gemma4_build_chat(
                    question=question,
                    image_path=image_path,
                    system_prompt=config.system_prompt,
                )
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(
                    text=[text],
                    images=[image_path],
                    return_tensors="pt",
                )
            elif backend == "llava_next":
                messages = llava_next_build_chat(
                    question=question,
                    image_path=image_path,
                    system_prompt=config.system_prompt,
                )
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(
                    text=[text],
                    images=[image_path],
                    return_tensors="pt",
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
            tqdm_stdlib.write(f"  skip {i}: {e}")
            continue

        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=config.eval_max_new_tokens,
            do_sample=False,
            **_generate_extra_kw(processor),
        )

        if backend in _QWEN_BACKENDS:
            if is_reinspection:
                input_len = inputs["input_ids"].shape[1] + config.n_queries
            else:
                input_len = inputs["input_ids"].shape[1]
        elif hasattr(model, "last_generation_prompt_lengths") and model.last_generation_prompt_lengths is not None:
            input_len = int(model.last_generation_prompt_lengths[0].item())
        else:
            input_len = int(inputs["input_ids"].shape[1])

        generated_text = processor.batch_decode(
            generated_ids[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip().lower()

        scored_answer = extract_final_answer(generated_text, cot_enabled=config.eval_cot_enabled)
        is_ok = match_answer(scored_answer, gt_answer)
        if is_ok:
            correct += 1
        total += 1
        sample_outputs.append({
            "idx": i,
            "image": sample["image"],
            "question": raw_question,
            "prompted_question": question,
            "eval_cot_prompt": config.eval_cot_prompt if config.eval_cot_enabled else "",
            "ground_truth": gt_answer,
            "model_output": generated_text,
            "scored_answer": scored_answer,
            "correct": is_ok,
        })

        if is_reinspection and hasattr(model, "get_attention_maps"):
            _, attn_vis = model.get_attention_maps()
            if attn_vis is not None:
                entropies.append(compute_attention_entropy(attn_vis))

        if total > 0:
            pbar.set_postfix(acc=f"{correct / total:.4f}", score=f"{correct}/{total}")

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
    """Hydra entry: run configured benchmarks and write JSON + comparison table.

    Args:
        cfg: Merged Hydra configuration for evaluation (``stage=eval`` preset).
    """
    config = ReInspectionConfig(**OmegaConf.to_container(cfg, resolve=True))
    _set_eval_reproducibility(config.seed)
    backend = config.backend

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
    else:
        resolved = resolve_pretrained_local_path(config.model_name_or_path)
        processor = AutoProcessor.from_pretrained(
            resolved,
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
    print(f"seed={config.seed} (greedy decode; cudnn deterministic)")
    print(f"Benchmarks ({len(present)} available): {', '.join(present) or '(none)'}")
    if missing:
        print(f"Benchmarks skipped (data not found): {', '.join(missing)}")
    print(f"{'=' * 60}\n")

    _init_wandb(config)
    conditions = [config.eval_condition]
    if config.eval_compare:
        conditions = ["frozen", "lora_only", "reinspection"]

    log_eval_checkpoint_plan(config, conditions)

    prefix = f"{backend}_eval"
    all_results = []
    wandb_step = 0

    for condition in conditions:
        if condition == "lora_only":
            ckpt = config.lora_checkpoint_dir or config.checkpoint_dir
            if ckpt is None:
                print(f"Skipping {condition}: no checkpoint directory")
                continue
            lora_dir = os.path.join(ckpt, "lora_weights")
            if not os.path.exists(lora_dir):
                abs_ck = os.path.abspath(ckpt)
                print(
                    f"Skipping {condition}: lora_weights not found at {lora_dir}\n"
                    f"  Checkpoint root (absolute): {abs_ck}\n"
                    "  Fix: point checkpoint_dir or lora_checkpoint_dir at a Stage-2 epoch folder "
                    "under models/<backend>/[experiment_name]/stage2/epoch_*/ — training only writes "
                    "lora_weights/ when saving Stage 2. Stage-1 dirs have reinspection_module.pt but "
                    "no LoRA. You can set lora_checkpoint_dir separately if LoRA lives elsewhere."
                )
                if os.path.isdir(abs_ck):
                    try:
                        print(
                            "  Directory contents: "
                            f"{sorted(os.listdir(abs_ck))}"
                        )
                    except OSError:
                        pass
                continue
        if condition == "reinspection" and config.checkpoint_dir is None:
            print(f"Skipping {condition}: no checkpoint_dir")
            continue

        # Load frozen baseline from cache if available (skips GPU inference).
        if condition == "frozen" and config.frozen_cache_file and os.path.exists(config.frozen_cache_file):
            cached = _load_frozen_cache(config.frozen_cache_file, config)
            if cached is not None:
                print(
                    f"\n{'=' * 60}\nCondition: frozen  "
                    f"[loaded from cache: {config.frozen_cache_file}]\n{'=' * 60}"
                )
                print_eval_model_paths(
                    "frozen",
                    resolve_eval_model_paths(
                        "frozen",
                        checkpoint_dir=config.checkpoint_dir,
                        lora_checkpoint_dir=config.lora_checkpoint_dir,
                        model_name_or_path=config.model_name_or_path,
                        frozen_cache_file=config.frozen_cache_file,
                    ),
                )
                print("", flush=True)
                for r in cached:
                    r["condition"] = "frozen"
                    all_results.append(r)
                    m = {
                        f"{prefix}/frozen/{r['benchmark']}/accuracy": r["accuracy"],
                        f"{prefix}/frozen/{r['benchmark']}/correct": r["correct"],
                        f"{prefix}/frozen/{r['benchmark']}/total": r["total"],
                    }
                    _log_wandb(m, wandb_step)
                    wandb_step += 1
                continue

        print(f"\n{'=' * 60}\nCondition: {condition}\n{'=' * 60}")
        print_eval_model_paths(
            condition,
            resolve_eval_model_paths(
                condition,
                checkpoint_dir=config.checkpoint_dir,
                lora_checkpoint_dir=config.lora_checkpoint_dir,
                model_name_or_path=config.model_name_or_path,
                frozen_cache_file=config.frozen_cache_file,
            ),
        )
        print("", flush=True)
        model, is_ri = load_condition_model(
            backend, condition, config, processor,
            checkpoint_dir=config.checkpoint_dir,
            lora_checkpoint_dir=config.lora_checkpoint_dir,
        )
        model.eval()

        condition_results = []
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
                condition=condition,
            )
            result["condition"] = condition
            condition_results.append(result)
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

        # Persist frozen results so future runs can skip this pass.
        if condition == "frozen" and config.frozen_cache_file:
            _save_frozen_cache(config.frozen_cache_file, condition_results, config)
            print(f"  Frozen baseline saved to {config.frozen_cache_file}")

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
