"""Motivation experiment: Spatial Minimal-Pair Flip Consistency & Attention Divergence.

Demonstrates two architectural limitations of standard VLMs on spatial reasoning:

1. **Flip Consistency** — Given the same image and two logically-inverse spatial
   questions (e.g. "Is A above B?" / "Is B above A?"), how often does the model
   give opposite answers?  A spatially-competent model should approach 100%.

2. **Attention Divergence** — For the same image with different spatial questions,
   how different are the model's attention patterns over vision tokens?  A model
   whose vision encoding is question-agnostic will show low divergence.

Entry point: ``python -m src.motivation stage=eval backend=internvl3 [overrides]``

Supports all backends (internvl3, qwen25vl, gemma4) and two conditions:
  - ``frozen``: base VLM without re-inspection (shows the problem)
  - ``reinspection``: VLM + re-inspection module (shows the fix)
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import warnings
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm as tqdm_stdlib
from transformers import AutoProcessor

import hydra
from omegaconf import DictConfig, OmegaConf

from src.config import ReInspectionConfig
from src.utils.progress import get_tqdm
from src.data.chat_template import build_chat_messages as intern_build_chat
from src.data.gemma4_chat import build_chat_messages as gemma4_build_chat
from src.data.minimal_pairs import MinimalPair, generate_pairs_from_vsr
from src.data.utils import build_chat_messages as qwen_build_chat
from src.evaluate import (
    load_condition_model,
    match_answer,
    model_device,
    normalize_answer,
    _generate_extra_kw,
)
from src.utils.attention_schema import (
    ATTN_SIGNAL_CMAP,
    ATTN_SIGNAL_LABELS,
    ATTN_SIGNAL_STORAGE,
    attention_capture_plan as _attention_capture_plan,
    default_display_signal as _default_display_signal,
    single_condition_display_signals as _single_condition_display_signals,
)
from src.utils.hydra_util import strip_deepspeed_local_rank_argv

strip_deepspeed_local_rank_argv()

for _logger_name in ("httpx", "httpcore", "urllib3"):
    logging.getLogger(_logger_name).setLevel(logging.WARNING)
for _logger_name in ("transformers.generation", "transformers.generation.utils"):
    logging.getLogger(_logger_name).setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*pad_token_id.*eos_token_id.*")
warnings.filterwarnings("ignore", message=".*not a valid argument for this processor.*")

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Attention extraction helpers                                       #
# ------------------------------------------------------------------ #

def _jsd(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """Jensen-Shannon divergence between two discrete distributions."""
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m))))


def _internvl3_merged_hw(model) -> Optional[Tuple[int, int]]:
    """Return (h, w) merged vision-token grid for single-tile InternVL3."""
    _base = getattr(model, "base_model", model)
    cfg = getattr(_base, "config", None)
    if cfg is None:
        return None
    vcfg = getattr(cfg, "vision_config", None)
    if vcfg is None:
        return None
    img_size = vcfg.image_size[0] if isinstance(vcfg.image_size, (list, tuple)) else vcfg.image_size
    patch = vcfg.patch_size[0] if isinstance(vcfg.patch_size, (list, tuple)) else vcfg.patch_size
    downsample = getattr(cfg, "downsample_ratio", 0.5)
    side = int(round((img_size // patch) * downsample))
    return side, side


def _normalize_spatial_signal(values, n_spatial: int) -> Optional[np.ndarray]:
    """Crop a 1D spatial signal to the patch grid and normalize it."""
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.shape[0] < n_spatial:
        return None
    arr = arr[:n_spatial]
    total = float(arr.sum())
    if not np.isfinite(total) or total <= 0.0:
        return None
    return arr / (total + 1e-8)


def _mean_a_vis(attn_vis, n_spatial: int) -> Optional[np.ndarray]:
    """Average A_vis over queries and normalize on the spatial grid."""
    if attn_vis is None:
        return None
    arr = np.asarray(attn_vis, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] == 0:
        return None
    return _normalize_spatial_signal(arr[0].mean(axis=0), n_spatial)


def _npz_available_signals(data) -> List[str]:
    """Read signal availability from either the expanded or legacy NPZ schema."""
    if "available_signals" in data.files:
        return [str(x) for x in np.asarray(data["available_signals"]).tolist()]
    if "attn_source" in data.files:
        return [str(data["attn_source"])]
    condition = str(data["condition"]) if "condition" in data.files else "frozen"
    return ["a_vis" if condition == "reinspection" else "hidden_cosine"]


def _npz_signal_pair(data, signal: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load one signal pair from expanded NPZs while remaining legacy-compatible."""
    prefix = ATTN_SIGNAL_STORAGE[signal]
    key_a = f"{prefix}_a"
    key_b = f"{prefix}_b"
    if key_a in data.files and key_b in data.files:
        return np.asarray(data[key_a]), np.asarray(data[key_b])

    legacy_source = str(data["attn_source"]) if "attn_source" in data.files else None
    if legacy_source == signal or (legacy_source is None and signal in _npz_available_signals(data)):
        return np.asarray(data["attn_a"]), np.asarray(data["attn_b"])
    return None, None


# ------------------------------------------------------------------ #
#  Core generation + attention capture                                #
# ------------------------------------------------------------------ #

def _decoder_image_attention(
    sequences: torch.Tensor,
    attentions,
    image_token_id: int,
    h_merged: int,
    w_merged: int,
    tokenizer=None,
) -> Optional[np.ndarray]:
    """Mean decoder self-attention from generated tokens -> image-token columns."""
    if attentions is None or len(attentions) == 0 or attentions[0][0] is None:
        print("  [decoder-attn] attentions missing — is attn_implementation='eager'?")
        return None

    from src.utils.visualize_attention import extract_decoder_image_attention

    try:
        mean_attn, _, _, _ = extract_decoder_image_attention(
            sequences, attentions, image_token_id, tokenizer,
            h_merged, w_merged,
        )
        return mean_attn
    except ValueError as exc:
        print(f"  [decoder-attn] {exc}")
        return None


@torch.no_grad()
def _generate_with_attention(
    model,
    processor,
    image_or_path: Union[str, Image.Image],
    question: str,
    backend: str,
    config: ReInspectionConfig,
    is_reinspection: bool,
    return_grid_info: bool = False,
    capture_base_vision_attn: bool = False,
    capture_decoder_attn: bool = False,
    force_single_tile: bool = False,
) -> Dict:
    """Run a single question through the model and return answer + attention info.

    Args:
        image_or_path: Either a file path string or a PIL Image.
        return_grid_info: If True, include ``image_grid_thw`` in the result
            (Qwen backends only) for attention-map visualisation.

    Returns dict with keys:
      - generated_text: str
      - reinspection_attn_vis: np.ndarray or None  (N_q, N_vis) from the module
      - base_last_hidden: np.ndarray or None  (D,) hidden state at generation start
      - image_grid_thw: list or None  (only when return_grid_info=True and available)
    """
    # If a PIL Image is given, write to a temp file so the processor gets a path.
    _tmp_file = None
    if isinstance(image_or_path, Image.Image):
        _tmp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        image_or_path.save(_tmp_file, format="JPEG")
        _tmp_file.close()
        image_path = _tmp_file.name
    else:
        image_path = image_or_path

    try:
        return _generate_with_attention_inner(
            model, processor, image_path, question, backend, config,
            is_reinspection, return_grid_info, capture_base_vision_attn,
            capture_decoder_attn=capture_decoder_attn,
            force_single_tile=force_single_tile,
        )
    finally:
        if _tmp_file is not None:
            os.unlink(_tmp_file.name)


@torch.no_grad()
def _generate_with_attention_inner(
    model, processor, image_path, question, backend, config,
    is_reinspection, return_grid_info, capture_base_vision_attn=False,
    capture_decoder_attn=False, force_single_tile: bool = False,
) -> Dict:
    device = model_device(model)

    if backend == "qwen25vl":
        messages = qwen_build_chat(question, image_path=image_path)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text], images=[image_path], return_tensors="pt",
            max_pixels=config.max_pixels, min_pixels=config.min_pixels,
        )
    elif backend == "gemma4":
        messages = gemma4_build_chat(question=question, image_path=image_path, system_prompt=config.system_prompt)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image_path], return_tensors="pt")
    else:
        messages = intern_build_chat(question=question, image_path=image_path, system_prompt=config.system_prompt)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if backend == "internvl3" and force_single_tile:
            inputs = processor(
                text=[text], images=[image_path], return_tensors="pt",
                crop_to_patches=False,
            )
        else:
            inputs = processor(
                text=[text], images=[image_path], return_tensors="pt",
                max_pixels=config.max_pixels, min_pixels=config.min_pixels,
                crop_to_patches=config.crop_to_patches_stage2,
            )

    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # --- Generate ---
    gen_kwargs = dict(max_new_tokens=16, do_sample=False, **_generate_extra_kw(processor))
    if capture_decoder_attn:
        gen_kwargs.update(output_attentions=True, return_dict_in_generate=True)

    gen_out = model.generate(**inputs, **gen_kwargs)

    if capture_decoder_attn:
        generated_ids = gen_out.sequences
        decoder_attentions = gen_out.attentions
    else:
        generated_ids = gen_out
        decoder_attentions = None

    # Determine input length for decoding
    if backend == "qwen25vl":
        input_len = inputs["input_ids"].shape[1] + (config.n_queries if is_reinspection else 0)
    elif hasattr(model, "last_generation_prompt_lengths") and model.last_generation_prompt_lengths is not None:
        input_len = int(model.last_generation_prompt_lengths[0].item())
    else:
        input_len = int(inputs["input_ids"].shape[1])

    generated_text = processor.batch_decode(
        generated_ids[:, input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip().lower()

    # --- Attention from Re-Inspection module (if applicable) ---
    ri_attn_vis = None
    if is_reinspection and hasattr(model, "get_attention_maps"):
        _, attn_vis = model.get_attention_maps()
        if attn_vis is not None:
            ri_attn_vis = attn_vis.cpu().numpy()  # (B, N_q, N_vis)

    # --- Hidden state capture for representation-level analysis ---
    # We extract the last hidden state at the position just before generation
    # by running a forward pass with output_hidden_states=True.
    base_last_hidden = None
    base_vision_attn = None
    try:
        fwd_model = model.base_model if hasattr(model, "base_model") else model
        with torch.no_grad():
            fwd_inputs = {k: v for k, v in inputs.items() if k != "mm_token_type_ids"}
            fwd_out = fwd_model(
                **fwd_inputs,
                output_hidden_states=True,
                use_cache=False,
            )
            # Last hidden state at the final input position
            hs = fwd_out.hidden_states[-1]  # (B, L, D)
            base_last_hidden = hs[0, -1, :].cpu().float().numpy()  # (D,)

            # Vision attention proxy for the frozen condition: cosine similarity
            # between each vision token's hidden state and the last text position.
            # Used for side-by-side comparison with re-inspection A_vis.
            if capture_base_vision_attn and not is_reinspection:
                # Qwen: mm_token_type_ids marks vision positions; InternVL3: image_token_id
                if "mm_token_type_ids" in inputs:
                    vis_mask = inputs["mm_token_type_ids"][0].bool()
                else:
                    _base = getattr(model, "base_model", model)
                    _img_tok = getattr(getattr(_base, "config", None), "image_token_id", None)
                    vis_mask = (inputs["input_ids"][0] == _img_tok) if _img_tok is not None else None
                if vis_mask is not None and vis_mask.any():
                    query_hs = hs[0, -1, :].float()
                    vis_hs = hs[0][vis_mask].float()  # (N_vis, D)
                    q_norm = query_hs / (query_hs.norm() + 1e-8)
                    v_norm = vis_hs / (vis_hs.norm(dim=-1, keepdim=True) + 1e-8)
                    sims = (v_norm @ q_norm).cpu().numpy()  # (N_vis,)
                    sims_exp = np.exp(sims - sims.max())
                    base_vision_attn = sims_exp / (sims_exp.sum() + 1e-8)
    except Exception:
        pass  # Not critical — skip silently

    # --- Decoder self-attention over image tokens (optional) ---
    decoder_image_attn = None
    if capture_decoder_attn and decoder_attentions is not None:
        _base = getattr(model, "base_model", model)
        img_tok = getattr(_base, "_image_token_id", None) or getattr(
            getattr(_base, "config", None), "image_token_id", None
        )
        h_m, w_m = None, None
        if img_tok is not None and "image_grid_thw" in inputs:
            _, _h, _w = inputs["image_grid_thw"][0].tolist()
            h_m, w_m = _h // 2, _w // 2
        elif img_tok is not None and backend == "internvl3":
            grid = _internvl3_merged_hw(model)
            if grid is not None:
                h_m, w_m = grid
        if img_tok is not None and h_m is not None and w_m is not None:
            tokenizer = getattr(processor, "tokenizer", None)
            decoder_image_attn = _decoder_image_attention(
                generated_ids, decoder_attentions, img_tok, h_m, w_m, tokenizer=tokenizer,
            )

    result = {
        "generated_text": generated_text,
        "reinspection_attn_vis": ri_attn_vis,
        "base_last_hidden": base_last_hidden,
        "base_vision_attn": base_vision_attn,
        "decoder_image_attn": decoder_image_attn,
    }

    if return_grid_info and "image_grid_thw" in inputs:
        result["image_grid_thw"] = inputs["image_grid_thw"][0].tolist()

    return result


# ------------------------------------------------------------------ #
#  Prong 1: Flip Consistency                                          #
# ------------------------------------------------------------------ #

@torch.no_grad()
def run_flip_consistency(
    model,
    processor,
    pairs: List[MinimalPair],
    backend: str,
    config: ReInspectionConfig,
    is_reinspection: bool,
    image_root: str,
    max_pairs: int = -1,
) -> Dict:
    """Evaluate flip consistency on minimal pairs.

    Returns a dict with overall metrics and per-relation breakdowns.
    """
    if max_pairs > 0:
        pairs = pairs[:max_pairs]

    results = {
        "total_pairs": 0,
        "consistent": 0,         # both correct (opposite answers)
        "both_correct": 0,       # both match ground truth
        "a_correct": 0,
        "b_correct": 0,
        "same_answer": 0,        # model gave identical answer to both (failure)
        "per_relation": defaultdict(lambda: {"total": 0, "consistent": 0, "both_correct": 0, "same_answer": 0}),
        "per_type": defaultdict(lambda: {"total": 0, "consistent": 0, "both_correct": 0, "same_answer": 0}),
        "samples": [],
    }
    model.eval()

    tqdm_cls = get_tqdm(config, cloud=True)
    for i, pair in enumerate(tqdm_cls(pairs, desc="Flip Consistency", miniters=50)):
        image_path = os.path.join(image_root, pair.image)
        if not os.path.isfile(image_path):
            continue

        try:
            out_a = _generate_with_attention(
                model, processor, image_path, pair.question_a,
                backend, config, is_reinspection,
            )
            out_b = _generate_with_attention(
                model, processor, image_path, pair.question_b,
                backend, config, is_reinspection,
            )
        except Exception as e:
            tqdm_stdlib.write(f"  skip pair {i}: {e}")
            continue

        gen_a = out_a["generated_text"]
        gen_b = out_b["generated_text"]
        a_ok = match_answer(gen_a, pair.answer_a)
        b_ok = match_answer(gen_b, pair.answer_b)

        # "Consistent" = model gave logically opposite answers
        # (regardless of whether each individual answer is correct)
        norm_a = normalize_answer(gen_a)
        norm_b = normalize_answer(gen_b)
        is_same = (norm_a == norm_b)
        is_consistent = not is_same

        results["total_pairs"] += 1
        results["a_correct"] += int(a_ok)
        results["b_correct"] += int(b_ok)
        results["both_correct"] += int(a_ok and b_ok)
        results["consistent"] += int(is_consistent)
        results["same_answer"] += int(is_same)

        for bucket_key, bucket_val in [("per_relation", pair.relation), ("per_type", pair.pair_type)]:
            results[bucket_key][bucket_val]["total"] += 1
            results[bucket_key][bucket_val]["consistent"] += int(is_consistent)
            results[bucket_key][bucket_val]["both_correct"] += int(a_ok and b_ok)
            results[bucket_key][bucket_val]["same_answer"] += int(is_same)

        results["samples"].append({
            "idx": i,
            "image": pair.image,
            "relation": pair.relation,
            "pair_type": pair.pair_type,
            "question_a": pair.question_a,
            "answer_a_gt": pair.answer_a,
            "answer_a_gen": gen_a,
            "a_correct": a_ok,
            "question_b": pair.question_b,
            "answer_b_gt": pair.answer_b,
            "answer_b_gen": gen_b,
            "b_correct": b_ok,
            "consistent": is_consistent,
        })

        if (i + 1) % 100 == 0:
            n = results["total_pairs"]
            print(
                f"  [{i+1}/{len(pairs)}]  "
                f"consistency={results['consistent']/n:.3f}  "
                f"both_correct={results['both_correct']/n:.3f}  "
                f"same_answer={results['same_answer']/n:.3f}",
                flush=True,
            )

    return results


# ------------------------------------------------------------------ #
#  Prong 2: Mirror Test (left/right only)                             #
# ------------------------------------------------------------------ #

MIRROR_RELATIONS = {"left of", "right of", "to the left of", "to the right of"}


@torch.no_grad()
def run_mirror_test(
    model,
    processor,
    pairs: List[MinimalPair],
    backend: str,
    config: ReInspectionConfig,
    is_reinspection: bool,
    image_root: str,
    max_pairs: int = -1,
) -> Dict:
    """Mirror test: flip the image horizontally and re-ask left/right questions.

    For each VSR sample with a left/right relation, a spatially-aware model
    should give the *opposite* answer when the image is mirrored (what was
    on the left is now on the right and vice-versa).

    Only pairs whose relation is in MIRROR_RELATIONS are used.
    """
    lr_pairs = [p for p in pairs if p.relation in MIRROR_RELATIONS]
    if max_pairs > 0:
        lr_pairs = lr_pairs[:max_pairs]

    results = {
        "total": 0,
        "mirror_consistent": 0,   # model gave opposite answer after flip
        "mirror_correct": 0,      # flipped answer matches expected flipped GT
        "orig_correct": 0,
        "per_relation": defaultdict(lambda: {
            "total": 0, "mirror_consistent": 0, "mirror_correct": 0,
        }),
        "samples": [],
    }
    model.eval()

    tqdm_cls = get_tqdm(config, cloud=True)
    for i, pair in enumerate(tqdm_cls(lr_pairs, desc="Mirror Test", miniters=50)):
        image_path = os.path.join(image_root, pair.image)
        if not os.path.isfile(image_path):
            continue

        try:
            # Original image, question A
            out_orig = _generate_with_attention(
                model, processor, image_path, pair.question_a,
                backend, config, is_reinspection,
            )

            # Flip image horizontally
            flipped_img = Image.open(image_path).convert("RGB").transpose(
                Image.FLIP_LEFT_RIGHT
            )

            # Same question on flipped image
            out_flip = _generate_with_attention(
                model, processor, flipped_img, pair.question_a,
                backend, config, is_reinspection,
            )
        except Exception as e:
            tqdm_stdlib.write(f"  skip mirror pair {i}: {e}")
            continue

        gen_orig = out_orig["generated_text"]
        gen_flip = out_flip["generated_text"]
        norm_orig = normalize_answer(gen_orig)
        norm_flip = normalize_answer(gen_flip)

        orig_ok = match_answer(gen_orig, pair.answer_a)
        # Expected: flipped answer is opposite of original GT
        expected_flip = "false" if pair.answer_a.strip().lower() == "true" else "true"
        flip_ok = (norm_flip == expected_flip)
        is_mirror_consistent = (norm_orig != norm_flip)

        results["total"] += 1
        results["orig_correct"] += int(orig_ok)
        results["mirror_consistent"] += int(is_mirror_consistent)
        results["mirror_correct"] += int(flip_ok)

        rel_stats = results["per_relation"][pair.relation]
        rel_stats["total"] += 1
        rel_stats["mirror_consistent"] += int(is_mirror_consistent)
        rel_stats["mirror_correct"] += int(flip_ok)

        results["samples"].append({
            "idx": i,
            "image": pair.image,
            "relation": pair.relation,
            "question": pair.question_a,
            "gt_answer": pair.answer_a,
            "gen_orig": gen_orig,
            "gen_flipped": gen_flip,
            "orig_correct": orig_ok,
            "mirror_consistent": is_mirror_consistent,
            "mirror_correct": flip_ok,
        })

        if (i + 1) % 50 == 0:
            n = results["total"]
            print(
                f"  [{i+1}/{len(lr_pairs)}]  "
                f"mirror_consistency={results['mirror_consistent']/n:.3f}  "
                f"mirror_correct={results['mirror_correct']/n:.3f}  "
                f"orig_acc={results['orig_correct']/n:.3f}",
                flush=True,
            )

    return results


# ------------------------------------------------------------------ #
#  Prong 3: Attention Divergence                                      #
# ------------------------------------------------------------------ #

@torch.no_grad()
def run_attention_divergence(
    model,
    processor,
    pairs: List[MinimalPair],
    backend: str,
    config: ReInspectionConfig,
    is_reinspection: bool,
    image_root: str,
    max_pairs: int = -1,
) -> Dict:
    """Measure attention divergence: for the same image with two different
    questions, compute Jensen-Shannon divergence between attention distributions
    over vision tokens.

    For the re-inspection model, we use A_vis (the module's visual attention).
    For the base model, we use the last hidden state cosine similarity as a proxy
    for representation divergence (since base model attention over vision tokens
    is not directly available without output_attentions=True which is expensive).
    """
    if max_pairs > 0:
        pairs = pairs[:max_pairs]

    divergences: List[float] = []
    cosine_sims: List[float] = []
    per_relation: Dict[str, List[float]] = defaultdict(list)
    samples = []
    model.eval()

    tqdm_cls = get_tqdm(config, cloud=True)
    for i, pair in enumerate(tqdm_cls(pairs, desc="Attention Divergence", miniters=50)):
        image_path = os.path.join(image_root, pair.image)
        if not os.path.isfile(image_path):
            continue

        try:
            out_a = _generate_with_attention(
                model, processor, image_path, pair.question_a,
                backend, config, is_reinspection,
            )
            out_b = _generate_with_attention(
                model, processor, image_path, pair.question_b,
                backend, config, is_reinspection,
            )
        except Exception as e:
            tqdm_stdlib.write(f"  skip pair {i}: {e}")
            continue

        sample = {
            "idx": i,
            "image": pair.image,
            "relation": pair.relation,
            "pair_type": pair.pair_type,
        }

        # Re-Inspection visual attention divergence (JSD over A_vis)
        attn_a = out_a.get("reinspection_attn_vis")
        attn_b = out_b.get("reinspection_attn_vis")
        if attn_a is not None and attn_b is not None:
            # Average across queries: (B, N_q, N_vis) → (N_vis,)
            dist_a = attn_a[0].mean(axis=0)
            dist_b = attn_b[0].mean(axis=0)
            # Ensure same length (may differ if vision token count varies)
            min_len = min(len(dist_a), len(dist_b))
            jsd = _jsd(dist_a[:min_len], dist_b[:min_len])
            divergences.append(jsd)
            per_relation[pair.relation].append(jsd)
            sample["jsd"] = jsd

        # Hidden-state cosine similarity (representation divergence proxy)
        h_a = out_a.get("base_last_hidden")
        h_b = out_b.get("base_last_hidden")
        if h_a is not None and h_b is not None:
            cos_sim = float(
                np.dot(h_a, h_b) / (np.linalg.norm(h_a) * np.linalg.norm(h_b) + 1e-10)
            )
            cosine_sims.append(cos_sim)
            sample["cosine_sim"] = cos_sim

        samples.append(sample)

        if (i + 1) % 100 == 0:
            msg_parts = [f"[{i+1}/{len(pairs)}]"]
            if divergences:
                msg_parts.append(f"mean_jsd={np.mean(divergences):.4f}")
            if cosine_sims:
                msg_parts.append(f"mean_cos_sim={np.mean(cosine_sims):.4f}")
            print("  " + "  ".join(msg_parts), flush=True)

    return {
        "mean_jsd": float(np.mean(divergences)) if divergences else None,
        "std_jsd": float(np.std(divergences)) if divergences else None,
        "mean_cosine_sim": float(np.mean(cosine_sims)) if cosine_sims else None,
        "std_cosine_sim": float(np.std(cosine_sims)) if cosine_sims else None,
        "num_pairs": len(samples),
        "per_relation_jsd": {
            rel: {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
            for rel, vals in sorted(per_relation.items())
        },
        "samples": samples,
    }


# ------------------------------------------------------------------ #
#  Prong 4: Attention Map Visualisation                               #
# ------------------------------------------------------------------ #

# Hand-picked examples spanning different spatial categories.
# Each tuple: (image_filename, relation, subject, object)
# These are chosen to span horizontal, vertical, and containment categories.
ATTN_VIS_EXAMPLES = [
    ("000000407386.jpg", "left of",  "laptop", "tv"),
    ("000000452072.jpg", "on top of", "cat", "laptop"),
    ("000000169660.jpg", "above",   "oven", "cake"),
    ("000000279541.jpg", "behind",  "person", "horse"),
    ("000000341681.jpg", "in front of", "person", "bus"),
    ("000000222455.jpg", "below",   "bench", "person"),
]


def _select_vis_examples(
    pairs: List[MinimalPair], image_root: str, n_examples: int = 6,
) -> List[MinimalPair]:
    """Select canonical visualization examples, shared across conditions."""
    selected: List[MinimalPair] = []
    example_images = {img for img, *_ in ATTN_VIS_EXAMPLES}
    for pair in pairs:
        if pair.image in example_images and len(selected) < n_examples:
            # Only include if image actually exists
            if os.path.isfile(os.path.join(image_root, pair.image)):
                selected.append(pair)
    if len(selected) < n_examples:
        for pair in pairs:
            if pair not in selected and os.path.isfile(os.path.join(image_root, pair.image)):
                selected.append(pair)
            if len(selected) >= n_examples:
                break
    return selected


@torch.no_grad()
def run_attention_visualization(
    model,
    processor,
    selected_pairs: List[MinimalPair],
    backend: str,
    config: ReInspectionConfig,
    image_root: str,
    output_dir: str,
    condition: str,
) -> List[str]:
    """Collect attention heatmap data for the given condition. Returns saved .npz paths.

    InternVL3 frozen stores decoder attention. InternVL3 reinspection stores
    both decoder attention and ``A_vis``. Other frozen backends fall back to a
    hidden-state cosine proxy; wrapper paths store ``A_vis``.

    Qwen backends use ``image_grid_thw``; InternVL3 uses the merged patch grid from
    ``vision_config`` (single-tile inputs via ``crop_to_patches=False``).
    """
    if backend not in ("qwen25vl", "internvl3"):
        print(f"  Attention visualisation not supported for backend={backend} — skipping.")
        return []

    is_ri = condition == "reinspection"
    requested_signals = _attention_capture_plan(backend, condition)
    force_single_tile = backend == "internvl3"

    fig_dir = os.path.join(output_dir, "attention_figures")
    os.makedirs(fig_dir, exist_ok=True)

    model.eval()
    saved_paths: List[str] = []

    for i, pair in enumerate(selected_pairs):
        image_path = os.path.join(image_root, pair.image)

        try:
            out_a = _generate_with_attention(
                model, processor, image_path, pair.question_a,
                backend, config, is_ri, return_grid_info=True,
                capture_base_vision_attn=("hidden_cosine" in requested_signals),
                capture_decoder_attn=("decoder" in requested_signals),
                force_single_tile=force_single_tile,
            )
            out_b = _generate_with_attention(
                model, processor, image_path, pair.question_b,
                backend, config, is_ri, return_grid_info=True,
                capture_base_vision_attn=("hidden_cosine" in requested_signals),
                capture_decoder_attn=("decoder" in requested_signals),
                force_single_tile=force_single_tile,
            )
        except Exception as e:
            print(f"  skip attn vis [{condition}] example {i}: {e}")
            continue

        if "image_grid_thw" in out_a:
            # Qwen: (T, H, W) before spatial merge
            _, h, w = out_a["image_grid_thw"]
            h_merged = h // 2
            w_merged = w // 2
        elif backend == "internvl3":
            grid = _internvl3_merged_hw(model)
            if grid is None:
                print(f"  skip attn vis [{condition}] example {i}: missing InternVL3 vision grid")
                continue
            h_merged, w_merged = grid
        else:
            print(f"  skip attn vis [{condition}] example {i}: missing grid info")
            continue
        n_spatial = h_merged * w_merged

        signal_pairs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        missing_signals: List[str] = []

        for signal in requested_signals:
            if signal == "decoder":
                sig_a = _normalize_spatial_signal(out_a.get("decoder_image_attn"), n_spatial)
                sig_b = _normalize_spatial_signal(out_b.get("decoder_image_attn"), n_spatial)
            elif signal == "a_vis":
                sig_a = _mean_a_vis(out_a.get("reinspection_attn_vis"), n_spatial)
                sig_b = _mean_a_vis(out_b.get("reinspection_attn_vis"), n_spatial)
            else:
                sig_a = _normalize_spatial_signal(out_a.get("base_vision_attn"), n_spatial)
                sig_b = _normalize_spatial_signal(out_b.get("base_vision_attn"), n_spatial)

            if sig_a is None or sig_b is None:
                missing_signals.append(signal)
                continue
            signal_pairs[signal] = (sig_a, sig_b)

        if not signal_pairs:
            requested = ", ".join(requested_signals)
            print(f"  skip attn vis [{condition}] example {i}: missing all requested signals ({requested})")
            continue

        if missing_signals:
            print(
                f"  [{condition}] example {i}: missing signals "
                f"{', '.join(missing_signals)}; saving {', '.join(signal_pairs)}"
            )

        available_signals = list(signal_pairs.keys())
        default_signal = _default_display_signal(condition, available_signals)
        default_a, default_b = signal_pairs[default_signal]

        npz_path = os.path.join(fig_dir, f"{condition}_attn_pair_{i}.npz")
        save_payload = {
            "attn_a": default_a,
            "attn_b": default_b,
            "h_merged": h_merged,
            "w_merged": w_merged,
            "image_path": image_path,
            "question_a": pair.question_a,
            "question_b": pair.question_b,
            "answer_a": out_a["generated_text"],
            "answer_b": out_b["generated_text"],
            "gt_a": pair.answer_a,
            "gt_b": pair.answer_b,
            "relation": pair.relation,
            "condition": condition,
            "attn_source": default_signal,
            "available_signals": np.asarray(available_signals, dtype="<U32"),
            "default_display_signal": default_signal,
        }
        for signal, (sig_a, sig_b) in signal_pairs.items():
            prefix = ATTN_SIGNAL_STORAGE[signal]
            save_payload[f"{prefix}_a"] = sig_a
            save_payload[f"{prefix}_b"] = sig_b
        np.savez(npz_path, **save_payload)
        saved_paths.append(npz_path)
        print(f"  [{condition}] Saved attention data: {npz_path}")

    return saved_paths


def _plot_single_condition_figure(npz_paths: List[str], fig_dir: str, condition: str) -> None:
    """Render a condition figure using the signals stored in the NPZ schema."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.utils.visualize_attention import plot_attention_heatmap

    first = np.load(npz_paths[0], allow_pickle=True)
    display_signals = _single_condition_display_signals(
        condition, _npz_available_signals(first),
    )
    n = len(npz_paths)
    n_cols = 1 + (2 * len(display_signals))
    fig, axes = plt.subplots(n, n_cols, figsize=(6 + 4 * len(display_signals), 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Image"]
    for signal in display_signals:
        label = ATTN_SIGNAL_LABELS.get(signal, signal)
        col_titles.extend([f"{label} · Q_A", f"{label} · Q_B"])
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10, fontweight="bold")

    for row, npz_path in enumerate(npz_paths):
        data = np.load(npz_path, allow_pickle=True)
        image = Image.open(str(data["image_path"])).convert("RGB")
        h_m, w_m = int(data["h_merged"]), int(data["w_merged"])
        q_a, q_b = str(data["question_a"]), str(data["question_b"])
        ans_a, ans_b = str(data["answer_a"]), str(data["answer_b"])
        gt_a, gt_b = str(data["gt_a"]), str(data["gt_b"])

        axes[row, 0].imshow(image)
        axes[row, 0].set_title(
            f"{os.path.basename(str(data['image_path']))}\n{str(data['relation'])}",
            fontsize=9,
        )
        axes[row, 0].axis("off")

        a_ok = "Y" if ans_a.strip().lower() == gt_a.strip().lower() else "N"
        b_ok = "Y" if ans_b.strip().lower() == gt_b.strip().lower() else "N"
        col = 1
        for signal in display_signals:
            sig_a, sig_b = _npz_signal_pair(data, signal)
            if sig_a is None or sig_b is None:
                axes[row, col].axis("off")
                axes[row, col + 1].axis("off")
                col += 2
                continue
            cmap = ATTN_SIGNAL_CMAP.get(signal, "cividis")
            plot_attention_heatmap(
                image, sig_a, h_m, w_m,
                title=f"Q: ...{q_a[-50:]}\nA: {ans_a} (GT={gt_a}) [{a_ok}]",
                ax=axes[row, col], cmap=cmap, alpha=0.5,
            )
            plot_attention_heatmap(
                image, sig_b, h_m, w_m,
                title=f"Q: ...{q_b[-50:]}\nA: {ans_b} (GT={gt_b}) [{b_ok}]",
                ax=axes[row, col + 1], cmap=cmap, alpha=0.5,
            )
            col += 2

    label = " + ".join(ATTN_SIGNAL_LABELS.get(sig, sig) for sig in display_signals)
    fig.suptitle(f"{label}: Question A vs. Question B", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = os.path.join(fig_dir, f"{condition}_attention_comparison.pdf")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {condition} attention figure: {out_path}")


def _plot_combined_attention_comparison(
    vis_npzs: Dict[str, List[str]], fig_dir: str,
) -> None:
    """Render one combined figure per matched signal shared by frozen and reinspection."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.utils.visualize_attention import plot_attention_heatmap

    def _idx(path: str) -> int:
        return int(os.path.basename(path).split("_pair_")[-1].split(".")[0])

    frozen_by_idx = {_idx(p): p for p in vis_npzs.get("frozen", [])}
    ri_by_idx = {_idx(p): p for p in vis_npzs.get("reinspection", [])}
    common = sorted(set(frozen_by_idx) & set(ri_by_idx))

    if not common:
        print("  No matching pairs between frozen and reinspection — skipping combined figure.")
        return

    frozen_first = np.load(frozen_by_idx[common[0]], allow_pickle=True)
    ri_first = np.load(ri_by_idx[common[0]], allow_pickle=True)
    shared_signals = [
        signal
        for signal in _npz_available_signals(frozen_first)
        if signal in _npz_available_signals(ri_first)
    ]
    if not shared_signals:
        print("  Frozen and reinspection figures do not share an attention signal — skipping combined figure.")
        return

    def _label(q, ans, gt):
        ok = "Y" if str(ans).strip().lower() == str(gt).strip().lower() else "N"
        return f"Q: {str(q)[-50:]}\nA: {ans} (GT={gt}) [{ok}]"

    n = len(common)
    for signal in shared_signals:
        fig, axes = plt.subplots(n, 5, figsize=(22, 4 * n))
        if n == 1:
            axes = axes[np.newaxis, :]

        col_titles = ["Image", "Frozen Q_A", "Frozen Q_B", "Re-Insp Q_A", "Re-Insp Q_B"]
        for col, title in enumerate(col_titles):
            axes[0, col].set_title(title, fontsize=10, fontweight="bold")

        cmap = ATTN_SIGNAL_CMAP.get(signal, "cividis")
        for row, idx in enumerate(common):
            fd = np.load(frozen_by_idx[idx], allow_pickle=True)
            rd = np.load(ri_by_idx[idx], allow_pickle=True)
            image = Image.open(str(fd["image_path"])).convert("RGB")
            h_m, w_m = int(fd["h_merged"]), int(fd["w_merged"])
            frozen_a, frozen_b = _npz_signal_pair(fd, signal)
            ri_a, ri_b = _npz_signal_pair(rd, signal)
            if frozen_a is None or frozen_b is None or ri_a is None or ri_b is None:
                continue

            axes[row, 0].imshow(image)
            axes[row, 0].set_title(
                f"{os.path.basename(str(fd['image_path']))}\n{str(fd['relation'])}", fontsize=8,
            )
            axes[row, 0].axis("off")

            plot_attention_heatmap(image, frozen_a, h_m, w_m,
                                   title=_label(fd["question_a"], fd["answer_a"], fd["gt_a"]),
                                   ax=axes[row, 1], cmap=cmap, alpha=0.5)
            plot_attention_heatmap(image, frozen_b, h_m, w_m,
                                   title=_label(fd["question_b"], fd["answer_b"], fd["gt_b"]),
                                   ax=axes[row, 2], cmap=cmap, alpha=0.5)
            plot_attention_heatmap(image, ri_a, h_m, w_m,
                                   title=_label(rd["question_a"], rd["answer_a"], rd["gt_a"]),
                                   ax=axes[row, 3], cmap=cmap, alpha=0.5)
            plot_attention_heatmap(image, ri_b, h_m, w_m,
                                   title=_label(rd["question_b"], rd["answer_b"], rd["gt_b"]),
                                   ax=axes[row, 4], cmap=cmap, alpha=0.5)

        signal_label = ATTN_SIGNAL_LABELS.get(signal, signal)
        if signal == "decoder":
            suptitle = "Frozen VLM vs. Re-Inspection: LLM decoder attention over image tokens"
        else:
            suptitle = f"Frozen VLM vs. Re-Inspection: {signal_label}"
        fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        suffix = "" if len(shared_signals) == 1 else f"_{signal}"
        out_path = os.path.join(fig_dir, f"combined_attention_comparison{suffix}.pdf")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        fig.savefig(out_path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved combined attention comparison: {out_path}")


# ------------------------------------------------------------------ #
#  Summary tables                                                      #
# ------------------------------------------------------------------ #

def print_flip_summary(all_results: Dict[str, Dict], output_file: Optional[str] = None) -> None:
    """Print a formatted summary table of flip consistency across conditions."""
    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("FLIP CONSISTENCY RESULTS")
    lines.append("=" * 80)

    for condition, res in all_results.items():
        n = res["total_pairs"]
        if n == 0:
            continue
        lines.append(f"\nCondition: {condition}")
        lines.append(f"  Total pairs:      {n}")
        lines.append(f"  Flip consistency: {res['consistent']/n:.1%} ({res['consistent']}/{n})")
        lines.append(f"  Both correct:     {res['both_correct']/n:.1%} ({res['both_correct']}/{n})")
        lines.append(f"  Same answer:      {res['same_answer']/n:.1%} ({res['same_answer']}/{n})  ← failure mode")
        lines.append(f"  Accuracy (A):     {res['a_correct']/n:.1%}")
        lines.append(f"  Accuracy (B):     {res['b_correct']/n:.1%}")

        # Per-relation breakdown
        if res.get("per_relation"):
            lines.append(f"\n  {'Relation':<20} {'Consist.':>9} {'Both OK':>9} {'Same Ans':>9} {'N':>5}")
            lines.append(f"  {'-'*20} {'-'*9} {'-'*9} {'-'*9} {'-'*5}")
            for rel, stats in sorted(res["per_relation"].items()):
                rn = stats["total"]
                if rn == 0:
                    continue
                lines.append(
                    f"  {rel:<20} {stats['consistent']/rn:>8.1%} {stats['both_correct']/rn:>8.1%} "
                    f"{stats['same_answer']/rn:>8.1%} {rn:>5}"
                )

    # Delta table if we have both conditions
    if "frozen" in all_results and "reinspection" in all_results:
        fr = all_results["frozen"]
        ri = all_results["reinspection"]
        if fr["total_pairs"] > 0 and ri["total_pairs"] > 0:
            n_fr, n_ri = fr["total_pairs"], ri["total_pairs"]
            lines.append(f"\n{'='*80}")
            lines.append("DELTA (reinspection - frozen)")
            lines.append(f"  Flip consistency: {ri['consistent']/n_ri - fr['consistent']/n_fr:+.1%}")
            lines.append(f"  Both correct:     {ri['both_correct']/n_ri - fr['both_correct']/n_fr:+.1%}")
            lines.append(f"  Same answer:      {ri['same_answer']/n_ri - fr['same_answer']/n_fr:+.1%}")

    lines.append("")
    table_str = "\n".join(lines)
    print(table_str)

    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(table_str + "\n")


def print_divergence_summary(all_results: Dict[str, Dict], output_file: Optional[str] = None) -> None:
    """Print attention divergence summary."""
    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("ATTENTION / REPRESENTATION DIVERGENCE")
    lines.append("=" * 80)

    for condition, res in all_results.items():
        lines.append(f"\nCondition: {condition}  (N={res['num_pairs']})")
        if res["mean_jsd"] is not None:
            lines.append(f"  Re-Inspection A_vis JSD:  {res['mean_jsd']:.4f} +/- {res['std_jsd']:.4f}")
            lines.append(f"    (higher = more question-conditioned visual attention)")
            if res.get("per_relation_jsd"):
                lines.append(f"\n    {'Relation':<20} {'Mean JSD':>10} {'Std':>8} {'N':>5}")
                lines.append(f"    {'-'*20} {'-'*10} {'-'*8} {'-'*5}")
                for rel, stats in res["per_relation_jsd"].items():
                    lines.append(f"    {rel:<20} {stats['mean']:>10.4f} {stats['std']:>8.4f} {stats['n']:>5}")
        if res["mean_cosine_sim"] is not None:
            lines.append(f"  Hidden-state cosine sim:  {res['mean_cosine_sim']:.4f} +/- {res['std_cosine_sim']:.4f}")
            lines.append(f"    (lower = more question-conditioned representations)")

    # Delta
    if "frozen" in all_results and "reinspection" in all_results:
        fr, ri = all_results["frozen"], all_results["reinspection"]
        lines.append(f"\n{'='*80}")
        lines.append("DELTA (reinspection - frozen)")
        if fr["mean_cosine_sim"] is not None and ri["mean_cosine_sim"] is not None:
            lines.append(f"  Cosine sim: {ri['mean_cosine_sim'] - fr['mean_cosine_sim']:+.4f}")
        if fr["mean_jsd"] is not None and ri["mean_jsd"] is not None:
            lines.append(f"  A_vis JSD:  {ri['mean_jsd'] - fr['mean_jsd']:+.4f}")

    lines.append("")
    table_str = "\n".join(lines)
    print(table_str)

    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(table_str + "\n")


def print_mirror_summary(all_results: Dict[str, Dict], output_file: Optional[str] = None) -> None:
    """Print mirror test summary."""
    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("MIRROR TEST RESULTS (left/right relations only)")
    lines.append("=" * 80)

    for condition, res in all_results.items():
        n = res["total"]
        if n == 0:
            continue
        lines.append(f"\nCondition: {condition}")
        lines.append(f"  Total pairs:          {n}")
        lines.append(f"  Mirror consistency:   {res['mirror_consistent']/n:.1%} ({res['mirror_consistent']}/{n})")
        lines.append(f"  Mirror correct:       {res['mirror_correct']/n:.1%} ({res['mirror_correct']}/{n})")
        lines.append(f"  Orig accuracy:        {res['orig_correct']/n:.1%}")

        if res.get("per_relation"):
            lines.append(f"\n  {'Relation':<25} {'Mirror Cons.':>13} {'Mirror Corr.':>13} {'N':>5}")
            lines.append(f"  {'-'*25} {'-'*13} {'-'*13} {'-'*5}")
            for rel, stats in sorted(res["per_relation"].items()):
                rn = stats["total"]
                if rn == 0:
                    continue
                lines.append(
                    f"  {rel:<25} {stats['mirror_consistent']/rn:>12.1%} "
                    f"{stats['mirror_correct']/rn:>12.1%} {rn:>5}"
                )

    # Delta
    if "frozen" in all_results and "reinspection" in all_results:
        fr, ri = all_results["frozen"], all_results["reinspection"]
        if fr["total"] > 0 and ri["total"] > 0:
            lines.append(f"\n{'='*80}")
            lines.append("DELTA (reinspection - frozen)")
            lines.append(f"  Mirror consistency: {ri['mirror_consistent']/ri['total'] - fr['mirror_consistent']/fr['total']:+.1%}")
            lines.append(f"  Mirror correct:     {ri['mirror_correct']/ri['total'] - fr['mirror_correct']/fr['total']:+.1%}")

    lines.append("")
    table_str = "\n".join(lines)
    print(table_str)

    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(table_str + "\n")


# ------------------------------------------------------------------ #
#  W&B integration                                                     #
# ------------------------------------------------------------------ #

def _init_wandb(config: ReInspectionConfig) -> None:
    try:
        import wandb
        os.environ["WANDB_SILENT"] = "true"
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name or f"motivation_{config.backend}",
            config=asdict(config),
            job_type="motivation",
            tags=[config.backend, "motivation"],
            save_code=True,
        )
    except Exception:
        pass


def _log_wandb(metrics: dict) -> None:
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(metrics)
    except Exception:
        pass


def _finish_wandb() -> None:
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass


# ------------------------------------------------------------------ #
#  Main entry                                                          #
# ------------------------------------------------------------------ #

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    config = ReInspectionConfig(**OmegaConf.to_container(cfg, resolve=True))
    backend = config.backend

    # --- Load processor ---
    processor = None
    if backend == "internvl3":
        from src.backends.internvl3 import load_processor
        processor = load_processor(config)
    elif backend == "gemma4":
        from src.backends.gemma4 import load_processor as load_gemma4_proc
        processor = load_gemma4_proc(config)
    else:
        processor = AutoProcessor.from_pretrained(
            config.model_name_or_path,
            max_pixels=config.max_pixels,
            min_pixels=config.min_pixels,
        )

    # --- Generate minimal pairs from VSR ---
    vsr_file = os.path.join(config.data_root, "vsr", "test.jsonl")
    image_root = os.path.join(config.data_root, "vsr", "images")
    if not os.path.exists(vsr_file):
        print(f"ERROR: VSR data not found at {vsr_file}")
        return

    all_pairs = generate_pairs_from_vsr(vsr_file, split="test")
    # Use only invert_rel pairs (cleanest test — same subject/object, opposite relation)
    inv_pairs = [p for p in all_pairs if p.pair_type == "invert_rel"]
    swap_pairs = [p for p in all_pairs if p.pair_type == "swap_args"]

    print(f"\n{'='*60}")
    print(f"Motivation Experiment: {backend}")
    print(f"Model: {config.model_name_or_path}")
    print(f"Minimal pairs: {len(inv_pairs)} invert_rel + {len(swap_pairs)} swap_args = {len(all_pairs)} total")
    print(f"{'='*60}\n")

    _init_wandb(config)

    # --- Conditions to evaluate ---
    conditions = ["frozen"]
    if config.checkpoint_dir is not None:
        conditions.append("reinspection")

    out_dir = os.path.dirname(config.output_file) or "outputs/motivation"
    os.makedirs(out_dir, exist_ok=True)

    flip_results: Dict[str, Dict] = {}
    mirror_results: Dict[str, Dict] = {}
    div_results: Dict[str, Dict] = {}
    vis_npzs: Dict[str, List[str]] = {}

    max_pairs = config.max_samples

    # Select visualization examples once so both conditions use the same images.
    vis_selected = _select_vis_examples(inv_pairs, image_root)

    for condition in conditions:
        print(f"\n{'='*60}")
        print(f"Condition: {condition}")
        print(f"{'='*60}")

        # Eager attention is required for output_attentions=True in generate().
        attn_impl = "eager" if backend == "internvl3" else None
        model, is_ri = load_condition_model(
            backend, condition, config, processor,
            checkpoint_dir=config.checkpoint_dir,
            lora_checkpoint_dir=config.lora_checkpoint_dir,
            attn_implementation=attn_impl,
        )
        model.eval()

        # --- Prong 1: Flip Consistency (use all pairs) ---
        print(f"\n--- Prong 1: Flip Consistency ({len(all_pairs)} pairs) ---")
        flip_res = run_flip_consistency(
            model, processor, all_pairs, backend, config, is_ri,
            image_root=image_root, max_pairs=max_pairs,
        )
        flip_results[condition] = flip_res

        n = flip_res["total_pairs"]
        if n > 0:
            wandb_metrics = {
                f"motivation/{condition}/flip_consistency": flip_res["consistent"] / n,
                f"motivation/{condition}/both_correct": flip_res["both_correct"] / n,
                f"motivation/{condition}/same_answer_rate": flip_res["same_answer"] / n,
                f"motivation/{condition}/accuracy_a": flip_res["a_correct"] / n,
                f"motivation/{condition}/accuracy_b": flip_res["b_correct"] / n,
            }
            _log_wandb(wandb_metrics)

        # Save per-sample results
        samples_file = os.path.join(out_dir, f"{condition}_flip_samples.json")
        with open(samples_file, "w", encoding="utf-8") as f:
            json.dump(flip_res.pop("samples", []), f, indent=2)

        # --- Prong 2: Mirror Test (left/right relations only) ---
        lr_pairs = [p for p in all_pairs if p.relation in MIRROR_RELATIONS]
        print(f"\n--- Prong 2: Mirror Test ({len(lr_pairs)} left/right pairs) ---")
        mirror_res = run_mirror_test(
            model, processor, lr_pairs, backend, config, is_ri,
            image_root=image_root, max_pairs=max_pairs,
        )
        mirror_results[condition] = mirror_res

        mn = mirror_res["total"]
        if mn > 0:
            mirror_wandb = {
                f"motivation/{condition}/mirror_consistency": mirror_res["mirror_consistent"] / mn,
                f"motivation/{condition}/mirror_correct": mirror_res["mirror_correct"] / mn,
            }
            _log_wandb(mirror_wandb)

        mirror_samples_file = os.path.join(out_dir, f"{condition}_mirror_samples.json")
        with open(mirror_samples_file, "w", encoding="utf-8") as f:
            json.dump(mirror_res.pop("samples", []), f, indent=2)

        # --- Prong 3: Attention Divergence (use invert_rel pairs — cleanest) ---
        print(f"\n--- Prong 3: Attention Divergence ({len(inv_pairs)} pairs) ---")
        div_res = run_attention_divergence(
            model, processor, inv_pairs, backend, config, is_ri,
            image_root=image_root, max_pairs=max_pairs,
        )
        div_results[condition] = div_res

        div_wandb = {}
        if div_res["mean_jsd"] is not None:
            div_wandb[f"motivation/{condition}/attn_jsd"] = div_res["mean_jsd"]
        if div_res["mean_cosine_sim"] is not None:
            div_wandb[f"motivation/{condition}/cosine_sim"] = div_res["mean_cosine_sim"]
        if div_wandb:
            _log_wandb(div_wandb)

        div_samples_file = os.path.join(out_dir, f"{condition}_divergence_samples.json")
        with open(div_samples_file, "w", encoding="utf-8") as f:
            json.dump(div_res.pop("samples", []), f, indent=2)

        # --- Prong 4: Attention Map Visualisation (both conditions) ---
        if backend in ("qwen25vl", "internvl3"):
            print(f"\n--- Prong 4: Attention Map Visualisation ({condition}) ---")
            npzs = run_attention_visualization(
                model, processor, vis_selected, backend, config,
                image_root=image_root, output_dir=out_dir, condition=condition,
            )
            if npzs:
                vis_npzs[condition] = npzs
                _plot_single_condition_figure(
                    npzs, os.path.join(out_dir, "attention_figures"), condition,
                )

        del model
        torch.cuda.empty_cache()

    # --- Combined attention figure (if both conditions ran) ---
    if len(vis_npzs) >= 2:
        print("\n--- Generating combined attention comparison figure ---")
        _plot_combined_attention_comparison(vis_npzs, os.path.join(out_dir, "attention_figures"))

    # --- Summary ---
    print_flip_summary(flip_results, os.path.join(out_dir, "flip_consistency_table.txt"))
    print_mirror_summary(mirror_results, os.path.join(out_dir, "mirror_test_table.txt"))
    print_divergence_summary(div_results, os.path.join(out_dir, "attention_divergence_table.txt"))

    # Save full results JSON
    full_results = {
        "backend": backend,
        "model": config.model_name_or_path,
        "flip_consistency": {k: {kk: vv for kk, vv in v.items() if kk != "samples"} for k, v in flip_results.items()},
        "mirror_test": {k: {kk: vv for kk, vv in v.items() if kk != "samples"} for k, v in mirror_results.items()},
        "attention_divergence": {k: {kk: vv for kk, vv in v.items() if kk != "samples"} for k, v in div_results.items()},
    }
    results_file = os.path.join(out_dir, f"motivation_{backend}_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nFull results saved to {results_file}")

    _finish_wandb()


if __name__ == "__main__":
    main()
