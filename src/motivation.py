"""Motivation experiment: Spatial Minimal-Pair Flip Consistency & Attention Divergence.

Demonstrates two architectural limitations of standard VLMs on spatial reasoning:

1. **Flip Consistency** — Given the same image and two logically-inverse spatial
   questions (e.g. "Is A above B?" / "Is B above A?"), how often does the model
   give opposite answers?  A spatially-competent model should approach 100%.

2. **Attention Divergence** — For the same image with different spatial questions,
   how different are the model's attention patterns over vision tokens?  A model
   whose vision encoding is question-agnostic will show low divergence.

Entry point: ``python -m src.motivation stage=eval backend=qwen3vl [overrides]``

Supports all backends (qwen3vl, qwen25vl, internvl3, gemma4) and two conditions:
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
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

import hydra
from omegaconf import DictConfig, OmegaConf

from src.config import ReInspectionConfig
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
from src.hydra_util import strip_deepspeed_local_rank_argv

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


# ------------------------------------------------------------------ #
#  Core generation + attention capture                                #
# ------------------------------------------------------------------ #

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
            is_reinspection, return_grid_info,
        )
    finally:
        if _tmp_file is not None:
            os.unlink(_tmp_file.name)


@torch.no_grad()
def _generate_with_attention_inner(
    model, processor, image_path, question, backend, config,
    is_reinspection, return_grid_info,
) -> Dict:
    device = model_device(model)

    if backend in ("qwen3vl", "qwen25vl"):
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
        inputs = processor(
            text=[text], images=[image_path], return_tensors="pt",
            max_pixels=config.max_pixels, min_pixels=config.min_pixels,
            crop_to_patches=config.crop_to_patches_stage2,
        )

    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # --- Generate ---
    generated_ids = model.generate(
        **inputs, max_new_tokens=16, do_sample=False,
        **_generate_extra_kw(processor),
    )

    # Determine input length for decoding
    if backend in ("qwen3vl", "qwen25vl"):
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
    except Exception:
        pass  # Not critical — skip silently

    result = {
        "generated_text": generated_text,
        "reinspection_attn_vis": ri_attn_vis,
        "base_last_hidden": base_last_hidden,
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

    for i, pair in enumerate(tqdm(pairs, desc="Flip Consistency", miniters=50)):
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
            tqdm.write(f"  skip pair {i}: {e}")
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

    for i, pair in enumerate(tqdm(lr_pairs, desc="Mirror Test", miniters=50)):
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
            tqdm.write(f"  skip mirror pair {i}: {e}")
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

    for i, pair in enumerate(tqdm(pairs, desc="Attention Divergence", miniters=50)):
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
            tqdm.write(f"  skip pair {i}: {e}")
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


@torch.no_grad()
def run_attention_visualization(
    model,
    processor,
    pairs: List[MinimalPair],
    backend: str,
    config: ReInspectionConfig,
    image_root: str,
    output_dir: str,
    n_examples: int = 6,
) -> None:
    """Generate attention heatmap data for side-by-side visualisation.

    For each selected minimal pair, extracts the re-inspection module's A_vis
    attention maps for both questions and saves per-example .npz files plus a
    composite figure.

    Only meaningful for the reinspection condition (base model doesn't expose
    per-vision-token attention).  Qwen backends provide ``image_grid_thw`` for
    mapping attention back to a spatial grid; other backends are skipped.
    """
    if backend not in ("qwen3vl", "qwen25vl"):
        print("  Attention visualisation currently supported for Qwen backends only — skipping.")
        return

    fig_dir = os.path.join(output_dir, "attention_figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Try to match hand-picked examples; fall back to first N pairs
    selected: List[MinimalPair] = []
    example_images = {img for img, *_ in ATTN_VIS_EXAMPLES}
    for pair in pairs:
        if pair.image in example_images and len(selected) < n_examples:
            selected.append(pair)
    if len(selected) < n_examples:
        for pair in pairs:
            if pair not in selected:
                selected.append(pair)
            if len(selected) >= n_examples:
                break

    model.eval()
    saved_data = []

    for i, pair in enumerate(selected):
        image_path = os.path.join(image_root, pair.image)
        if not os.path.isfile(image_path):
            continue

        try:
            out_a = _generate_with_attention(
                model, processor, image_path, pair.question_a,
                backend, config, True, return_grid_info=True,
            )
            out_b = _generate_with_attention(
                model, processor, image_path, pair.question_b,
                backend, config, True, return_grid_info=True,
            )
        except Exception as e:
            print(f"  skip attn vis example {i}: {e}")
            continue

        attn_a = out_a.get("reinspection_attn_vis")
        attn_b = out_b.get("reinspection_attn_vis")
        grid_thw = out_a.get("image_grid_thw")

        if attn_a is None or attn_b is None or grid_thw is None:
            print(f"  skip attn vis example {i}: missing attention or grid info")
            continue

        _, h, w = grid_thw
        spatial_merge = 2
        h_merged = h // spatial_merge
        w_merged = w // spatial_merge

        # Average across queries: (B, N_q, N_vis) → (N_vis,)
        agg_a = attn_a[0].mean(axis=0)[:h_merged * w_merged]
        agg_b = attn_b[0].mean(axis=0)[:h_merged * w_merged]
        # Normalise to sum to 1
        agg_a = agg_a / (agg_a.sum() + 1e-8)
        agg_b = agg_b / (agg_b.sum() + 1e-8)

        # Save per-example data
        npz_path = os.path.join(fig_dir, f"attn_pair_{i}.npz")
        np.savez(
            npz_path,
            attn_a=agg_a, attn_b=agg_b,
            h_merged=h_merged, w_merged=w_merged,
            image_path=image_path,
            question_a=pair.question_a, question_b=pair.question_b,
            answer_a=out_a["generated_text"], answer_b=out_b["generated_text"],
            gt_a=pair.answer_a, gt_b=pair.answer_b,
            relation=pair.relation,
        )

        saved_data.append({
            "npz": npz_path,
            "image": pair.image,
            "relation": pair.relation,
        })

        print(f"  Saved attention data: {npz_path}")

    # Generate composite figure
    if saved_data:
        _plot_attention_comparison(saved_data, fig_dir)


def _plot_attention_comparison(saved_data: List[Dict], fig_dir: str) -> None:
    """Render a composite attention comparison figure from saved .npz files."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.visualize_attention import plot_attention_heatmap

    n = len(saved_data)
    fig, axes = plt.subplots(n, 3, figsize=(14, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, entry in enumerate(saved_data):
        data = np.load(entry["npz"], allow_pickle=True)
        image = Image.open(str(data["image_path"])).convert("RGB")
        h_m = int(data["h_merged"])
        w_m = int(data["w_merged"])
        q_a = str(data["question_a"])
        q_b = str(data["question_b"])
        ans_a = str(data["answer_a"])
        ans_b = str(data["answer_b"])
        gt_a = str(data["gt_a"])
        gt_b = str(data["gt_b"])

        # Column 0: original image
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f"Image: {entry['image']}\nRelation: {entry['relation']}",
                                fontsize=9)
        axes[row, 0].axis("off")

        # Column 1: attention for question A
        a_ok = "Y" if ans_a.strip().lower() == gt_a.strip().lower() else "N"
        plot_attention_heatmap(
            image, data["attn_a"], h_m, w_m,
            title=f"Q: ...{q_a[-50:]}\nA: {ans_a} (GT={gt_a}) [{a_ok}]",
            ax=axes[row, 1], cmap="hot", alpha=0.5,
        )

        # Column 2: attention for question B
        b_ok = "Y" if ans_b.strip().lower() == gt_b.strip().lower() else "N"
        plot_attention_heatmap(
            image, data["attn_b"], h_m, w_m,
            title=f"Q: ...{q_b[-50:]}\nA: {ans_b} (GT={gt_b}) [{b_ok}]",
            ax=axes[row, 2], cmap="hot", alpha=0.5,
        )

    fig.suptitle("Re-Inspection Attention: Question A vs. Question B",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = os.path.join(fig_dir, "attention_comparison.pdf")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved attention comparison figure: {out_path}")


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

    max_pairs = config.max_samples

    for condition in conditions:
        print(f"\n{'='*60}")
        print(f"Condition: {condition}")
        print(f"{'='*60}")

        model, is_ri = load_condition_model(
            backend, condition, config, processor,
            checkpoint_dir=config.checkpoint_dir,
            lora_checkpoint_dir=config.lora_checkpoint_dir,
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

        # --- Prong 4: Attention Visualization (reinspection only) ---
        if is_ri:
            print(f"\n--- Prong 4: Attention Map Visualisation ---")
            run_attention_visualization(
                model, processor, inv_pairs, backend, config,
                image_root=image_root, output_dir=out_dir,
            )

        del model
        torch.cuda.empty_cache()

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
