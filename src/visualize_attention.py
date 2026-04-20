"""Attention visualization for VLM + Re-Inspection.

Produces two complementary views:
  1. LLM decoder self-attention from generated tokens → image patches.
     This is the standard VLM attention visualization (per HF guide):
     run generate() with `output_attentions=True`, aggregate heads/layers,
     index the columns corresponding to image tokens, reshape to the patch
     grid, and overlay on the image. Requires `attn_implementation="eager"`.
  2. Internal ReInspectionModule cross-attention (A_vis): R queries → V.
     This shows what the bottleneck module focuses on in its own latent
     space — it is *not* the same signal as the decoder's attention.

The decoder view is currently implemented for Qwen3-VL only (other backends
pre-build `inputs_embeds` in different ways; scope them separately).
"""

import argparse
import importlib
import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.config import ReInspectionConfig


def _load_backend(backend: str):
    """Return (load_model_fn, build_chat_messages_fn) for the requested backend."""
    if backend in ("qwen3vl", "qwen25vl"):
        backend_mod = importlib.import_module(f"src.backends.{backend}")
        chat_mod = importlib.import_module("src.data.utils")
    elif backend == "internvl3":
        backend_mod = importlib.import_module("src.backends.internvl3")
        chat_mod = importlib.import_module("src.data.chat_template")
    elif backend == "gemma4":
        backend_mod = importlib.import_module("src.backends.gemma4")
        chat_mod = importlib.import_module("src.data.gemma4_chat")
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    return backend_mod.load_model, chat_mod.build_chat_messages


def _load_processor(backend: str, config: ReInspectionConfig):
    from transformers import AutoProcessor

    path = config.processor_path
    if backend in ("qwen3vl", "qwen25vl"):
        return AutoProcessor.from_pretrained(
            path, max_pixels=config.max_pixels, min_pixels=config.min_pixels
        )
    return AutoProcessor.from_pretrained(path)


def _process_inputs(backend: str, processor, text: str, image_path: str, config: ReInspectionConfig):
    kwargs = dict(text=[text], images=[image_path], return_tensors="pt")
    if backend in ("qwen3vl", "qwen25vl"):
        kwargs["max_pixels"] = config.max_pixels
        kwargs["min_pixels"] = config.min_pixels
    elif backend == "internvl3":
        # Single-tile for viz: tiled InternVL has no consistent square layout.
        kwargs["crop_to_patches"] = False
    return processor(**kwargs)


def _vision_grid(backend: str, model, inputs, fallback_n: int) -> Tuple[int, int]:
    """Return (h_merged, w_merged) for the vision token grid."""
    if backend in ("qwen3vl", "qwen25vl") and "image_grid_thw" in inputs:
        _, h, w = inputs["image_grid_thw"][0].tolist()
        spatial_merge = 2
        return h // spatial_merge, w // spatial_merge

    base = getattr(model, "base_model", model)
    cfg = getattr(base, "config", None)
    img_tok = getattr(cfg, "image_token_id", None)
    n_vis = int((inputs["input_ids"][0] == img_tok).sum().item()) if img_tok is not None else fallback_n

    if backend == "internvl3":
        vcfg = getattr(cfg, "vision_config", None)
        img_size = vcfg.image_size[0] if isinstance(vcfg.image_size, (list, tuple)) else vcfg.image_size
        patch = vcfg.patch_size[0] if isinstance(vcfg.patch_size, (list, tuple)) else vcfg.patch_size
        downsample = getattr(cfg, "downsample_ratio", 0.5)
        side = int(round((img_size // patch) * downsample))
        if n_vis != side * side:
            raise ValueError(
                f"InternVL3 vision token count {n_vis} != expected {side*side} "
                f"(image_size={img_size}, patch={patch}, downsample={downsample})."
            )
        return side, side

    side = max(1, int(round(n_vis ** 0.5)))
    return side, side


def _decode_generation(backend: str, model, processor, inputs, generated_ids, n_queries: int) -> str:
    if backend in ("qwen3vl", "qwen25vl"):
        input_len = inputs["input_ids"].shape[1] + n_queries
    elif (
        hasattr(model, "last_generation_prompt_lengths")
        and model.last_generation_prompt_lengths is not None
    ):
        input_len = int(model.last_generation_prompt_lengths[0].item())
    else:
        input_len = int(inputs["input_ids"].shape[1])
    return processor.batch_decode(
        generated_ids[:, input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


# ---------------------------------------------------------------------------
# Decoder self-attention → image patches
# ---------------------------------------------------------------------------

def _extract_decoder_image_attention(
    model,
    processor,
    sequences: torch.Tensor,
    attentions: Tuple[Tuple[torch.Tensor, ...], ...],
    h_patches: int,
    w_patches: int,
    n_show: int = 8,
) -> Tuple[Optional[np.ndarray], List[np.ndarray], List[str], List[int]]:
    """Aggregate decoder self-attention from each generated token over image tokens.

    `attentions` is the `GenerateDecoderOnlyOutput.attentions` structure:
      - tuple of length `num_generated_tokens`
      - each element is a tuple over decoder layers
      - each layer tensor has shape [B, heads, q_len, k_len]
      - step 0 (prefill): q_len = prefill_len; token t's logits come from row -1
      - step t > 0: q_len = 1; row 0 is the only row, k_len = prefill_len + t

    We average heads, then layers, then index columns at image-token positions.
    """
    if attentions is None or len(attentions) == 0:
        return None, [], [], []

    img_tok = model._image_token_id
    seq0 = sequences[0]
    img_positions = (seq0 == img_tok).nonzero(as_tuple=True)[0]
    n_img = img_positions.numel()
    expected = h_patches * w_patches
    if n_img != expected:
        raise ValueError(
            f"Image token count in sequence ({n_img}) != grid {h_patches}×{w_patches}={expected}. "
            "Patch-grid assumption is off — refusing to produce a misleading heatmap."
        )

    prefill_len = attentions[0][0].shape[-1]
    if int(img_positions.max().item()) >= prefill_len:
        raise ValueError("Image positions extend past prefill length — unexpected.")

    num_gen = len(attentions)
    per_token_grids: List[np.ndarray] = []
    for t, layer_attns in enumerate(attentions):
        layer_rows = []
        for la in layer_attns:
            # Pull the query row that produced generated token t.
            if t == 0:
                row = la[0, :, -1, :]  # [heads, prefill_len]
            else:
                row = la[0, :, 0, :]   # [heads, prefill_len + t]
            # Restrict to image columns and avg heads.
            img_attn = row[:, img_positions].float().mean(dim=0)  # [n_img]
            layer_rows.append(img_attn)
        grid = torch.stack(layer_rows, dim=0).mean(dim=0)  # avg over layers
        per_token_grids.append(grid.cpu().numpy().reshape(h_patches, w_patches))

    # Token labels and a "valid" mask that drops pure-whitespace / special tokens.
    gen_token_ids = seq0[prefill_len: prefill_len + num_gen]
    tokenizer = getattr(processor, "tokenizer", processor)
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    token_strs: List[str] = []
    valid_indices: List[int] = []
    for i, tok in enumerate(gen_token_ids.tolist()):
        s = tokenizer.decode([tok], skip_special_tokens=False)
        token_strs.append(s)
        if tok in special_ids:
            continue
        if not s.strip():
            continue
        valid_indices.append(i)

    if valid_indices:
        mean_over_answer = np.stack(
            [per_token_grids[i] for i in valid_indices], axis=0
        ).mean(axis=0)
    else:
        mean_over_answer = per_token_grids[0]

    show_idx = valid_indices[:n_show] if valid_indices else list(range(min(n_show, num_gen)))
    return mean_over_answer, [per_token_grids[i] for i in show_idx], [token_strs[i] for i in show_idx], show_idx


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_attention_heatmap(
    image: Image.Image,
    attn_map: np.ndarray,
    h_patches: int,
    w_patches: int,
    title: str = "",
    ax=None,
    cmap: str = "jet",
    alpha: float = 0.5,
    upper_pct: float = 99.0,
    lower_pct: float = 0.0,
    gamma: float = 1.0,
):
    """Overlay an attention map on the image with percentile-based contrast."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 6))

    attn_2d = np.asarray(attn_map, dtype=np.float32).reshape(h_patches, w_patches)
    img_w, img_h = image.size
    attn_resized = np.array(
        Image.fromarray(attn_2d, mode="F").resize((img_w, img_h), Image.BICUBIC)
    )

    lo = float(np.percentile(attn_resized, lower_pct))
    hi = float(np.percentile(attn_resized, upper_pct))
    if hi <= lo:
        hi = lo + 1e-8
    norm = np.clip((attn_resized - lo) / (hi - lo), 0.0, 1.0)
    norm = np.power(norm, gamma)

    ax.imshow(image)
    ax.imshow(norm, cmap=cmap, alpha=alpha, vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=11)
    ax.axis("off")


def _plot_summary(
    image: Image.Image,
    decoder_mean: Optional[np.ndarray],
    a_vis_mean: Optional[np.ndarray],
    h_patches: int,
    w_patches: int,
    question: str,
    answer: str,
    save_path: str,
):
    panels = [("Original Image", None)]
    if decoder_mean is not None:
        panels.append(("LLM decoder → image\n(mean over answer tokens)", decoder_mean))
    if a_vis_mean is not None:
        panels.append(("ReInspection A_vis\n(R queries → V, mean)", a_vis_mean))

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 6))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, arr) in zip(axes, panels):
        if arr is None:
            ax.imshow(image)
            ax.set_title(title, fontsize=12)
            ax.axis("off")
        else:
            plot_attention_heatmap(image, arr, h_patches, w_patches, title=title, ax=ax)

    fig.suptitle(f"Q: {question}\nA: {answer}", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def _plot_per_token_decoder(
    image: Image.Image,
    maps: List[np.ndarray],
    labels: List[str],
    h_patches: int,
    w_patches: int,
    save_path: str,
):
    if not maps:
        return
    n = len(maps)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).flatten() if n > 1 else np.array([axes])
    for i, (m, lbl) in enumerate(zip(maps, labels)):
        plot_attention_heatmap(
            image, m, h_patches, w_patches,
            title=f"gen[{i}]: {lbl!r}", ax=axes[i],
        )
    for i in range(n, len(axes)):
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def _plot_per_query_avis(
    image: Image.Image,
    attn_per_query: np.ndarray,
    h_patches: int,
    w_patches: int,
    save_path: str,
    n_show: int = 8,
):
    n_q = min(attn_per_query.shape[0], n_show)
    cols = min(4, n_q)
    rows = (n_q + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).flatten() if n_q > 1 else np.array([axes])
    for i in range(n_q):
        plot_attention_heatmap(
            image, attn_per_query[i], h_patches, w_patches,
            title=f"R query {i}", ax=axes[i],
        )
    for i in range(n_q, len(axes)):
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main visualization entry
# ---------------------------------------------------------------------------

@torch.no_grad()
def visualize_single(
    model,
    processor,
    image_path: str,
    question: str,
    config: ReInspectionConfig,
    backend: str,
    build_chat_messages,
    save_dir: str = "figures",
    sample_name: str = "sample",
    max_new_tokens: int = 64,
    n_show: int = 8,
):
    os.makedirs(save_dir, exist_ok=True)

    messages = build_chat_messages(question, image_path=image_path)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = _process_inputs(backend, processor, text, image_path, config)
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    capture_decoder = (backend == "qwen3vl")

    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
    if capture_decoder:
        gen_kwargs.update(output_attentions=True, return_dict_in_generate=True)

    gen_out = model.generate(**inputs, **gen_kwargs)
    if capture_decoder:
        generated_ids = gen_out.sequences
        decoder_attentions = gen_out.attentions
        if decoder_attentions is None or decoder_attentions[0][0] is None:
            raise RuntimeError(
                "output_attentions returned None — the model is not using eager "
                "attention. Reload with attn_implementation='eager'."
            )
    else:
        generated_ids = gen_out
        decoder_attentions = None

    generated_text = _decode_generation(
        backend, model, processor, inputs, generated_ids, config.n_queries
    )

    # Internal ReInspection A_vis (always available after generate()).
    _, attn_vis = model.get_attention_maps()

    # Determine vision grid. Prefer `image_grid_thw` for Qwen; fallback to A_vis width.
    fallback_n = attn_vis.shape[-1] if attn_vis is not None else 0
    h_merged, w_merged = _vision_grid(backend, model, inputs, fallback_n)
    n_patches = h_merged * w_merged

    image = Image.open(image_path).convert("RGB")

    # A_vis aggregation: mean over queries, normalize.
    a_vis_mean = None
    attn_per_query = None
    if attn_vis is not None:
        av = attn_vis.detach().float().cpu()
        a_vis_mean = av[0].mean(dim=0).numpy()[:n_patches]
        a_vis_mean = a_vis_mean / (a_vis_mean.sum() + 1e-8)
        attn_per_query = av[0].numpy()[:, :n_patches]

    decoder_mean = None
    per_token_maps: List[np.ndarray] = []
    per_token_labels: List[str] = []
    if capture_decoder and decoder_attentions is not None:
        decoder_mean, per_token_maps, per_token_labels, _ = _extract_decoder_image_attention(
            model, processor, generated_ids, decoder_attentions,
            h_merged, w_merged, n_show=n_show,
        )

    # Summary figure: original + decoder-mean + A_vis-mean.
    _plot_summary(
        image, decoder_mean, a_vis_mean, h_merged, w_merged,
        question, generated_text,
        os.path.join(save_dir, f"{sample_name}_attention.pdf"),
    )

    if per_token_maps:
        _plot_per_token_decoder(
            image, per_token_maps, per_token_labels, h_merged, w_merged,
            os.path.join(save_dir, f"{sample_name}_decoder_per_token.pdf"),
        )

    if attn_per_query is not None:
        _plot_per_query_avis(
            image, attn_per_query, h_merged, w_merged,
            os.path.join(save_dir, f"{sample_name}_per_query.pdf"),
            n_show=n_show,
        )

    print(f"Question: {question}")
    print(f"Answer:   {generated_text}")
    if a_vis_mean is not None:
        print(f"A_vis entropy: {-(a_vis_mean * np.log(a_vis_mean + 1e-8)).sum():.3f}")
    if decoder_mean is not None:
        dm = decoder_mean / (decoder_mean.sum() + 1e-8)
        print(f"Decoder-attn entropy: {-(dm * np.log(dm + 1e-8)).sum():.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="figures")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--sample_name", type=str, default="sample")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--n_show", type=int, default=8)
    parser.add_argument(
        "--backend",
        type=str,
        default="qwen3vl",
        choices=["qwen3vl", "qwen25vl", "internvl3", "gemma4"],
    )
    args = parser.parse_args()

    config = ReInspectionConfig()
    if args.config:
        import yaml

        with open(args.config, "r", encoding="utf-8") as f:
            overrides = yaml.safe_load(f)
        for k, v in overrides.items():
            if hasattr(config, k):
                setattr(config, k, v)

    load_model_fn, build_chat_messages = _load_backend(args.backend)

    # Eager attention is required for `output_attentions=True` in generate().
    # Only Qwen3-VL's load_model currently plumbs this kwarg; other backends
    # fall back to their defaults (decoder viz is skipped for them).
    load_kwargs = dict(device_map="auto")
    if args.backend == "qwen3vl":
        load_kwargs["attn_implementation"] = "eager"
    model = load_model_fn(config, **load_kwargs)

    reinspection_path = os.path.join(args.checkpoint_dir, "reinspection_module.pt")
    if os.path.exists(reinspection_path):
        state_dict = torch.load(reinspection_path, map_location="cpu", weights_only=True)
        model.reinspection.load_state_dict(state_dict)

    lora_path = os.path.join(args.checkpoint_dir, "lora_weights")
    if os.path.exists(lora_path):
        from peft import PeftModel

        model.base_model.model.language_model = PeftModel.from_pretrained(
            model.base_model.model.language_model, lora_path
        )

    processor = _load_processor(args.backend, config)

    visualize_single(
        model, processor,
        image_path=args.image,
        question=args.question,
        config=config,
        backend=args.backend,
        build_chat_messages=build_chat_messages,
        save_dir=args.output_dir,
        sample_name=args.sample_name,
        max_new_tokens=args.max_new_tokens,
        n_show=args.n_show,
    )


if __name__ == "__main__":
    main()
