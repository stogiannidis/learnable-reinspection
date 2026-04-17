"""Attention map visualization for VLM + Re-Inspection (Qwen3-VL / InternVL3)."""

import argparse
import importlib
import os

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
        kwargs["crop_to_patches"] = config.crop_to_patches_stage2
    return processor(**kwargs)


def _vision_grid(backend: str, model, inputs, attn_n_kv: int):
    """Return (h_merged, w_merged) for the vision token grid."""
    if backend in ("qwen3vl", "qwen25vl") and "image_grid_thw" in inputs:
        _, h, w = inputs["image_grid_thw"][0].tolist()
        spatial_merge = 2
        return h // spatial_merge, w // spatial_merge

    base = getattr(model, "base_model", model)
    img_tok = getattr(getattr(base, "config", None), "image_token_id", None)
    if img_tok is not None:
        n_vis = int((inputs["input_ids"][0] == img_tok).sum().item())
    else:
        n_vis = attn_n_kv
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


def plot_attention_heatmap(
    image: Image.Image,
    attn_map: np.ndarray,
    h_patches: int,
    w_patches: int,
    title: str = "",
    ax=None,
    cmap: str = "hot",
    alpha: float = 0.5,
):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 6))

    attn_2d = np.asarray(attn_map, dtype=np.float32).reshape(h_patches, w_patches)
    img_w, img_h = image.size
    attn_resized = np.array(
        Image.fromarray(attn_2d, mode="F").resize((img_w, img_h), Image.BILINEAR)
    )

    vmax = float(attn_resized.max())
    if vmax <= 0:
        vmax = 1.0

    ax.imshow(image)
    ax.imshow(attn_resized, cmap=cmap, alpha=alpha, vmin=0, vmax=vmax)
    ax.set_title(title, fontsize=12)
    ax.axis("off")


def plot_per_query_attention(
    image: Image.Image,
    attn_vis: np.ndarray,
    h_patches: int,
    w_patches: int,
    n_show: int = 8,
    save_path: str = None,
):
    n_q = min(attn_vis.shape[0], n_show)
    cols = min(4, n_q)
    rows = (n_q + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    axes = np.array(axes).flatten()

    for i in range(n_q):
        plot_attention_heatmap(
            image, attn_vis[i], h_patches, w_patches,
            title=f"Query {i}", ax=axes[i],
        )

    for i in range(n_q, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


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
):
    os.makedirs(save_dir, exist_ok=True)

    messages = build_chat_messages(question, image_path=image_path)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = _process_inputs(backend, processor, text, image_path, config)
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    generated_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    generated_text = _decode_generation(backend, model, processor, inputs, generated_ids, config.n_queries)

    _, attn_vis = model.get_attention_maps()
    if attn_vis is None:
        print("No attention maps available")
        return

    # Always pull off GPU and upcast bf16 → float32 before NumPy.
    attn_vis = attn_vis.detach().float().cpu()

    h_merged, w_merged = _vision_grid(backend, model, inputs, attn_vis.shape[-1])
    n_patches = h_merged * w_merged

    image = Image.open(image_path).convert("RGB")

    attn_agg = attn_vis[0].mean(dim=0).numpy()[:n_patches]
    attn_agg = attn_agg / (attn_agg.sum() + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis("off")

    plot_attention_heatmap(
        image, attn_agg, h_merged, w_merged,
        title=f"Re-Inspection Attention\nQ: {question}\nA: {generated_text}",
        ax=axes[1],
    )

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{sample_name}_attention.pdf")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()

    attn_per_query = attn_vis[0].numpy()[:, :n_patches]
    plot_per_query_attention(
        image, attn_per_query, h_merged, w_merged,
        save_path=os.path.join(save_dir, f"{sample_name}_per_query.pdf"),
    )

    print(f"Question: {question}")
    print(f"Answer: {generated_text}")
    print(f"Attention entropy: {-(attn_agg * np.log(attn_agg + 1e-8)).sum():.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="figures")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--sample_name", type=str, default="sample")
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
    model = load_model_fn(config, device_map="auto")

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
    )


if __name__ == "__main__":
    main()
