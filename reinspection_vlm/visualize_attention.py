"""Attention map visualization for Qwen3-VL + Re-Inspection (unified package)."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

from reinspection_vlm.backends.qwen3vl import load_model
from reinspection_vlm.config import ReInspectionConfig
from reinspection_vlm.data.utils import build_chat_messages


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

    attn_2d = attn_map.reshape(h_patches, w_patches)
    img_w, img_h = image.size
    attn_resized = np.array(
        Image.fromarray(attn_2d).resize((img_w, img_h), Image.BILINEAR)
    )

    ax.imshow(image)
    ax.imshow(attn_resized, cmap=cmap, alpha=alpha, vmin=0, vmax=attn_resized.max())
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
    save_dir: str = "figures",
    sample_name: str = "sample",
):
    os.makedirs(save_dir, exist_ok=True)

    messages = build_chat_messages(question, image_path=image_path)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[image_path],
        return_tensors="pt",
        max_pixels=config.max_pixels,
        min_pixels=config.min_pixels,
    )
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    generated_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    input_len = inputs["input_ids"].shape[1] + config.n_queries
    generated_text = processor.batch_decode(
        generated_ids[:, input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    attn_task, attn_vis = model.get_attention_maps()

    if attn_vis is None:
        print("No attention maps available")
        return

    image_grid_thw = inputs["image_grid_thw"][0]
    _, h, w = image_grid_thw.tolist()
    spatial_merge = 2
    h_merged = h // spatial_merge
    w_merged = w // spatial_merge

    image = Image.open(image_path).convert("RGB")

    attn_agg = attn_vis[0].mean(dim=0).cpu().numpy()
    attn_agg = attn_agg[:h_merged * w_merged]
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

    attn_per_query = attn_vis[0].cpu().numpy()[:, :h_merged * w_merged]
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
    args = parser.parse_args()

    config = ReInspectionConfig()
    if args.config:
        import yaml

        with open(args.config, "r", encoding="utf-8") as f:
            overrides = yaml.safe_load(f)
        for k, v in overrides.items():
            if hasattr(config, k):
                setattr(config, k, v)

    model = load_model(config, device_map="auto")
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

    processor = AutoProcessor.from_pretrained(
        config.model_name_or_path,
        max_pixels=config.max_pixels,
        min_pixels=config.min_pixels,
    )

    visualize_single(
        model, processor,
        image_path=args.image,
        question=args.question,
        config=config,
        save_dir=args.output_dir,
        sample_name=args.sample_name,
    )


if __name__ == "__main__":
    main()
