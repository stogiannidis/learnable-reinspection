"""Attention-map visualization for InternVL3 re-inspection."""

from __future__ import annotations

import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from peft import PeftModel

from reinspection_vlm.backends.internvl3 import load_model, load_processor
from reinspection_vlm.data.chat_template import build_chat_messages
from reinspection_vlm.train_common import load_config


def _decode_with_prompt_length(model, processor, generated_ids, inputs) -> str:
    prompt_lengths = model.last_generation_prompt_lengths
    prompt_len = int(prompt_lengths[0].item()) if prompt_lengths is not None else int(inputs["input_ids"].shape[1])
    return processor.batch_decode(
        generated_ids[:, prompt_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def _overlay_heatmap(ax, image: Image.Image, attn_grid: np.ndarray, title: str) -> None:
    resized = np.array(Image.fromarray(attn_grid.astype(np.float32)).resize(image.size, Image.BILINEAR))
    ax.imshow(image)
    ax.imshow(resized, cmap="hot", alpha=0.45)
    ax.set_title(title)
    ax.axis("off")


@torch.no_grad()
def visualize_single(model, processor, image_path: str, question: str, config, output_dir: str, sample_name: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    image = Image.open(image_path).convert("RGB")

    messages = build_chat_messages(
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
    inputs = {key: value.to(model.device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}

    generated_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    answer = _decode_with_prompt_length(model, processor, generated_ids, inputs)

    _, attn_vis = model.get_attention_maps()
    if attn_vis is None:
        raise RuntimeError("No attention maps available. Ensure the re-inspection wrapper is loaded.")

    attn = attn_vis[0].mean(dim=0).detach().float().cpu().numpy()
    side = int(math.isqrt(config.image_seq_length))
    if side * side != config.image_seq_length:
        raise ValueError("Expected image_seq_length to be a square number")

    num_tiles = max(1, attn.shape[0] // config.image_seq_length)
    attn = attn[: num_tiles * config.image_seq_length].reshape(num_tiles, config.image_seq_length)
    tile_masses = attn.sum(axis=-1)
    mean_tile_grid = attn.reshape(num_tiles, side, side).mean(axis=0)
    mean_tile_grid /= mean_tile_grid.sum() + 1e-8

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original image")
    axes[0].axis("off")

    overlay_title = "Re-inspection attention"
    if num_tiles > 1:
        overlay_title += " (per-tile average)"
    _overlay_heatmap(axes[1], image, mean_tile_grid, overlay_title)

    axes[2].bar(np.arange(num_tiles), tile_masses)
    axes[2].set_title("Attention mass by tile")
    axes[2].set_xlabel("Tile index")
    axes[2].set_ylabel("Mass")

    fig.suptitle(f"Q: {question}\nA: {answer}", fontsize=12)
    fig.tight_layout()
    save_path = os.path.join(output_dir, f"{sample_name}_attention.pdf")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="figures")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--sample_name", type=str, default="sample")
    args = parser.parse_args()

    config = load_config(args.config, output_dir="outputs")
    processor = load_processor(config)
    model = load_model(config, device_map="auto", processor=processor)

    reinspection_path = os.path.join(args.checkpoint_dir, "reinspection_module.pt")
    if os.path.exists(reinspection_path):
        state_dict = torch.load(reinspection_path, map_location="cpu", weights_only=True)
        model.reinspection.load_state_dict(state_dict)

    lora_path = os.path.join(args.checkpoint_dir, "lora_weights")
    if os.path.exists(lora_path):
        model.base_model.model.language_model = PeftModel.from_pretrained(
            model.base_model.model.language_model,
            lora_path,
        )

    model.eval()
    visualize_single(
        model=model,
        processor=processor,
        image_path=args.image,
        question=args.question,
        config=config,
        output_dir=args.output_dir,
        sample_name=args.sample_name,
    )


if __name__ == "__main__":
    main()
