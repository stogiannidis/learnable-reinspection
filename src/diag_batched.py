"""Dump _find_insert_positions + _extract_vision_and_text internals for idx 3
across three batch constructions, to find why B>1 corrupts the non-longest row.

For whatsup idx 3 we report: where the attention-mask zeros sit (left/right pad),
the insert position, and the extracted vision/text token counts — for
  (1) bs=1 alone           (correct reference)
  (2) processor-batched     (old path, padding_side=left)
  (3) left_pad_collate      (the fix)
Plus the R max-abs-diff vs bs=1 for each, to confirm where R first goes wrong.
"""

from __future__ import annotations

import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src.config import ReInspectionConfig
from src.evaluate import (
    BENCHMARK_CONFIGS, _resolve_data_path, _set_eval_reproducibility, load_condition_model,
)
from src.eval_prompting import build_eval_question
from src.data.chat_template import build_chat_messages as intern_build_chat
from src.data.spatial_dataset import SpatialVQADataset
from src.utils.batch_collate import left_pad_collate

BENCH, IDXS = "whatsup", [0, 3]


def _dev(m):
    return next(m.parameters()).device


@torch.no_grad()
def _extract(model, inputs, device):
    iid = inputs["input_ids"].to(device)
    attn = inputs["attention_mask"].to(device)
    pix = inputs["pixel_values"].to(device, model.base_model.dtype)
    emb = model.base_model.get_input_embeddings()(iid)
    emb, _ = model._encode_vision_and_scatter(iid, emb, pixel_values=pix)
    ipos = model._find_insert_positions(iid, attention_mask=attn)
    V, T, Vm, Tm = model._extract_vision_and_text(emb, iid, ipos, attention_mask=attn)
    R, *_ = model.reinspection(V, T, V_mask=Vm, T_mask=Tm, need_weights=True)
    return dict(attn=attn, ipos=ipos, Vm=Vm, Tm=Tm, R=R)


def _pad_desc(attn_row):
    z = (attn_row == 0).nonzero(as_tuple=False).squeeze(-1)
    if z.numel() == 0:
        return "no-pad"
    first, last = int(z[0]), int(z[-1])
    side = "LEFT" if first == 0 else "RIGHT" if last == attn_row.numel() - 1 else "MID"
    return f"{side}-pad n={z.numel()} (first={first},last={last})"


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    config = ReInspectionConfig(**OmegaConf.to_container(cfg, resolve=True))
    _set_eval_reproducibility(config.seed)
    from src.backends.internvl3 import load_processor
    processor = load_processor(config)
    model, _ = load_condition_model(
        "internvl3", "reinspection", config, processor,
        checkpoint_dir=config.checkpoint_dir, lora_checkpoint_dir=config.lora_checkpoint_dir)
    model.eval()
    device = _dev(model)
    tok = processor.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    print(f"\n{'#'*70}\nDIAG v4: extraction internals for {BENCH} idx {IDXS[1]}\n{'#'*70}")

    rel = BENCHMARK_CONFIGS[BENCH]
    data_file = _resolve_data_path(rel["data_file"], config.data_root)
    image_root = _resolve_data_path(rel["image_root"], config.data_root)
    ds = SpatialVQADataset(
        data_file=data_file, image_root=image_root, processor=processor, backend="internvl3",
        split="test", max_pixels=config.max_pixels, min_pixels=config.min_pixels,
        crop_to_patches=config.crop_to_patches_stage2, system_prompt=config.system_prompt,
        answer_ignore_index=config.answer_ignore_index)
    texts, imgs = [], []
    for i in IDXS:
        s = ds.samples[i]
        q = build_eval_question(s["question"], config)
        ip = os.path.join(image_root, s["image"])
        texts.append(processor.apply_chat_template(
            intern_build_chat(question=q, image_path=ip, system_prompt=config.system_prompt),
            tokenize=False, add_generation_prompt=True))
        imgs.append(ip)

    def proc_one(t, ip):
        return processor(text=[t], images=[ip], return_tensors="pt",
                         max_pixels=config.max_pixels, min_pixels=config.min_pixels,
                         crop_to_patches=config.crop_to_patches_stage2)

    # (1) bs=1 idx3
    s1 = _extract(model, proc_one(texts[1], imgs[1]), device)
    r1 = s1["R"][0]
    # (2) processor-batched (padding_side=left)
    tok.padding_side = "left"
    ob = processor(text=texts, images=imgs, return_tensors="pt", padding=True,
                   max_pixels=config.max_pixels, min_pixels=config.min_pixels,
                   crop_to_patches=config.crop_to_patches_stage2)
    so = _extract(model, ob, device)
    # (3) left_pad_collate
    nb = left_pad_collate([proc_one(texts[0], imgs[0]), proc_one(texts[1], imgs[1])], pad_id)
    sn = _extract(model, nb, device)

    def rowinfo(tag, S, row):
        vc = int(S["Vm"][row].sum()); tc = int(S["Tm"][row].sum())
        ip = int(S["ipos"][row]); pad = _pad_desc(S["attn"][row])
        rdiff = (S["R"][row].float() - r1.float()).abs().max().item()
        print(f"  {tag:16s} L={S['attn'].shape[1]:5d} {pad:28s} insert={ip:5d} vcount={vc:5d} tcount={tc:4d} R_maxdiff_vs_bs1={rdiff:.3e}")

    print(f"  {'bs=1 idx3':16s} L={s1['attn'].shape[1]:5d} {'no-pad':28s} insert={int(s1['ipos'][0]):5d} "
          f"vcount={int(s1['Vm'][0].sum()):5d} tcount={int(s1['Tm'][0].sum()):4d} R_maxdiff_vs_bs1=0")
    rowinfo("proc-batched r1", so, 1)
    rowinfo("collate r1", sn, 1)
    print(f"{'#'*70}")


if __name__ == "__main__":
    main()
