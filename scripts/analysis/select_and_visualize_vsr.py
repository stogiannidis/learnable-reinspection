"""Select N reinspection-correct + N reinspection-wrong VSR instances, then
render per-token text→image attention (frozen vs reinspection) for each.

Correctness is judged under the *exact* conditions the visualizer uses — this
checkpoint, eager attention, InternVL3 single-tile, the dataset's direct VSR
question, greedy decode — so the "correct/failed" label matches the answer shown
in each figure.

Pipeline (each model loaded once):
  1. load reinspection model → scan shuffled VSR examples (cheap generate, no
     attention) until 5 correct + 5 wrong are found;
  2. capture reinspection per-token attention for the 10 selected;
  3. free it, load frozen → capture frozen per-token attention for the same 10;
  4. render per-instance side-by-side + generated-token figures + a montage.

Run on a GPU (see k8s/token_attention_selection.yaml). Reuses the rendering and
config helpers from visualize_token_attention.py and the extraction engine in
src/utils/token_attention.py.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from src.config import ReInspectionConfig
from src.evaluate import _generate_extra_kw, match_answer
from src.utils.token_attention import (
    KIND_TEXT,
    _process_inputs,
    capture_token_attention,
)
from src.utils.visualize_attention import plot_attention_heatmap


def _load_cli():
    """Import the CLI module (render + config helpers) without running main()."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "visualize_token_attention.py")
    spec = importlib.util.spec_from_file_location("viz_token_attn_cli", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CLI = _load_cli()


@torch.no_grad()
def generate_answer(model, processor, backend, config, image_path, question, is_ri,
                    build_chat, max_new_tokens: int = 8) -> str:
    """Greedy answer (no attention) under the same input pipeline as the viz."""
    device = next(model.parameters()).device
    tokenizer = getattr(processor, "tokenizer", processor)
    messages = build_chat(question, image_path=image_path, system_prompt=config.system_prompt)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _process_inputs(backend, processor, text, image_path, config)
    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                         **_generate_extra_kw(processor))
    seq = gen[0] if is_ri else gen[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(seq, skip_special_tokens=True).strip()


def render_montage(items: List[dict], save_path: str, signal: str = "generated_mean"):
    """2 rows (correct / failed) × N: reinspection attention overlay per instance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = ["correct", "failed"]
    cols = max(sum(1 for it in items if it["group"] == g) for g in groups)
    fig, axes = plt.subplots(2, cols, figsize=(4.2 * cols, 9), squeeze=False)
    for r, g in enumerate(groups):
        row_items = [it for it in items if it["group"] == g]
        for c in range(cols):
            ax = axes[r, c]
            if c >= len(row_items):
                ax.axis("off")
                continue
            it = row_items[c]
            grid = it.get(signal)
            title = f'{it["img"]}\nGT={it["gt"]}  RI={it["ri_pred"]!r}'
            if grid is None:
                ax.imshow(it["image"]); ax.set_title(title, fontsize=8); ax.axis("off")
            else:
                plot_attention_heatmap(it["image"], grid, it["h"], it["w"],
                                       title=title, ax=ax, cmap="inferno", alpha=0.5)
        axes[r, 0].set_ylabel(f"{g.upper()}", fontsize=13, fontweight="bold")
    fig.suptitle("Re-Inspection attention over image patches (mean over answer tokens)\n"
                 "top = reinspection CORRECT, bottom = reinspection WRONG", fontsize=13, y=1.02)
    plt.tight_layout()
    CLI._savefig(fig, save_path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default="internvl3", choices=["internvl3", "llava_next"])
    p.add_argument("--checkpoint_dir", default="models/internvl3/s2_internvl/stage2/epoch_2")
    p.add_argument("--lora_checkpoint_dir", default="models/internvl3/s2_internvl/stage2/epoch_2")
    p.add_argument("--output_dir", default="outputs/token_attn/selection")
    p.add_argument("--vsr_file", default="/data/datasets/vsr/test.jsonl")
    p.add_argument("--image_root", default="/data/datasets/vsr/images")
    p.add_argument("--n_correct", type=int, default=5)
    p.add_argument("--n_wrong", type=int, default=5)
    p.add_argument("--max_scan", type=int, default=120)
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--max_input_tokens", type=int, default=-1)
    p.add_argument("--layer_reduce", default="mean")
    p.add_argument("--conditions", default="frozen,reinspection")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    overrides = dict(backend=args.backend, attn_implementation="eager", bf16=True,
                     checkpoint_dir=args.checkpoint_dir, lora_checkpoint_dir=args.lora_checkpoint_dir)
    config = CLI._config_from_backend_yaml(args.backend, overrides)
    processor = CLI._load_processor(args.backend, config)
    build_chat = CLI._build_chat_fn(args.backend)

    from src.evaluate import load_condition_model

    # Read + shuffle VSR examples (only those whose image exists).
    examples = []
    for line in open(args.vsr_file, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        img = os.path.basename(r.get("image", ""))
        if img and os.path.isfile(os.path.join(args.image_root, img)):
            examples.append({"img": img, "question": r["question"], "gt": str(r["answer"]).strip()})
    random.seed(args.seed)
    random.shuffle(examples)

    # ---- Phase 1: scan with reinspection model to find 5 correct + 5 wrong ----
    print(f"\n=== Phase 1: scanning (reinspection, {args.backend}) ===", flush=True)
    ri_model, is_ri = load_condition_model(
        args.backend, "reinspection", config, processor,
        checkpoint_dir=args.checkpoint_dir, lora_checkpoint_dir=args.lora_checkpoint_dir,
        attn_implementation="eager")
    ri_model.eval()

    correct, wrong = [], []
    for i, ex in enumerate(examples[: args.max_scan]):
        if len(correct) >= args.n_correct and len(wrong) >= args.n_wrong:
            break
        path = os.path.join(args.image_root, ex["img"])
        try:
            pred = generate_answer(ri_model, processor, args.backend, config, path,
                                   ex["question"], is_ri, build_chat, args.max_new_tokens)
        except Exception as e:
            print(f"  scan skip {ex['img']}: {e}", flush=True)
            continue
        ok = match_answer(pred, ex["gt"])
        ex = {**ex, "ri_pred": pred, "ri_correct": ok}
        bucket = correct if ok else wrong
        cap = args.n_correct if ok else args.n_wrong
        if len(bucket) < cap:
            bucket.append(ex)
            print(f"  [{len(correct)}c/{len(wrong)}w] {ex['img']} GT={ex['gt']} "
                  f"RI={pred!r} {'OK' if ok else 'WRONG'}", flush=True)

    selected = [{**e, "group": "correct"} for e in correct] + [{**e, "group": "failed"} for e in wrong]
    if not selected:
        raise SystemExit("No examples selected.")
    print(f"\nselected {len(correct)} correct + {len(wrong)} wrong", flush=True)

    # ---- Phase 2: capture reinspection attention for the selected ----
    def _capture(model, is_ri_flag):
        out = {}
        for idx, ex in enumerate(selected):
            path = os.path.join(args.image_root, ex["img"])
            out[idx] = capture_token_attention(
                model, processor, args.backend, config, image_path=path,
                question=ex["question"], is_reinspection=is_ri_flag, build_chat_messages=build_chat,
                max_new_tokens=args.max_new_tokens, layer_reduce=args.layer_reduce,
                max_input_tokens=args.max_input_tokens)
        return out

    captures: Dict[str, dict] = {}
    if "reinspection" in conditions:
        print("\n=== Phase 2: reinspection attention capture ===", flush=True)
        captures["reinspection"] = _capture(ri_model, True)
    del ri_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Phase 3: frozen ----
    if "frozen" in conditions:
        print("\n=== Phase 3: frozen attention capture ===", flush=True)
        fr_model, _ = load_condition_model(args.backend, "frozen", config, processor,
                                           attn_implementation="eager")
        fr_model.eval()
        captures["frozen"] = _capture(fr_model, False)
        del fr_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Phase 4: render per-instance + montage ----
    print("\n=== Phase 4: rendering ===", flush=True)
    montage_items = []
    for idx, ex in enumerate(selected):
        path = os.path.join(args.image_root, ex["img"])
        image = Image.open(path).convert("RGB")
        stem = f"{ex['group']}_{idx}_{os.path.splitext(ex['img'])[0]}"
        d = os.path.join(args.output_dir, ex["group"], stem)
        os.makedirs(d, exist_ok=True)
        ri_res = captures.get("reinspection", {}).get(idx)
        fr_res = captures.get("frozen", {}).get(idx)
        any_res = ri_res or fr_res
        h, w = any_res["h"], any_res["w"]
        suptitle = (f"{ex['img']} | GT={ex['gt']} | RI={ex['ri_pred']!r} "
                    f"({'correct' if ex['group'] == 'correct' else 'WRONG'})\nQ: {ex['question']}")
        CLI.render_side_by_side(
            image, h, w,
            fr_res["input_maps"] if fr_res else [],
            ri_res["input_maps"] if ri_res else [],
            os.path.join(d, "input_tokens.png"), suptitle=suptitle)
        results_for_summary = {}
        for cond, res in (("frozen", fr_res), ("reinspection", ri_res)):
            if res is None:
                continue
            results_for_summary[cond] = res
            CLI.render_single_condition(
                image, h, w, res["generated_maps"],
                os.path.join(d, f"generated_tokens_{cond}.png"),
                suptitle=f"{cond} generated→image | A: {res['answer']!r}",
                cmap=CLI.FROZEN_CMAP if cond == "frozen" else CLI.RI_CMAP)
        CLI.render_summary(image, results_for_summary, os.path.join(d, "summary.png"))
        montage_items.append({
            "image": image, "h": h, "w": w, "img": ex["img"], "gt": ex["gt"],
            "ri_pred": ex["ri_pred"], "group": ex["group"],
            "generated_mean": (ri_res or fr_res).get("generated_mean"),
        })
        print(f"  rendered {ex['group']}/{stem}", flush=True)

    render_montage(montage_items, os.path.join(args.output_dir, "montage.png"))

    # Selection record.
    rec = [{k: e[k] for k in ("group", "img", "question", "gt", "ri_pred", "ri_correct")} for e in selected]
    with open(os.path.join(args.output_dir, "selection.json"), "w", encoding="utf-8") as f:
        json.dump({"checkpoint": args.checkpoint_dir, "backend": args.backend, "selected": rec}, f, indent=2)
    print(f"\nDone → {args.output_dir}  (selection.json + montage.png + per-instance dirs)", flush=True)


if __name__ == "__main__":
    main()
