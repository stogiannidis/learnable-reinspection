---
name: internvl3 attention visualization
overview: Enable genuine LLM-decoder → image attention for InternVL3 in the motivation pipeline (same recipe used for Qwen3-VL), replacing the misleading hidden-state cosine proxy that currently produces flat, question-independent heatmaps.
todos:
  - id: backend-kwarg
    content: Add attn_implementation kwarg to src/backends/internvl3.py::load_model and forward to from_pretrained.
    status: completed
  - id: evaluate-plumb
    content: Plumb attn_implementation through src/evaluate.py::load_condition_model for InternVL3 (both frozen and reinspection branches); update docstring.
    status: completed
  - id: motivation-gates
    content: In src/motivation.py, flip use_decoder_attn and attn_impl gates to include internvl3.
    status: completed
  - id: motivation-single-tile
    content: Add force_single_tile kwarg to _generate_with_attention(_inner); when set on InternVL3, pass crop_to_patches=False and drop Qwen-only pixel args.
    status: completed
  - id: motivation-grid
    content: Add InternVL3 branch to the decoder-attn extraction block that derives h_merged/w_merged from vision_config * downsample_ratio; remove the sqrt fallback.
    status: completed
  - id: motivation-vis-call
    content: In run_attention_visualization, pass force_single_tile=True for InternVL3.
    status: completed
  - id: standalone-script
    content: In src/visualize_attention.py, extend capture_decoder and attn_implementation=eager to InternVL3.
    status: completed
  - id: smoke-test
    content: Run motivation on a tiny max_samples (e.g. 4) for backend=internvl3 and verify frozen_attn_pair_*.npz has attn_source=decoder and the figure heatmaps differ between Q_A and Q_B.
    status: completed
isProject: false
---

# Fix InternVL3 attention visualization in motivation.py

## Why current figures are broken

`outputs/motivation/attention_figures/frozen_attention_comparison.png` shows near-identical washed-out heatmaps across every question because the "frozen" path for non-Qwen backends uses a hidden-state cosine-similarity proxy, not attention. Real decoder attention is gated to Qwen3-VL in two places:

- [src/motivation.py](src/motivation.py) line 728: `use_decoder_attn = (backend == "qwen3vl")`
- [src/motivation.py](src/motivation.py) line 1227: `attn_impl = "eager" if backend == "qwen3vl" else None`

So InternVL3 never loads with eager attention, `output_attentions=True` returns `None`, and the code silently falls back to the cosine proxy.

## Fix: extend the Qwen3-VL recipe to InternVL3

Decoder self-attention extraction (`_decoder_image_attention` in [src/motivation.py](src/motivation.py)) is already backend-agnostic — it just needs (image_token_id, h_merged, w_merged) and the attentions tuple. Making InternVL3 go through the same path requires four mechanical changes.

### 1. [src/backends/internvl3.py](src/backends/internvl3.py) — accept `attn_implementation`

`load_model` currently hardcodes the SDPA default. Add the kwarg and forward it:

```python
def load_model(config, device_map="auto", processor=None, attn_implementation=None):
    ...
    kw = dict(torch_dtype=dtype, device_map=device_map)
    if attn_implementation is not None:
        kw["attn_implementation"] = attn_implementation
    base_model = InternVLForConditionalGeneration.from_pretrained(resolved, **kw)
```

### 2. [src/evaluate.py](src/evaluate.py) — plumb it through `load_condition_model`

The InternVL3 branch around line 440+ ignores `attn_implementation`. Pass it to both the `frozen` base load and the `reinspection` wrapper load. Update the docstring that says "Qwen3-VL only".

### 3. [src/motivation.py](src/motivation.py) — six focused edits

- **Line 728**: `use_decoder_attn = backend in ("qwen3vl", "internvl3")`.
- **Line 1227**: `attn_impl = "eager" if backend in ("qwen3vl", "internvl3") else None`.
- **`_generate_with_attention` / `_generate_with_attention_inner`**: add a `force_single_tile: bool = False` kwarg. When `True` and `backend == "internvl3"`, override the processor call to pass `crop_to_patches=False` (and drop the Qwen-only `max_pixels`/`min_pixels` that InternVL3's processor ignores). This guarantees a predictable 16×16 single-tile grid matching `vision_config.image_size // patch_size * downsample_ratio`.
- **Decoder-attn extraction block** (currently gated on `"image_grid_thw" in inputs`, lines 300–311): add an `elif backend == "internvl3"` branch that reads `h_merged = w_merged = int(round((vcfg.image_size // vcfg.patch_size) * cfg.downsample_ratio))` from `model.base_model.config` (or `model.config` for frozen). Mirrors the logic already in [src/visualize_attention.py](src/visualize_attention.py)'s `_vision_grid`.
- **`run_attention_visualization`**: pass `force_single_tile=True` in the two `_generate_with_attention` calls when backend is InternVL3. Remove the "approximate square grid from sqrt of vision-token count" fallback (`vision_grid_hw`) — we now have the exact grid.
- **`_decoder_image_attention`**: no change needed; it already works generically from `(image_token_id, h_merged, w_merged)`.

### 4. [src/visualize_attention.py](src/visualize_attention.py) — unblock the standalone script

- Line 352: `capture_decoder = backend in ("qwen3vl", "internvl3")`.
- Line 466 load-kwargs block: also set `attn_implementation="eager"` for `internvl3`.
- The existing `_vision_grid` already computes the InternVL3 grid correctly, and `_process_inputs` already sets `crop_to_patches=False` for InternVL3.

## Reinspection path sanity check

In `InternVL3WithReInspection.generate`, R tokens are inserted with `pad_token_id`, so `(sequences == image_token_id)` still locates the original image-token positions; `prefill_len = attentions[0][0].shape[-1]` matches the augmented prompt length; columns at `img_positions` are still exactly the `h_merged*w_merged` image-token columns. The existing Qwen-tested extraction logic works unchanged.

## Expected outcome

After this change, running

```
python -m src.motivation stage=motivation backend=internvl3
```

will produce `outputs/motivation/attention_figures/frozen_attention_comparison.{pdf,png}` (and the combined figure, if a checkpoint is supplied) with genuine per-token decoder attention that visibly shifts between "Is A above B?" and "Is B above A?". The per-NPZ `attn_source` field will read `"decoder"` for both conditions, so `_plot_combined_attention_comparison` will pick the correct suptitle ("LLM decoder attention over image tokens") automatically.

## Trade-off (confirmed with user)

Global eager attention will slow the InternVL3 motivation run (1143 Flip + Mirror + Divergence pairs) by roughly 1.5–2×. Accepted in exchange for a single consistent code path that matches Qwen3-VL.