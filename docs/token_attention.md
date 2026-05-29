# Per-token text→image attention visualization

Renders, for one `(image, question)` pair, **one heatmap per text token** showing
how that token's decoder self-attention is distributed over the image patch grid,
overlaid on the image. Covers:

- **Every input token** (prefill): system prompt, special/boundary tokens, the
  user query, and the appended assistant-generation suffix.
- **Every generated answer token** (decode).
- **Frozen base VLM vs. Re-Inspection** side by side (input tokens; the two
  conditions generate different answers, so generated-token panels are per
  condition). Re-Inspection `R` tokens get their own panels.

Backends: **InternVL3** and **LLaVA-Next** (LLaMA-family Mistral-7B LM).

## How it works

`model.generate(output_attentions=True, return_dict_in_generate=True)` exposes the
decoder self-attention. The result's `attentions[0]` is the **prefill** matrix
(`[batch, heads, L, L]` per layer) — every input-token row attends to the image
columns; `attentions[t>0]` are the single-row decode steps. We average heads,
reduce over layers (`--layer_reduce`, default mean over all layers; `last`, an
int, or `start:end` also work), index the image-token columns, reshape to the
patch grid, upsample (BICUBIC) and overlay (`alpha=0.5`).

**Materializing weights requires `attn_implementation="eager"`** — the launcher
sets this automatically.

### Patch grids

| Backend | Image tokens visualized | Grid |
|---|---|---|
| InternVL3 | all (single-tile, `crop_to_patches=False`) | 448/14·0.5 = **16×16 = 256** |
| LLaVA-Next | **base global view only** (first 576 image columns) | 336/14 = **24×24 = 576** |

LLaVA-Next's AnyRes packs `[base_global_view, high-res grid + newline features]`;
only the base global view reshapes cleanly to a rectangle and overlays onto the
whole image, so the irregular unpadded high-res grid is intentionally excluded.

### Re-Inspection bookkeeping

The wrapper splices `n_queries` learned `R` tokens at `insert_position` (before the
assistant marker for InternVL3; after it for LLaVA-Next). Image columns sit before
the insert in both, so their indices are unchanged; `R` columns are labelled
separately. The prefill is recovered from one `wrapper.generate(...)` call
(`attentions[0]` has `k_len == prompt_len + n_queries`).

## Running (needs a GPU)

```bash
bash src/scripts/run_token_attention.sh \
  --backend internvl3 \
  --image /data/datasets/vsr/images/000000000142.jpg \
  --question "Is the cat to the left of the laptop?" \
  --conditions frozen,reinspection \
  --checkpoint_dir models/internvl3/stage1/epoch_5 \
  --lora_checkpoint_dir models/internvl3/gqa_sg/stage2/epoch_1 \
  --output_dir outputs/token_attn --sample_name vsr_demo
```

Frozen-only needs no checkpoint:

```bash
bash src/scripts/run_token_attention.sh --backend llava_next --conditions frozen \
  --image <img> --question "<q>" --output_dir outputs/token_attn
```

On the cluster: `kubectl apply -f k8s/token_attention_viz.yaml`.

### Key flags

| Flag | Meaning |
|---|---|
| `--backend` | `internvl3` \| `llava_next` |
| `--conditions` | subset of `frozen,reinspection` |
| `--checkpoint_dir` / `--lora_checkpoint_dir` | Re-Inspection module / Stage-2 LoRA (the module is loaded from the LoRA dir first when both are given) |
| `--layer_reduce` | `mean` (default) \| `last` \| `<int>` \| `start:end` |
| `--max_new_tokens` | answer length to decode (default 32) |
| `--max_input_tokens` | cap rendered input tokens (-1 = all) |
| `--skip_special` | drop special tokens from input rows |
| `--no_r_tokens` | omit `R`-token panels |

## Outputs (`<output_dir>/<backend>_<sample_name>/`)

- `input_tokens.{png,pdf}` — rows = input tokens, columns = Frozen vs Re-Inspection.
- `generated_tokens_<cond>.{png,pdf}` — per generated answer token.
- `r_tokens_reinspection.{png,pdf}` — `R`-token maps (reinspection only).
- `summary.{png,pdf}` — aggregate input-text-mean and generated-mean per condition.
- `maps.npz` — raw grids + labels + answers for reproducible re-plotting.

## Code map

- `src/utils/token_attention.py` — extraction engine (pure, CPU-unit-tested).
- `scripts/analysis/visualize_token_attention.py` — model-touching CLI + rendering.
- `tests/test_token_attention.py` — CPU tests for all extraction math + a render smoke test.
- `src/utils/visualize_attention.py` — the older generated-token-only tool this builds on.
