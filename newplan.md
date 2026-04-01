# Plan: Re-Inspection Module on InternVL3-8B

## Context

The synthetic grid PoC validated the core mechanism: task-conditioned re-inspection tokens produce dramatically better visual grounding than static learnable tokens (100% localization, 20x attention mass on targets, entropy 4.06→2.75). Now we implement this on a real VLM.

**Why InternVL3-8B**: Clean single-point vision→LLM interface (no DeepStack), simple MLP projector, Qwen2.5-7B backbone. No multi-scale injection mechanisms that could confound or wash out the re-inspection signal. V2PE (variable position encoding for visual tokens) is orthogonal — just a position_id scaling, not an architectural change.

**PoC results to build on**: Type B (localization) 100%, Type A (relations) ~54% (near chance — expected limitation of the toy decoder, not the mechanism). Real LLM already understands spatial relations; it just needs better grounding.

---

## InternVL3-8B Architecture (for reference)

```
Image → dynamic tiling (1-12 tiles + thumbnail @ 448×448)
      → InternViT-300M (24 layers, patch=14) → 1024 tokens/tile
      → pixel_shuffle (4x compress) → 256 tokens/tile
      → MLP projector (LN → Linear → GELU → Linear) → (256/tile, 3584)
      → replace <IMG_CONTEXT> placeholders in text sequence
      → Qwen2.5-7B LLM → logits
```

Key dimensions:
- ViT hidden: 1024
- After pixel shuffle: 4096 (= 1024 × 4)
- LLM hidden (= projector output): 3584
- LLM layers: 28, heads: 28, KV heads: 4

---

## Modified Architecture

```
Image → InternViT → pixel_shuffle → MLP → V (N_v × 3584)     [frozen]
Text  → embed_tokens → T (N_t × 3584)                         [frozen]

         ┌──────────────────────────────────────┐
         │        Re-Inspection Module           │
         │  (operates in bottleneck d_r = 256)   │
         │                                       │
         │  V_r = V · W_down_v    (N_v × 256)   │
         │  T_r = T · W_down_t    (N_t × 256)   │
         │                                       │
         │  Q⁰ ∈ ℝ^{N_q × 256}  (learnable)     │
         │       ↓                               │
         │  Stage 1: CrossAttn(Q⁰, T_r) → Q¹    │  task conditioning
         │       ↓                               │
         │  Stage 2: CrossAttn(Q¹, V_r) → R_r   │  visual re-inspection
         │       ↓                               │
         │  R = R_r · W_up       (N_q × 3584)   │
         └──────────────────────────────────────┘

Sequence to LLM:
  [sys_tokens] ... [<img> V </img>] [question_tokens] [R₁...R_Nq] [<assistant>] ...
                                                       ↑ inserted here

Qwen2.5-7B LLM (with LoRA on q_proj, v_proj) → logits → loss on answer tokens
```

### Injection point in code

Inside `InternVLChatModel.forward()`, after vision token replacement (step 7), before LLM call (step 8):

```python
# existing code:
input_embeds[img_context_mask] = vit_embeds.reshape(-1, C)

# >>> OUR INJECTION <<<
# 1. Extract V: input_embeds at positions where img_context_mask is True
# 2. Extract T: input_embeds at text positions (up to end of user message)
# 3. R, A_task, A_vis = self.reinspection_module(V, T)
# 4. Insert R into input_embeds before <|im_start|>assistant position
# 5. Extend attention_mask by N_q positions
# 6. Extend position_ids: R gets standard text position_ids (increment by 1)

outputs = self.language_model(inputs_embeds=input_embeds, ...)
```

### V2PE compatibility

V2PE assigns visual tokens smaller position increments (e.g., delta=1/4 instead of 1). R tokens are NOT visual tokens — they get standard text position_ids (delta=1), incrementing from the last text position before `<|im_start|>assistant`. No conflict.

---

## Implementation Steps

### Step 1: Project scaffold and dependencies

Create `reinspection_internvl3/` with:

```
reinspection_internvl3/
  config.py                  # ReInspectionConfig dataclass
  reinspection_module.py     # Two-stage cross-attention (bottleneck)
  modeling.py                # InternVL3WithReInspection wrapper
  data/
    refcoco.py               # Stage 1: RefCOCO/+/g with bbox→patch mask
    spatial_vqa.py            # Stage 2: VSR + What'sUp + GQA + SpatialBench
    chat_template.py          # Format samples into InternVL3 chat format
  train_stage1.py             # Stage 1 training script
  train_stage2.py             # Stage 2 training script
  evaluate.py                 # Benchmark evaluation
  visualize.py                # Attention map extraction + figures
  configs/
    stage1.yaml
    stage2.yaml
    deepspeed_z2.json
  scripts/
    run_stage1.sh
    run_stage2.sh
    run_eval.sh
  requirements.txt
```

Dependencies:
- `transformers>=4.49` (InternVL3 support)
- `peft>=0.7`
- `accelerate`, `deepspeed`
- `datasets`, `Pillow`
- `matplotlib` (for visualization)

### Step 2: Re-Inspection Module (`reinspection_module.py`)

Implement from the formulation in `reinspection_module_formulation.md` Section 3, with bottleneck:

```python
class ReInspectionModule(nn.Module):
    def __init__(self, d_model=3584, d_bottleneck=256, n_queries=32, n_heads=4, d_ff=1024):
        # Down projections
        self.w_down_v = nn.Linear(d_model, d_bottleneck)
        self.w_down_t = nn.Linear(d_model, d_bottleneck)

        # Base queries
        self.base_queries = nn.Parameter(torch.randn(n_queries, d_bottleneck) * 0.02)

        # Stage 1: task conditioning (cross-attn to text)
        self.cross_attn_task = MultiHeadCrossAttention(d_bottleneck, n_heads)
        self.ln1 = nn.LayerNorm(d_bottleneck)
        self.ffn1 = FFN(d_bottleneck, d_ff)
        self.ln2 = nn.LayerNorm(d_bottleneck)

        # Stage 2: visual re-inspection (cross-attn to vision)
        self.cross_attn_vis = MultiHeadCrossAttention(d_bottleneck, n_heads)
        self.ln3 = nn.LayerNorm(d_bottleneck)
        self.ffn2 = FFN(d_bottleneck, d_ff)
        self.ln4 = nn.LayerNorm(d_bottleneck)

        # Up projection
        self.w_up = nn.Linear(d_bottleneck, d_model)

    def forward(self, V, T):
        V_r = self.w_down_v(V)  # (B, N_v, 256)
        T_r = self.w_down_t(T)  # (B, N_t, 256)

        Q0 = self.base_queries.unsqueeze(0).expand(B, -1, -1)

        # Stage 1
        out, A_task = self.cross_attn_task(Q0, T_r)
        Q1 = self.ln2(self.ln1(Q0 + out) + self.ffn1(self.ln1(Q0 + out)))  # simplified

        # Stage 2
        out, A_vis = self.cross_attn_vis(Q1, V_r)
        R_r = self.ln4(self.ln3(Q1 + out) + self.ffn2(self.ln3(Q1 + out)))

        R = self.w_up(R_r)  # (B, N_q, 3584)
        return R, A_task, A_vis
```

Parameters: ~1.8M (see formulation doc Section 7, bottleneck variant).

### Step 3: Model wrapper (`modeling.py`)

Subclass `InternVLChatModel` → `InternVL3WithReInspection`:

**Key methods to implement:**

1. **`__init__`**: Load parent + create `ReInspectionModule` + store `reinspect_token_id` (new special token)

2. **`forward()`** override:
   - Call `self.extract_feature(pixel_values)` to get `vit_embeds`
   - Get `input_embeds = embed_tokens(input_ids)`
   - Replace img placeholders: `input_embeds[img_mask] = vit_embeds`
   - **Extract V**: gather embeddings at `img_context_token_id` positions
   - **Extract T**: gather embeddings at non-image positions up to end of user turn
   - **Run module**: `R, A_task, A_vis = self.reinspection_module(V, T)`
   - **Find insertion point**: locate `<|im_start|>assistant` token position
   - **Insert R**: splice R tokens into `input_embeds` at that position
   - **Extend masks**: pad `attention_mask` by N_q, extend `position_ids` with sequential text IDs
   - **Shift labels**: if labels provided, insert -100 (ignore) at R positions
   - Call `self.language_model(inputs_embeds=..., attention_mask=..., ...)`
   - Return loss + A_vis (stored for visualization)

3. **`generate()`** override:
   - Same R insertion logic but for inference
   - After inserting R, delegate to parent's `generate()`

4. **`get_attention_maps()`**: Return stored A_vis from last forward pass

**Handling batched sequences of different lengths:**
- Each sample may have different N_v (different number of tiles) and N_t
- V extraction: use `img_context_token_id` mask per sample
- T extraction: use inverse of img mask, truncated to user message end
- R insertion position: find per-sample `<|im_start|>assistant` token
- Pad sequences after insertion to align batch

### Step 4: Data loading

**Stage 1 — RefCOCO (`data/refcoco.py`):**
- Load RefCOCO/RefCOCO+/RefCOCOg from HuggingFace datasets or local
- Each sample: image, referring expression, bounding box [x1, y1, x2, y2]
- Format as chat: `User: <image>\nDescribe the location of: {expression}\nAssistant: {answer}`
- Compute patch-level attention target mask from bbox:
  - Map bbox to the tiled image grid
  - For each tile's 16×16 patch grid (after pixel shuffle), mark patches overlapping bbox
  - Flatten to (N_v,) binary mask, normalize to distribution
- Use `InternVL3Processor` for image preprocessing (dynamic tiling)

**Stage 2 — Spatial VQA (`data/spatial_vqa.py`):**
Unified loader for:
- **VSR**: image + "The {obj1} is to the {relation} of {obj2}" → True/False (~10K)
- **What'sUp** (A+B): orientation understanding → multiple choice
- **GQA spatial subset**: filter GQA for spatial questions (~50K)
- **SpatialBench**: depth/proximity/contact/counting/size

All formatted into InternVL3 chat template:
```
<|im_start|>system\nYou are a helpful assistant.<|im_end|>
<|im_start|>user\n<image>\n{question}<|im_end|>
<|im_start|>assistant\n{answer}<|im_end|>
```

### Step 5: Training

**Stage 1 — Spatial Grounding Warm-up (`train_stage1.py`):**

| Setting | Value |
|---|---|
| Trainable | Re-Inspection Module (all params) + MLP projector (optionally) |
| Frozen | ViT, LLM |
| Loss | L_CE + λ · L_attn (KL divergence, A_vis vs bbox mask) |
| Data | RefCOCO combined (~120K) |
| LR (module) | 1e-4 |
| LR (projector) | 1e-5 (if unfreezing) |
| Scheduler | cosine, 3% warmup |
| Batch size | 32 (gradient accum to effective 128) |
| Epochs | 3-5 |
| λ | 0.5 (tune on val) |
| Precision | bf16 |

Purpose: Bootstrap the module's attention maps to be spatially meaningful before fine-tuning on reasoning tasks. Direct gradients to the module (bypasses frozen LLM).

**Stage 2 — Spatial Reasoning Fine-tuning (`train_stage2.py`):**

| Setting | Value |
|---|---|
| Trainable | Re-Inspection Module + LoRA on LLM (q_proj, v_proj) |
| Frozen | ViT, MLP projector |
| Loss | L_CE only (drop L_attn) |
| Data | VSR + What'sUp + GQA spatial + SpatialBench |
| LR (module) | 5e-5 |
| LR (LoRA) | 2e-5 |
| LoRA rank | 16, alpha=32 |
| Scheduler | cosine, 3% warmup |
| Batch size | 16 (gradient accum to effective 64) |
| Epochs | 5-10 |
| Precision | bf16 |
| DeepSpeed | ZeRO-2 |

Purpose: Train the LLM (via LoRA) to use the re-inspection tokens for spatial reasoning.

### Step 6: Evaluation (`evaluate.py`)

Load best checkpoint, run inference on test splits:

| Benchmark | Metric | What it tests |
|---|---|---|
| VSR | Accuracy | Binary spatial relations |
| What'sUp A | Accuracy | Object orientation |
| What'sUp B | Accuracy | Object orientation (harder) |
| GQA spatial | Accuracy | Spatial VQA |
| SpatialBench | Per-category accuracy | Depth/proximity/size/counting |

Compare three conditions:
1. **InternVL3-8B frozen** (zero-shot baseline)
2. **InternVL3-8B + LoRA only** (fine-tuning without module — controls for LoRA effect)
3. **InternVL3-8B + Re-Inspection + LoRA** (ours)

Also compute attention metrics where bbox annotations available:
- Attention entropy of A_vis
- Attention mass on target patches

### Step 7: Visualization (`visualize.py`)

- Extract A_vis during inference: shape (N_q, N_v)
- Map N_v back to image coordinates (accounting for tiling and pixel shuffle)
- Average across queries → single heatmap per image
- Overlay on original image
- Generate comparison figures: baseline attention (from LLM's own attention to vision tokens) vs re-inspection attention (A_vis)

---

## Parameter Budget

| Component | Parameters | Trainable? |
|---|---|---|
| InternViT-300M | 300M | Frozen |
| MLP projector | ~29M | Stage 1 only |
| Re-Inspection Module | **~1.8M** | Yes (both stages) |
| LoRA (q_proj + v_proj, r=16) | **~6.3M** | Stage 2 only |
| Qwen2.5-7B LLM | 7.6B | Frozen (LoRA adapters only) |
| **Total trainable** | **~8.1M** | **0.1% of model** |

---

## Memory Estimate (InternVL3-8B, bf16)

| Component | Memory |
|---|---|
| Model weights (bf16) | ~16 GB |
| Re-Inspection Module | ~4 MB |
| LoRA adapters | ~13 MB |
| Optimizer states (AdamW, trainable only) | ~50 MB |
| Activations (batch=4, seq=2048) | ~8 GB |
| **Total** | **~25 GB** |

Fits on a single A100-40GB. Use gradient accumulation for larger effective batch sizes, or 2× GPUs with DeepSpeed ZeRO-2 for batch=16+.

---

## Verification Checklist

1. **Smoke test**: Load InternVL3-8B + ReInspectionModule, single forward pass with test image + question. Verify: output shape correct, no NaNs, R tokens at correct position, loss computes.
2. **Sequence check**: Print token sequence for a sample to verify R insertion position is between user message end and assistant start.
3. **Gradient check**: Verify gradients flow to ReInspectionModule params and LoRA params, but NOT to ViT or frozen LLM params.
4. **Stage 1 convergence**: After 1 epoch on RefCOCO, L_attn should decrease. Visualize A_vis — should start concentrating on referred objects.
5. **Stage 2 evaluation**: After fine-tuning, VSR accuracy should exceed both the frozen baseline and the LoRA-only baseline.
6. **Figures**: Generate attention map comparison for 4-6 examples showing the re-inspection module focuses on task-relevant regions.

---

## Critical Files

| File | Purpose |
|---|---|
| `reinspection_module_formulation.md` | Math spec — Sections 3 (module) and 9 (training) |
| InternVL3 `modeling_internvl_chat.py` | Source for subclassing — `forward()`, `extract_feature()`, `generate()` |
| InternVL3 `configuration_internvl_chat.py` | Config structure |
| `Concepts/Differentiable Visual Reinspection.md` | Broader context for future extension |
