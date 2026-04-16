# Learnable Re-Inspection: Technical Documentation

## Table of Contents

1. [Introduction](#1-introduction)
2. [Motivation and Problem Statement](#2-motivation-and-problem-statement)
3. [Method Overview](#3-method-overview)
4. [ReInspectionModule Architecture](#4-reinspectionmodule-architecture)
5. [VLM Integration](#5-vlm-integration)
6. [Training Pipeline](#6-training-pipeline)
7. [Attention Supervision Losses](#7-attention-supervision-losses)
8. [Data Pipeline](#8-data-pipeline)
9. [Evaluation Framework](#9-evaluation-framework)
10. [Configuration Reference](#10-configuration-reference)
11. [Deployment and Infrastructure](#11-deployment-and-infrastructure)
12. [Design Decisions and Rationale](#12-design-decisions-and-rationale)

---

## 1. Introduction

**Learnable Re-Inspection** is a lightweight, plug-in module for Vision-Language Models (VLMs) that enables task-conditioned visual re-examination. It introduces a two-stage bottleneck cross-attention mechanism that produces a fixed set of **R tokens** -- compact, question-aware visual summaries -- injected into the VLM's token sequence before the assistant's response. The method targets spatial reasoning tasks (e.g., "Is the cup to the left of the bottle?") where a single-pass encoding of the image is insufficient.

The module is backend-agnostic and currently supports two VLM architectures:

| Backend | Base Model | d_model | Source |
|---------|-----------|---------|--------|
| **Qwen3-VL** | `Qwen/Qwen3-VL-8B-Instruct` | 4096 | HuggingFace |
| **InternVL3** | `OpenGVLab/InternVL3-8B-hf` | 3584 | HuggingFace |

---

## 2. Motivation and Problem Statement

### The Bottleneck in Standard VLMs

Modern VLMs process images through a Vision Transformer (ViT) encoder, project the resulting patch embeddings into the LLM's embedding space, and concatenate them with text token embeddings. The LLM then attends over this combined sequence to generate a response. This is fundamentally a **single-pass** process:

```
Image --> [ViT] --> Vision Tokens --+
                                    +--> [LLM] --> Response
Question --> [Tokenizer] --> Text Tokens --+
```

The vision tokens are computed independently of the question. For spatial reasoning, this creates a mismatch: the model must determine object locations and spatial relationships from visual features that were encoded without knowledge of what relationships matter.

### The Re-Inspection Hypothesis

Humans solve spatial reasoning tasks by **looking back** at relevant image regions after understanding the question. Learnable Re-Inspection formalizes this intuition:

1. **Read the question** to understand what spatial relationships to evaluate.
2. **Re-examine the image** with that understanding, focusing attention on relevant regions.
3. **Produce a summary** of the re-examination that the LLM can use during generation.

The key constraint is efficiency: the re-inspection must be lightweight enough to add negligible inference cost while being expressive enough to capture task-relevant spatial information.

---

## 3. Method Overview

### Architecture at a Glance

```
                          +-----------------------------------------+
                          |         ReInspectionModule               |
                          |            (~7M params)                  |
                          |                                         |
  V (vision tokens) ---> W_down_v --+                               |
                         (d->d_r)   |                               |
                                    v                               |
  T (text tokens)  ---> W_down_t -> CrossAttn_1 -> CrossAttn_2 --> W_up --> R tokens
                        (d->d_r)   (task cond.)  (visual re-insp)  (d_r->d)
                                    ^                               |
                    Q^0 (learnable) +                               |
                                                                    |
                          +-----------------------------------------+

  Token Sequence:
  [V tokens] [T tokens] [R tokens] [<|im_start|> assistant] [response...]
```

### Mathematical Formulation

Given vision embeddings $V \in \mathbb{R}^{B \times N_v \times d}$ and text embeddings $T \in \mathbb{R}^{B \times N_t \times d}$:

**Stage 1 -- Task Conditioning:**
$$T_\downarrow = W_{\text{down\_t}} \cdot T \quad \in \mathbb{R}^{B \times N_t \times d_r}$$
$$Q^0 \in \mathbb{R}^{N_q \times d_r} \quad \text{(learnable parameters)}$$
$$Q^1 = Q^0 + \text{CrossAttn}_1(\text{LN}(Q^0),\; T_\downarrow) \quad \text{(residual)}$$
$$Q^1 = Q^1 + \text{FFN}_1(\text{LN}(Q^1)) \quad \text{(feedforward + residual)}$$

**Stage 2 -- Visual Re-Inspection:**
$$V_\downarrow = W_{\text{down\_v}} \cdot V \quad \in \mathbb{R}^{B \times N_v \times d_r}$$
$$R_r = Q^1 + \text{CrossAttn}_2(\text{LN}(Q^1),\; V_\downarrow) \quad \text{(residual)}$$
$$R_r = R_r + \text{FFN}_2(\text{LN}(R_r)) \quad \text{(feedforward + residual)}$$

**Output Projection:**
$$R = W_{\text{up}} \cdot R_r \quad \in \mathbb{R}^{B \times N_q \times d}$$

where $d_r = 512$ is the bottleneck dimension, $N_q = 64$ is the number of learnable queries, and $d \in \{3584, 4096\}$ is the model embedding dimension.

### Cross-Attention Details

Each `BottleneckCrossAttention` module implements standard multi-head cross-attention entirely within $d_r$:

$$\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}\right) V$$

with $n_h = 8$ heads and $d_h = d_r / n_h = 64$ per head. Projections:
- $W_Q, W_K, W_V, W_O \in \mathbb{R}^{d_r \times d_r}$ (with bias)
- Uses `torch.nn.functional.scaled_dot_product_attention` when attention weights are not needed, falling back to explicit softmax when `need_weights=True`
- Optional key-value masking via additive bias: $(1 - \text{mask}) \cdot (-\infty)$

### FFN Blocks

Each FFN block follows the standard pre-norm Transformer pattern:

$$\text{FFN}(x) = W_2 \cdot \text{Dropout}(\text{GELU}(\text{Dropout}(W_1 \cdot x)))$$

with expansion factor $m = 4$: $W_1 \in \mathbb{R}^{d_r \times m \cdot d_r}$, $W_2 \in \mathbb{R}^{m \cdot d_r \times d_r}$.

---

## 4. ReInspectionModule Architecture

**Source:** `src/reinspection_module.py`

### Component Inventory

| Component | Shape | Parameters |
|-----------|-------|------------|
| `base_queries` | $(N_q, d_r)$ | 32,768 |
| `W_down_t` | $(d, d_r)$, no bias | 2,097,152 (Qwen) / 1,835,008 (InternVL) |
| `W_down_v` | $(d, d_r)$, no bias | 2,097,152 (Qwen) / 1,835,008 (InternVL) |
| `W_up` | $(d_r, d)$, no bias | 2,097,152 (Qwen) / 1,835,008 (InternVL) |
| `norm_q0` | $(d_r,)$ | 1,024 |
| `cross_text` | 4 linear layers in $d_r$ | 1,049,088 |
| `norm1` | $(d_r,)$ | 1,024 |
| `ffn1` | $d_r \to 4 d_r \to d_r$ | 2,099,200 |
| `norm_q1` | $(d_r,)$ | 1,024 |
| `cross_vis` | 4 linear layers in $d_r$ | 1,049,088 |
| `norm2` | $(d_r,)$ | 1,024 |
| `ffn2` | $d_r \to 4 d_r \to d_r$ | 2,099,200 |

**Total (Qwen):** ~12.6M parameters
**Total (InternVL):** ~11.8M parameters

### Weight Initialization

- **Linear layers:** Xavier uniform initialization
- **LayerNorm:** weight = 1, bias = 0
- **W_up:** initialized to **all zeros** -- this is critical. At initialization, $R = \mathbf{0}$, so the model behaves identically to the original VLM. Training smoothly transitions from the base model's behavior.
- **base_queries:** $\mathcal{N}(0, \sqrt{2/(d + d_r)})$ (Kaiming-like scaling)

### Forward Signature

```python
def forward(
    self,
    V: torch.Tensor,           # (B, N_v, d_model) -- vision embeddings
    T: torch.Tensor,           # (B, N_t, d_model) -- text embeddings
    V_mask: Optional[BoolTensor],  # (B, N_v) -- True for valid vision tokens
    T_mask: Optional[BoolTensor],  # (B, N_t) -- True for valid text tokens
    need_weights: bool = False,     # return attention maps for supervision/viz
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    # Returns:
    #   R:      (B, N_q, d_model) -- re-inspection tokens
    #   A_task: (B, N_q, N_t) or None -- task attention weights (head-averaged)
    #   A_vis:  (B, N_q, N_v) or None -- visual attention weights (head-averaged)
```

---

## 5. VLM Integration

### 5.1 Qwen3-VL Integration

**Source:** `src/backends/qwen3vl.py`

**Class:** `Qwen3VLWithReInspection(nn.Module)` -- wraps (does not subclass) `Qwen3VLForConditionalGeneration`.

#### Training Forward Flow

```
1. Embed input_ids via get_input_embeddings()
2. Encode pixel_values via get_image_features() -> deepstack features
3. masked_scatter() vision embeddings into input token positions
4. Compute 3D MRoPE position_ids from ORIGINAL sequence (before R insertion)
5. Find last <|im_start|> token position (before assistant turn)
6. Extract V = all vision token embeddings, T = non-vision tokens before insert point
7. Run ReInspectionModule(V, T) -> R, A_task, A_vis
8. Insert R tokens at insert position:
   - inputs_embeds: [pre...] [R_0..R_{N_q-1}] [post...]
   - attention_mask: R positions = 1
   - labels: R positions = -100 (ignored)
   - position_ids: R tokens get incrementing positions in ALL 3 MRoPE dims
9. Forward through language_model with DeepStack visual_pos_masks
10. Compute lm_head(hidden_states) -> logits -> CE loss
```

#### MRoPE Position Encoding for R Tokens

Qwen3-VL uses Multiplicative Rotary Position Embeddings with three independent dimensions: temporal (t), height (h), width (w). For text tokens, all three dimensions increment uniformly. Vision tokens use spatial (h, w) positions matching their patch grid location.

R tokens are assigned uniformly incrementing positions across all three MRoPE dims, starting from the last text position before the insertion point and shifting all subsequent positions by $N_q$. This matches the text token encoding pattern, which the LLM has seen during pretraining.

#### Generation Strategy

Generation uses a **forward pre-hook** approach to preserve the base model's internal pipelines:

```
1. Insert N_q placeholder tokens (pad_id) into input_ids at the insertion point
2. Let base model handle vision encoding, MRoPE computation, and DeepStack normally
3. Register a forward pre-hook on language_model that:
   a. Extracts V and T from the base model's inputs_embeds (using visual_pos_masks)
   b. Runs ReInspectionModule(V, T) -> R
   c. Replaces placeholder embeddings with R embeddings in-place
4. Hook fires only during prefill (not during autoregressive token generation)
5. Remove hook after generation completes
```

This preserves correct MRoPE spatial positions, DeepStack visual feature injection, and proper KV cache state.

#### DeepStack Compatibility

DeepStack injects per-layer vision features into specific LLM layers. The `visual_pos_masks` tensor marks which positions in the sequence are vision tokens. After R token insertion, the mask is expanded: R positions are marked `False` (they are not vision tokens), ensuring DeepStack does not attempt to inject vision features at R positions.

### 5.2 InternVL3 Integration

**Source:** `src/backends/internvl3.py`

**Class:** `InternVL3WithReInspection(nn.Module)` -- wraps `InternVLForConditionalGeneration`.

#### Key Differences from Qwen

| Aspect | Qwen3-VL | InternVL3 |
|--------|----------|-----------|
| Position encoding | 3D MRoPE (t, h, w) | Standard 1D position IDs |
| Insert point detection | Last `<\|im_start\|>` token | First supervised label position (training) or generation suffix match (inference) |
| Vision features | DeepStack (per-layer) | Single-pass ViT projection |
| Video support | Yes | No |
| R token gradient flow | Standard autograd | Detached buffer (prevents bf16 overflow) |
| Loss computation | Base model's `loss_function()` | Manual shift + `F.cross_entropy` in fp32 |

#### Gradient Detachment

InternVL3's R token insertion uses a fresh `torch.zeros` buffer with `requires_grad=False`. This detaches R from the frozen backbone's backward path, preventing CE loss gradients from flowing back through frozen bf16 layers and causing numerical overflow (inf gradients). The reinspection module is trained via attention supervision in Stage 1; in Stage 2, only LoRA parameters receive CE gradients.

#### Generation Suffix Detection

At initialization, the wrapper probes the tokenizer to detect the generation prompt suffix (the token sequence added by `add_generation_prompt=True` that is absent in `add_generation_prompt=False`). During inference, R tokens are inserted just before this suffix.

```python
# Example: if the generation suffix is [<|im_start|>, assistant, \n]
# Then R tokens are inserted right before these tokens in the input sequence
```

---

## 6. Training Pipeline

**Source:** `src/train_common.py`

### 6.1 Two-Stage Training Strategy

#### Stage 1: Grounding Warm-up

**Goal:** Teach the ReInspectionModule where to attend in the image given a referring expression.

- **Dataset:** RefCOCO + RefCOCO+ + RefCOCOg (train splits)
- **Task:** Given an expression (e.g., "the dog on the left"), predict its bounding box coordinates
- **Trainable parameters:** ReInspectionModule only (~7-13M params)
- **Frozen:** Entire VLM backbone (ViT + LLM)
- **Loss:** $\mathcal{L} = \mathcal{L}_{CE} + \lambda \cdot \mathcal{L}_{attn}$
- **Optional (InternVL):** Unfreeze the multi-modal projector with a separate learning rate

#### Stage 2: Spatial Reasoning Fine-tuning

**Goal:** Teach the LLM to use R tokens for spatial reasoning tasks.

- **Dataset:** Concatenation of VSR, What'sUp, GQA-Spatial, SpatialBench (train splits)
- **Task:** Spatial VQA -- answer questions about spatial relationships between objects
- **Trainable parameters:** ReInspectionModule + LoRA adapters on LLM
- **Frozen:** ViT encoder and non-LoRA LLM parameters
- **Loss:** $\mathcal{L} = \mathcal{L}_{CE}$ (no attention supervision)
- **Stage 1 checkpoint:** Loaded into ReInspectionModule before Stage 2 begins

### 6.2 Training Loop Details

#### Distributed Setup

```python
dist.init_process_group(backend="nccl")  # if RANK in environment
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
```

Supports both DDP (`torch.nn.parallel.DistributedDataParallel`) and DeepSpeed ZeRO. When a DeepSpeed config is provided, `deepspeed.initialize()` replaces DDP wrapping.

#### Optimizer

AdamW with weight decay 0.01 and separate learning rate groups:

**Stage 1:**
```python
optimizer = AdamW([
    {"params": reinspection.parameters(), "lr": 1e-4},
    # InternVL only, if stage1_train_projector=True:
    {"params": projector.parameters(), "lr": 1e-5},
], weight_decay=0.01)
```

**Stage 2:**
```python
optimizer = AdamW([
    {"params": reinspection.parameters(), "lr": 5e-5},
    {"params": lora_parameters(),         "lr": 2e-5},
], weight_decay=0.01)
```

#### Learning Rate Schedule

Cosine schedule with linear warmup:

$$\text{num\_updates} = \left\lfloor \frac{|\text{DataLoader}| \times \text{n\_epochs}}{\text{grad\_accum}} \right\rfloor$$
$$\text{num\_warmup} = \lfloor \text{warmup\_ratio} \times \text{num\_updates} \rfloor$$

The schedule advances once per optimizer step (not per micro-batch).

#### Gradient Safety

The training loop implements two levels of non-finite detection:

1. **Loss-level:** Before backward, check `torch.isfinite(loss)`. All-reduce across DDP ranks. If any rank has non-finite loss, skip the batch.

2. **Gradient-level:** Before `clip_grad_norm_`, check all parameter gradients for finiteness. All-reduce across DDP ranks. If any rank has non-finite gradients, zero gradients and skip the optimizer step. This prevents the IEEE 754 artifact where `clip_grad_norm_(inf) = NaN` (because `inf * 0 = NaN`).

```python
# Gradient guard (prevents inf * 0 = NaN in clip_grad_norm_)
_grad_finite = torch.tensor(1.0, device=device)
for p in trainable_params:
    if p.grad is not None and not torch.isfinite(p.grad).all():
        _grad_finite.fill_(0.0)
        break
if dist.is_initialized():
    dist.all_reduce(_grad_finite, op=dist.ReduceOp.MIN)
if _grad_finite.item() < 0.5:
    optimizer.zero_grad(set_to_none=True)  # Skip step
else:
    torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm=1.0)
    optimizer.step()
```

#### LoRA Configuration (Stage 2)

Applied to the language model's attention layers using PEFT:

```python
LoraConfig(
    r=16,            # Qwen (32 for InternVL)
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="FEATURE_EXTRACTION",
)
```

LoRA is applied to `model.base_model.model.language_model` via `get_peft_model()`.

#### Checkpointing

After each epoch, the training loop saves:

```
models/<backend>/stage<N>/epoch_<E>/
    reinspection_module.pt        # ReInspectionModule state_dict
    lora_weights/                 # (Stage 2 only) PEFT adapter
        adapter_config.json
        adapter_model.bin
```

Default Hydra `output_dir` is `models`; the full path is `{output_dir}/{backend}/stage<N>/epoch_<E>/`. Override `output_dir` for a separate experiment root.

With DeepSpeed ZeRO, parameters are gathered to rank 0 using `deepspeed.zero.GatheredParameters()` before saving.

#### Logging

Metrics are logged to Weights & Biases (if available) at every step:

- `{backend}_stage{N}/train/loss` -- total loss
- `{backend}_stage{N}/train/ce_loss` -- cross-entropy loss
- `{backend}_stage{N}/train/attn_loss` -- attention supervision loss (Stage 1 only)
- `{backend}_stage{N}/train/lr` -- current learning rate
- `{backend}_stage{N}/epoch/loss` -- epoch-averaged loss

Console logging every 50 steps.

---

## 7. Attention Supervision Losses

**Source:** `src/attn_loss.py`

Stage 1 training uses attention supervision to guide the visual cross-attention heads toward bounding box regions. Two loss functions are supported:

### 7.1 Focal Loss (Default for Both Backends)

**Function:** `compute_attn_loss_focal(A_vis, attn_target, image_grid_thw, n_queries, alpha, gamma)`

Binary focal loss with per-query spatial diversification:

$$\mathcal{L}_{focal} = -\frac{1}{N} \sum_{i} \alpha_t^{(i)} \cdot (1 - p_t^{(i)})^\gamma \cdot \text{BCE}(p^{(i)}, y^{(i)})$$

where:
- $p \in [0, 1]$ is the predicted attention weight (rescaled: $p = A_{vis} \cdot N_v$, clamped to $[10^{-6}, 1-10^{-6}]$)
- $y \in \{0, 1\}$ is the binarized bbox target
- $\alpha = 0.25$ (class balancing factor)
- $\gamma = 2.0$ (focusing parameter)
- $p_t = y \cdot p + (1-y) \cdot (1-p)$
- $\alpha_t = y \cdot \alpha + (1-y) \cdot (1-\alpha)$

#### Per-Query Target Diversification

When `image_grid_thw` is provided, the bbox target is partitioned into vertical strips, with each of the $N_q$ queries assigned a different strip:

```
Original bbox target (all queries see the same region):
    +-----------+
    |  bbox     |
    |  region   |
    +-----------+

Diversified targets (each query gets a vertical strip):
    +---+---+---+---+
    | Q0| Q1| Q2|Q3 |  ...  | Q_{N_q-1}
    +---+---+---+---+
```

The partition is computed over the **rows** of positive patches:
1. Find the row range $[r_{min}, r_{max})$ of positive patches
2. Divide into $N_q$ equal strips
3. Each query $q$ gets target patches only within strip $q$

This encourages different queries to attend to different spatial sub-regions of the bounding box, promoting representational diversity.

### 7.2 KL Divergence Loss (Alternative)

**Function:** `compute_attn_loss_kl(attn_vis, attn_target)`

$$\mathcal{L}_{KL} = \frac{1}{N_q} \cdot D_{KL}(\text{target} \;\|\; \text{pred})$$

where:
- Both `pred` and `target` are normalized to sum to 1 (clamped $\geq 10^{-8}$)
- The target is expanded from $(B, N_v)$ to $(B, N_q, N_v)$ -- same target for all queries
- Uses `F.kl_div` with `reduction="batchmean"`, divided by $N_q$

### 7.3 Target Shape Alignment

The attention loss handler (`_stage1_attn_loss`) aligns target dimensions with predicted attention:

```python
n_v = outputs.attn_vis.shape[-1]
if target.shape[-1] != n_v:
    target = F.pad(target[:, :n_v], (0, max(0, n_v - target.shape[-1])))
```

---

## 8. Data Pipeline

### 8.1 RefCOCO Dataset (Stage 1)

**Source:** `src/data/refcoco.py`

Loads annotations from RefCOCO, RefCOCO+, and RefCOCOg with unified preprocessing.

#### Input Format

```json
{
    "expression": "the dog on the left",
    "bbox": [x, y, w, h],        // COCO format (pixels)
    "image_w": 640,
    "image_h": 480,
    "image": "COCO_train2014_000000012345.jpg"
}
```

#### Processing

1. **Normalize bbox:** COCO format $\to$ $[x_1/w, y_1/h, x_2/w, y_2/h]$ (normalized coordinates)
2. **Build answer string:** `"[0.123, 0.456, 0.789, 0.012]"`
3. **Build chat messages:** Backend-specific chat template
4. **Tokenize:** Process prompt-only and full (prompt+answer) separately to compute label mask
5. **Labels:** First $N_{prompt}$ tokens set to -100 (ignored); only answer tokens are supervised
6. **Attention target:** Bbox mapped to patch-level supervision mask

#### Bbox-to-Patch Mapping (Qwen)

**Function:** `bbox_to_patch_mask(bbox, image_grid_thw, spatial_merge_size=2)`

Maps a normalized bounding box to Qwen3-VL's merged patch grid:

```
1. Compute merged grid dimensions:
   h_merged = h_patches / spatial_merge_size  (spatial_merge_size = 2)
   w_merged = w_patches / spatial_merge_size

2. Map bbox to grid coordinates:
   col_range = [floor(x1 * w_merged), ceil(x2 * w_merged)]
   row_range = [floor(y1 * h_merged), ceil(y2 * h_merged)]

3. Create binary mask over n_patches = t * h_merged * w_merged tokens

4. Normalize: mask = mask / sum(mask)
```

#### Bbox-to-Patch Mapping (InternVL)

**Function:** `intern_bbox_to_patch_mask(bbox, num_image_patches, image_seq_length=256)`

Maps to InternVL's fixed 16x16 patch grid:

```
1. Grid side = sqrt(image_seq_length) = 16

2. Map bbox to grid:
   col_range = [floor(x1 * 16), ceil(x2 * 16)]
   row_range = [floor(y1 * 16), ceil(y2 * 16)]

3. Create mask over 256 patches (repeated for multi-patch images)

4. Normalize: mask = mask / sum(mask)
```

### 8.2 Spatial VQA Dataset (Stage 2)

**Source:** `src/data/spatial_dataset.py`

#### Benchmarks

| Benchmark | Format | Description |
|-----------|--------|-------------|
| **VSR** | JSONL | Visual Spatial Reasoning: binary spatial relation judgments |
| **What'sUp** | JSON | Orientation and spatial arrangement questions |
| **GQA-Spatial** | JSON | GQA subset focused on spatial relationships |
| **SpatialBench** | JSON | Comprehensive spatial reasoning benchmark |

#### Input Format

```json
{
    "question": "Is the cup to the left of the plate?",
    "answer": "yes",
    "image": "image_00123.jpg",
    "split": "train"
}
```

#### Processing

1. Build chat messages (backend-specific)
2. Tokenize prompt-only and full (prompt+answer)
3. Labels: first $N_{prompt}$ tokens = -100
4. Error handling: retry with random replacement on corrupt images (up to 10 retries for InternVL, 3 for Qwen)

#### Dataset Composition

`build_spatial_dataset()` creates a `ConcatDataset` from all available benchmarks:

```python
datasets = ["vsr", "whatsup", "gqa_spatial", "spatialbench"]
# Each with its own data_file and image_root under data_root
# Missing datasets are silently skipped
```

### 8.3 Chat Templates

#### Qwen3-VL (`data/utils.py`)

```python
messages = [
    {"role": "user", "content": [
        {"type": "image", "image": "/path/to/image.jpg"},
        {"type": "text", "text": "Where is the cat?"}
    ]},
    {"role": "assistant", "content": "The cat is on the left side."}  # training only
]
```

#### InternVL3 (`data/chat_template.py`)

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
        {"type": "image", "image": "/path/to/image.jpg"},
        {"type": "text", "text": "Where is the cat?"}
    ]},
    {"role": "assistant", "content": "The cat is on the left side."}  # training only
]
```

### 8.4 Collation

**Function:** `collate_fn(batch)` in `train_common.py`

Handles heterogeneous tensor shapes across a batch:

| Key Type | Strategy |
|----------|----------|
| `pixel_values`, `image_grid_thw`, `video_grid_thw` | `torch.cat` along dim 0 (concatenate across batch) |
| `attn_target_mask`, `bbox_norm` | `pad_sequence` with padding value 0.0 |
| `labels` | `pad_sequence` with padding value -100 |
| Scalar tensors | `torch.stack` |
| Uniform-shape tensors | `torch.stack` |
| Variable-length tensors | `pad_sequence` with padding value 0 |
| Non-tensors | Kept as list |

---

## 9. Evaluation Framework

**Source:** `src/evaluate.py`

### 9.1 Evaluation Conditions

The evaluation framework supports three conditions for controlled comparison:

| Condition | Model | Trainable Weights |
|-----------|-------|------------------|
| **frozen** | Base VLM (no modifications) | None |
| **lora_only** | Base VLM + LoRA (no ReInspection) | LoRA adapters |
| **reinspection** | Base VLM + ReInspection + LoRA | ReInspection + LoRA |

The `--compare` flag runs all three conditions sequentially.

### 9.2 Metrics

#### Accuracy

Normalized exact match or substring containment:

```python
def normalize_answer(text):
    return " ".join(text.strip().lower().split())

gt_norm = normalize_answer(gt_answer)
gen_norm = normalize_answer(generated_text)
is_correct = (gen_norm == gt_norm) or (gt_norm in gen_norm)
```

#### Attention Entropy

For re-inspection models, the visual attention entropy measures how focused the attention distribution is:

$$H = -\frac{1}{N_q} \sum_{q=1}^{N_q} \sum_{v=1}^{N_v} \bar{a}_{q,v} \log \bar{a}_{q,v}$$

where $\bar{a}$ is the head-averaged attention distribution, renormalized to sum to 1. Lower entropy indicates more focused spatial attention.

### 9.3 Generation Configuration

```python
model.generate(
    **inputs,
    max_new_tokens=64,
    do_sample=False,  # greedy decoding
)
```

#### Output Decoding

After generation, prompt tokens must be stripped. The prompt length differs between conditions:

- **Frozen/LoRA-only:** `input_len = inputs["input_ids"].shape[1]`
- **Reinspection (Qwen):** `input_len = inputs["input_ids"].shape[1] + n_queries` (R tokens added to sequence)
- **Reinspection (InternVL):** `input_len = model.last_generation_prompt_lengths[0]` (tracked internally)

### 9.4 Output Format

Results are saved as JSON with per-sample details:

```json
[
    {
        "condition": "reinspection",
        "benchmark": "vsr",
        "accuracy": 0.72,
        "correct": 360,
        "total": 500,
        "skipped": 0,
        "mean_attention_entropy": 4.52
    }
]
```

Per-sample outputs are saved to separate files: `{condition}_{benchmark}_samples.json`.

A formatted comparison table is printed and optionally saved:

```
===========================================================
Comparison Table
===========================================================
Benchmark      |  frozen  | lora_only | reinspection
---------------+---------+-----------+-------------
vsr            |  62.3%  |   68.1%   |    72.4%
whatsup        |  55.0%  |   61.2%   |    67.8%
gqa_spatial    |  70.1%  |   73.5%   |    76.2%
spatialbench   |  48.9%  |   54.3%   |    59.1%
---------------+---------+-----------+-------------
Mean           |  59.1%  |   64.3%   |    68.9%
===========================================================
```

---

## 10. Configuration Reference

**Source:** `src/config.py`

### ReInspectionConfig Dataclass

All fields with their defaults and per-backend overrides:

#### Module Dimensions

| Field | Default | Qwen3-VL | InternVL3 | Description |
|-------|---------|----------|-----------|-------------|
| `d_model` | 4096 | 4096 | 3584 | LLM embedding dimension |
| `d_bottleneck` | 512 | 512 | 512 | Bottleneck dimension $d_r$ |
| `n_queries` | 64 | 64 | 64 | Number of learnable query tokens $N_q$ |
| `n_heads` | 8 | 8 | 8 | Attention heads in cross-attention |
| `ffn_mult` | 4 | 4 | 4 | FFN expansion factor |
| `dropout` | 0.0 | 0.0 | 0.0 | Dropout rate |

#### Model

| Field | Default | Description |
|-------|---------|-------------|
| `model_name_or_path` | `"Qwen/Qwen3-VL-8B-Instruct"` | HuggingFace model ID |
| `processor_name_or_path` | `None` | Processor ID (defaults to model path) |
| `image_seq_length` | 256 | InternVL patch grid size (16x16) |
| `answer_ignore_index` | -100 | Label ignore value |

#### LoRA (Stage 2)

| Field | Default | Qwen3-VL | InternVL3 |
|-------|---------|----------|-----------|
| `lora_r` | 16 | 16 | 32 |
| `lora_alpha` | 32 | 32 | 32 |
| `lora_dropout` | 0.05 | 0.05 | 0.05 |
| `lora_target_modules` | `["q_proj", "v_proj"]` | same | same |

#### Stage 1 Training

| Field | Default | Qwen3-VL | InternVL3 |
|-------|---------|----------|-----------|
| `stage1_lr_module` | 1e-4 | 1e-4 | 1e-4 |
| `stage1_lr_projector` | 1e-5 | -- | 1e-5 |
| `stage1_epochs` | 5 | 5 | 5 |
| `stage1_batch_size` | 4 | 8 | 4 |
| `stage1_grad_accum` | 8 | 8 | 16 |
| `stage1_warmup_ratio` | 0.03 | 0.03 | 0.03 |
| `stage1_attn_loss_weight` | 10.0 | 10.0 | 0.5 |
| `stage1_attn_loss_type` | `"focal"` | `"focal"` | `"focal"` |
| `stage1_train_projector` | `False` | `False` | `False` |

**Effective batch size (Stage 1):**
- Qwen: $8 \times 8 \times 2\text{ GPUs} = 128$
- InternVL: $4 \times 16 \times 2\text{ GPUs} = 128$

#### Stage 2 Training

| Field | Default | Qwen3-VL | InternVL3 |
|-------|---------|----------|-----------|
| `stage2_lr_module` | 5e-5 | 5e-5 | 5e-5 |
| `stage2_lr_lora` | 2e-5 | 2e-5 | 2e-5 |
| `stage2_epochs` | 10 | 10 | 4 |
| `stage2_batch_size` | 4 | 8 | 8 |
| `stage2_grad_accum` | 8 | 8 | 8 |
| `stage2_warmup_ratio` | 0.03 | 0.03 | 0.05 |

**Effective batch size (Stage 2):**
- Qwen: $8 \times 8 \times 2\text{ GPUs} = 128$
- InternVL: $8 \times 8 \times 2\text{ GPUs} = 128$

#### Image Processing

| Field | Default | Description |
|-------|---------|-------------|
| `max_pixels` | 1,003,520 | $1280 \times 28 \times 28$ |
| `min_pixels` | 3,136 | $4 \times 28 \times 28$ |
| `crop_to_patches_stage1` | `False` | InternVL dynamic patch cropping |
| `crop_to_patches_stage2` | `True` | InternVL dynamic patch cropping |

#### General

| Field | Default |
|-------|---------|
| `seed` | 42 |
| `bf16` | `True` |
| `gradient_checkpointing` | `True` |
| `max_grad_norm` | 1.0 |

### YAML Configuration Files

**Hydra:** `src/configs/config.yaml` composes `backend/*.yaml` (e.g. `internvl3`, `qwen3vl`) and `stage/*.yaml` (e.g. `stage1`, `stage2`, `eval`). Override with `key=value` on the CLI.

**DeepSpeed JSON** (paths referenced from backend YAML):

| Location | Purpose |
|----------|---------|
| `configs/qwen3vl/deepspeed_z*.json` | ZeRO configs for Qwen runs |
| `configs/internvl3/deepspeed_z*.json` | ZeRO configs for InternVL runs |

**Legacy InternVL:** select Hydra `backend=internvl3_legacy` (composes `backend/internvl3.yaml` then overrides module geometry and stage1/stage2 batch recipe for older checkpoints).

**Reference YAML** (same recipe as `internvl3_legacy`, for docs or manual merge; not loaded by Hydra defaults):

| File | Purpose |
|------|---------|
| `config_archive/internvl3/stage1_legacy.yaml` | Stage 1 legacy hyperparameters |
| `config_archive/internvl3/stage2_legacy.yaml` | Stage 2 legacy hyperparameters |

---

## 11. Deployment and Infrastructure

### 11.1 Docker

**File:** `Dockerfile`

```
Base:    nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04
Python:  3.12
CUDA:    13.0.2 with cuDNN
Arch:    sm_80 (A100), sm_90 (H100)
```

Key environment variables set in the image:

| Variable | Purpose |
|----------|---------|
| `TORCH_CUDA_ARCH_LIST` | `"8.0;9.0"` (A100 + H100) |
| `HF_HOME` | `/data/Huggingface` (shared model cache) |
| `HF_HUB_ENABLE_HF_TRANSFER` | `1` (fast downloads) |
| `NCCL_P2P_DISABLE` | `0` (enable peer-to-peer) |
| `NCCL_IB_DISABLE` | `1` (disable InfiniBand) |
| `TOKENIZERS_PARALLELISM` | `false` (avoid fork deadlocks) |

### 11.2 DeepSpeed Configuration

**Files:** `src/configs/{qwen3vl,internvl3}/deepspeed_z3.json`

Both backends use identical ZeRO-3 configs:

```json
{
    "bf16": {"enabled": true},
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "none"},
        "offload_param": {"device": "none"},
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": 2e8,
        "stage3_prefetch_bucket_size": 2e8,
        "stage3_param_persistence_threshold": 1e6
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": 1.0,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto"
}
```

Key settings:
- **ZeRO Stage 3:** Full parameter, gradient, and optimizer state partitioning
- **No offloading:** All states remain on GPU (sufficient for 2x H100 80GB)
- **Communication overlap:** Enabled for reduced latency
- **Gradient clipping:** 1.0 (matches `max_grad_norm`)
- **Batch sizes:** Auto-configured from training arguments

The scripts default to `deepspeed_z2.json` (referenced but ZeRO-3 configs are also provided).

### 11.3 Kubernetes

Job manifests in `k8s/`:

| File | Backend | Stage | GPUs |
|------|---------|-------|------|
| `stage1.yaml` | Qwen3-VL | 1 | 2x H100 80GB |
| `stage2.yaml` | Qwen3-VL | 2 | 2x H100 80GB |
| `internvl_stage1.yaml` | InternVL3 | 1 | 2x H100 80GB |
| `internvl_stage2.yaml` | InternVL3 | 2 | 2x H100 80GB |
| `eval.yaml` | Both | Eval | 1x A100 |
| `internvl_eval.yaml` | InternVL3 | Eval | 1x A100 |

Resource requests:
- **Training:** 2x `nvidia.com/gpu` (H100 80GB HBM3), 128Gi memory
- **Evaluation:** 1x `nvidia.com/gpu` (A100), 64Gi memory
- **NFS mount:** `/data` for shared datasets, `models/` (checkpoints), and HuggingFace cache
- **Secrets:** `HF_TOKEN` and `WANDB_API_KEY` via Kubernetes Secrets
- **TTL:** 36,000 seconds (10 hours)

### 11.4 Shell Scripts

**Directory:** `src/scripts/`

| Script | Purpose | Launcher |
|--------|---------|----------|
| `run_stage1.sh` | Stage 1 grounding | `deepspeed` |
| `run_stage2.sh` | Stage 2 fine-tuning | `deepspeed` |
| `run_eval.sh` | Benchmark evaluation | `python` |

All scripts:
- Use `set -euo pipefail`
- Auto-detect repo root from script location
- Support `BACKEND=qwen3vl|internvl3` environment variable
- Configure defaults per backend (output dirs, W&B project names)
- Accept overrides via environment variables

---

## 12. Design Decisions and Rationale

### 12.1 Bottleneck Projection

**Decision:** Project from $d_{model}$ (4096/3584) to $d_r = 512$ before cross-attention.

**Rationale:** Full-dimension cross-attention between 64 queries and hundreds/thousands of vision tokens at $d = 4096$ would be prohibitively expensive and difficult to train with only ~7M parameters. The bottleneck reduces computation quadratically while preserving sufficient representational capacity. The separate down-projections for text ($W_{\text{down\_t}}$) and vision ($W_{\text{down\_v}}$) allow each modality to be projected into the bottleneck space in a modality-specific way.

### 12.2 Two-Stage Cross-Attention

**Decision:** Separate task conditioning (text cross-attention) from visual re-inspection (vision cross-attention).

**Rationale:** A single cross-attention over concatenated text+vision tokens would conflate "understanding the question" with "finding the answer in the image." The two-stage design enforces an information flow: first understand *what* to look for, then *look for it*. This mirrors the cognitive process of re-inspection and provides clean attention maps for interpretability and supervision.

### 12.3 Zero-Initialized Output Projection

**Decision:** $W_{\text{up}}$ is initialized to all zeros.

**Rationale:** At the start of training, $R = W_{\text{up}} \cdot R_r = \mathbf{0}$. The R tokens contribute nothing to the LLM's computation, making the model equivalent to the original frozen VLM. This eliminates the cold-start problem: training begins from a known-good state rather than random perturbation of the base model's representations.

### 12.4 Per-Query Diversification

**Decision:** In focal loss, partition the bbox into vertical strips assigned to different queries.

**Rationale:** Without diversification, all 64 queries would be supervised to attend to the same bounding box region, leading to representational collapse. Vertical strip assignment encourages each query to specialize in a different spatial sub-region, maximizing the information density of the 64 R tokens.

### 12.5 Gradient Detachment in InternVL

**Decision:** R tokens are inserted via a `requires_grad=False` buffer in the InternVL backend.

**Rationale:** InternVL's frozen backbone layers use bf16, which has limited dynamic range ($\sim 10^{38}$). During backward pass, CE loss gradients can amplify through the frozen layers, producing `inf` values. When `clip_grad_norm_` encounters `inf`, the normalization involves `inf * 0 = NaN` (IEEE 754), corrupting all gradients. Detaching R tokens from the frozen backbone's computational graph eliminates this pathway entirely.

### 12.6 Forward Hook for Qwen Generation

**Decision:** Use a `register_forward_pre_hook` to inject R tokens during generation, rather than pre-computing `inputs_embeds`.

**Rationale:** Qwen3-VL's generation pipeline involves complex internal state management: 3D MRoPE position computation, DeepStack per-layer vision injection, and KV cache initialization. Pre-computing `inputs_embeds` and bypassing the base model's `generate()` would require reimplementing all of this state management. The hook approach inserts R tokens at the embedding level while preserving the base model's internal pipeline intact.

### 12.7 Separate Learning Rates

**Decision:** Use different learning rates for the reinspection module (1e-4 / 5e-5) and LoRA (2e-5).

**Rationale:** The reinspection module is randomly initialized and needs a higher learning rate to converge within a few epochs. LoRA adapters modify a pre-trained model's weights and benefit from a lower learning rate to avoid destabilizing learned representations. The 2.5x ratio between module and LoRA rates was empirically determined.

### 12.8 Stage 1 Attention Loss Weights

**Decision:** $\lambda = 10.0$ for Qwen focal loss, $\lambda = 0.5$ for InternVL.

**Rationale:** The focal loss and KL divergence operate on different scales. Focal loss values are typically much smaller than CE loss (due to the $(1-p_t)^\gamma$ focusing factor), requiring a larger weight to influence optimization. The InternVL KL loss produces larger values, so a smaller weight maintains balance with the CE loss. These values were tuned to ensure roughly equal gradient magnitudes from both loss terms.

---

## Appendix A: Output Dataclass

**Source:** `src/outputs.py`

```python
@dataclass
class ReInspectionOutput(ModelOutput):
    loss: Optional[FloatTensor] = None           # Total training loss
    logits: Optional[FloatTensor] = None         # LM head logits
    past_key_values: Optional[tuple] = None      # KV cache
    hidden_states: Optional[tuple] = None        # Layer hidden states
    attentions: Optional[tuple] = None           # LLM attention weights
    rope_deltas: Optional[LongTensor] = None     # MRoPE deltas (Qwen)
    image_hidden_states: Optional[FloatTensor] = None  # Vision features (InternVL)
    attn_task: Optional[FloatTensor] = None      # (B, N_q, N_t) task attention
    attn_vis: Optional[FloatTensor] = None       # (B, N_q, N_v) visual attention
```

## Appendix B: Project File Structure

```
learnable-reinspection/
    src/
        __init__.py
        config.py                    # ReInspectionConfig dataclass
        reinspection_module.py       # Core module (BottleneckCrossAttention, ReInspectionModule)
        outputs.py                   # ReInspectionOutput dataclass
        train.py                     # Hydra entry (deepspeed --module src.train …)
        hydra_util.py                # e.g. strip --local_rank for Hydra
        train_common.py              # Shared training loop, optimizer setup, checkpointing
        evaluate.py                  # Benchmark evaluation with comparison tables
        attn_loss.py                 # Focal and KL attention supervision losses
        visualize_attention.py       # Attention map visualization
        backends/
            __init__.py
            qwen3vl.py               # Qwen3VLWithReInspection wrapper
            internvl3.py             # InternVL3WithReInspection wrapper
        data/
            __init__.py
            refcoco.py               # RefCOCO dataset (Stage 1)
            spatial_dataset.py       # Spatial VQA datasets (Stage 2)
            utils.py                 # Qwen chat helpers, bbox_to_patch_mask
            chat_template.py         # InternVL chat helpers
        configs/
            config.yaml              # Hydra root defaults
            backend/
                internvl3.yaml
                internvl3_legacy.yaml
                qwen3vl.yaml
            stage/
                stage1.yaml
                stage2.yaml
                eval.yaml
            qwen3vl/
                deepspeed_z2.json
                deepspeed_z3.json
            internvl3/
                deepspeed_z2.json
                deepspeed_z3.json
        config_archive/
            internvl3/
                stage1_legacy.yaml
                stage2_legacy.yaml
        scripts/
            run_stage1.sh
            run_stage2.sh
            run_eval.sh
        requirements.txt
    k8s/
        stage1.yaml                  # Training job (runs run_stage1.sh)
        stage2.yaml
        eval.yaml
        secrets.yaml
    Dockerfile
    requirements.vlm.txt
    CLAUDE.md
    AGENTS.md
```
