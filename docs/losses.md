# Losses

Reference for every training loss currently used by the codebase, grouped by
stage and tied back to the implementation.

This document describes the **effective behavior of the current code**, not
just the intended training design.

---

## Overview

Training is split into two stages:

- **Stage 1** trains the re-inspection module on referring-expression grounding
  data, with optional auxiliary supervision.
- **Stage 2** continues training on spatial VQA with standard causal language
  modeling only.

At a high level:

- **Stage 1 total loss**
  `L_stage1 = L_ce + lambda_attn * L_attn + lambda_ground * L_ground + lambda_qt * L_qt`
- **Stage 2 total loss**
  `L_stage2 = L_ce`

The main implementation lives in:

- `src/training/trainer.py`
- `src/model/attn_loss.py`
- `src/model/bbox_head.py`
- `src/model/query_text_infonce.py`
- backend wrappers in `src/backends/*.py`

## Notation

The formulas below use the following symbols:

- `B`: batch size
- `Q` or `N_q`: number of learned re-inspection queries
- `V`: number of visual tokens seen by the re-inspection module
- `S_t`: number of text tokens extracted for the re-inspection module
- `d_r`: bottleneck width (`d_bottleneck` in config)
- `L_ce`: causal language-model cross-entropy
- `L_attn`: attention KL loss
- `L_ground`: box grounding loss
- `L_qt`: query-text InfoNCE

Unless stated otherwise, every loss described here is reduced to a single
scalar per batch before being combined by the trainer.

---

## Stage 1

Stage 1 always starts from the model's causal language-model loss and may add
up to three auxiliary terms:

1. attention KL loss
2. box grounding loss
3. query-text InfoNCE

The trainer builds the Stage-1 objective in `src/training/trainer.py` inside
the main loop and validation loop.

### Stage-1 supervision sources

Stage 1 combines signals coming from both the dataset and the model outputs.

Per-sample dataset fields built by `src/data/refcoco.py`:

- `labels`: token-level supervision for coordinate generation
- `attn_target_mask`: box-derived distribution over visual tokens
- `bbox_norm`: normalized target box `[x1, y1, x2, y2]`

Relevant model outputs exposed through `ReInspectionOutput`:

- `loss`: backend LM loss
- `attn_vis`: mean-over-head visual attention of re-inspection queries
- `R_bottleneck`: final bottleneck query states after vision interaction
- `Q_text_bottleneck`: query states after text interaction, before vision
- `text_bottleneck`: down-projected text tokens
- `text_bottleneck_mask`: valid text-token mask

### Total Stage-1 objective

In the current trainer, the per-batch loss is:

```text
L_stage1 =
    L_ce
  + stage1_attn_loss_weight * L_attn           if stage1_use_attn_loss
  + stage1_grounding_loss_weight * L_ground    if stage1_use_grounding_loss
  + stage1_query_text_infonce_weight * L_qt    if stage1_use_query_text_infonce
```

The toggles come from `ReInspectionConfig`:

- `stage1_use_attn_loss`
- `stage1_use_grounding_loss`
- `stage1_use_query_text_infonce`

The corresponding weights come from:

- `stage1_attn_loss_weight`
- `stage1_grounding_loss_weight`
- `stage1_query_text_infonce_weight`

### When each Stage-1 term is active

Each term has a different activation condition in code:

- `L_ce` is active whenever the backend returns `outputs.loss`
- `L_attn` requires `stage1_use_attn_loss`, `outputs.attn_vis`, and
  `attn_target_mask` in the batch
- `L_ground` requires `stage1_use_grounding_loss`, `outputs.R_bottleneck`,
  `bbox_norm` in the batch, and an attached `bbox_head`
- `L_qt` requires `stage1_use_query_text_infonce` and the three optional
  tensors `Q_text_bottleneck`, `text_bottleneck`, and `text_bottleneck_mask`

### 1. Causal LM loss (`L_ce`)

This is the base supervised language-model loss returned by each backend as
`outputs.loss`.

What it supervises:

- **Stage 1**: the model is trained to generate the target box coordinates as
  text, for example `"[x1, y1, x2, y2]"`.
- Prompt tokens are masked to `-100`, so only assistant answer tokens
  contribute to the loss.

How it is computed:

- **InternVL3**: chunked next-token cross-entropy over shifted logits/labels
  in `src/backends/internvl3.py`
- **Gemma 4**: chunked next-token cross-entropy over shifted logits/labels
  in `src/backends/gemma4.py`
- **Qwen2.5-VL**: backend-native LM loss via `self.base_model.loss_function`
  in `src/backends/qwen25vl.py`

Common behavior:

- labels before the answer span are set to `-100`
- tokens after shifting are scored with causal next-token prediction

Where the labels come from in Stage 1:

- `RefCOCODataset` converts the ground-truth box to a normalized coordinate
  string such as `"[0.123, 0.456, 0.789, 0.900]"`
- the prompt-only version of the conversation is tokenized first
- the full prompt-plus-answer conversation is tokenized next
- all prompt positions are masked to `-100` in `labels`

For InternVL3 and Gemma 4, the loss is explicitly computed on shifted tensors:

```text
shift_logits = logits[..., :-1, :]
shift_labels = labels[..., 1:]
L_ce = CrossEntropy(shift_logits, shift_labels, ignore_index=-100)
```

InternVL3 and Gemma 4 use chunked cross-entropy helpers to avoid materializing
very large `(B*T, vocab)` tensors in FP32 all at once.

This is the only loss term that is always present in Stage 1.

### 2. Attention KL loss (`L_attn`)

Implementation:

- `src/model/attn_loss.py::compute_attn_loss_kl`
- called by `_stage1_attn_loss` in `src/training/trainer.py`

Purpose:

- supervise the re-inspection module's visual attention `attn_vis`
  against a target patch distribution derived from the ground-truth box

Where `attn_vis` comes from:

- the re-inspection module first performs text cross-attention, then vision
  cross-attention
- when attention weights are requested, the vision cross-attention layer
  materializes `softmax(scores)` over visual tokens
- the module averages those attention weights over heads before returning
  `attn_vis`

Inputs:

- `attn_vis`: shape `(B, Q, V)`
  - `B`: batch size
  - `Q`: number of learned bottleneck queries
  - `V`: number of visual tokens
- `attn_target_mask`: shape `(B, V)`
  - built from the normalized ground-truth box
  - uniform mass over patches that fall inside the box

How the target is built:

- **InternVL3**: `intern_bbox_to_patch_mask(...)` in `src/data/refcoco.py`
- **Qwen2.5-VL**: `bbox_to_patch_mask(...)` in `src/data/utils.py`
- **Gemma 4**: current dataset path returns an empty attention target, so this
  loss has no effective supervision unless that dataset path is changed

Target construction semantics:

- the target is not a learned heatmap or segmentation mask
- it is a uniform distribution over all visual tokens whose patch region lies
  inside the target box
- InternVL repeats the patch mask over image crops if multiple crops are used
- Qwen constructs the target over the merged patch grid defined by
  `image_grid_thw` and `spatial_merge_size=2`

Formula:

```text
pred   = normalize(attn_vis over V)
target = normalize(attn_target_mask over V)
target is broadcast across Q queries

L_attn = KL(target || pred), averaged with batchmean reduction
         and divided by Q
```

Implementation details:

- both `pred` and `target` are clamped to at least `1e-8`
- if `attn_target_mask` width does not match `attn_vis.shape[-1]`, the trainer
  truncates or zero-pads the target before computing the loss
- the current trainer always uses KL; it does not branch to focal loss
- because the KL helper renormalizes `target`, zero-padding changes the support
  but does not change the fact that the final target is a probability
  distribution

Important note:

- backend YAMLs still contain `stage1_attn_loss_type: kl`
- the current trainer no longer reads that field
- effective behavior is simply: if `stage1_use_attn_loss` is `true`, the code
  uses **KL attention supervision**

### 3. Grounding loss (`L_ground`)

Implementation:

- `src/model/bbox_head.py::BboxHead`
- `src/model/bbox_head.py::compute_grounding_loss`
- called by `_stage1_grounding_loss` in `src/training/trainer.py`

Purpose:

- regress a normalized box directly from the bottleneck representation

Where `R_bottleneck` comes from:

- after the queries attend to text, they attend to visual tokens
- the resulting query states are passed through a second FFN block
- that final query tensor is exposed as `R_bottleneck` and is used both for
  token reinsertion and for optional box regression

Inputs:

- `R_bottleneck`: final bottleneck query states of shape `(B, N_q, d_r)`
- `bbox_norm`: normalized target box of shape `(B, 4)` with coordinates
  `[x1, y1, x2, y2]`

How predictions are formed:

1. mean-pool `R_bottleneck` over the query dimension
2. pass through a 3-layer MLP with GELU activations
3. apply sigmoid to keep coordinates in `[0, 1]`

Formula:

```text
L_ground_raw = stage1_grounding_l1_weight   * L1(box_pred, box_gt)
             + stage1_grounding_giou_weight * (1 - GIoU(box_pred, box_gt))
```

What the GIoU helper does:

- sorts predicted corners so `(x1, y1)` is the top-left and `(x2, y2)` is the
  bottom-right before IoU math
- computes intersection area, union area, and enclosing-box area
- returns the mean of `1 - GIoU`

Warmup:

The trainer multiplies the raw grounding loss by:

```text
warmup = min(1.0, global_step / stage1_grounding_warmup_steps)
```

So the effective term is:

```text
L_ground = warmup * L_ground_raw
```

Then the outer Stage-1 loss applies the global weight
`stage1_grounding_loss_weight`.

Important implementation detail:

`L_ground` is only non-zero if the model has an attached `bbox_head`.
The current trainer attaches that head in:

- `_setup_model_intern_stage1(...)`

It does **not** currently attach the head in:

- `_setup_model_qwen25_stage1(...)`
- `_setup_model_gemma4_stage1(...)`

That means:

- **InternVL3 Stage 1**: grounding loss can be active
- **Qwen2.5-VL Stage 1**: `stage1_use_grounding_loss` may be `true` in config,
  but the trainer returns zero because no `bbox_head` is attached
- **Gemma 4 Stage 1**: same as Qwen; the grounding term is configured but not
  actually active with the current setup code

### 4. Query-text InfoNCE (`L_qt`)

Implementation:

- `src/model/query_text_infonce.py::compute_query_text_infonce_loss`
- called by `_stage1_query_text_infonce_loss` in `src/training/trainer.py`

Purpose:

- align the text-conditioned query states with the corresponding text
  representation in bottleneck space

Where the two sides come from:

- `Q_text_bottleneck` is the query tensor immediately after the text
  cross-attention block and FFN
- `text_bottleneck` is the text stream after projection from model width
  `d_model` down to bottleneck width `d_r`
- `text_bottleneck_mask` marks valid text positions for masked averaging

Inputs:

- `Q_text_bottleneck`: shape `(B, N_q, d_r)`
  - queries after text cross-attention and FFN, before vision cross-attention
- `text_bottleneck`: shape `(B, S_t, d_r)`
  - down-projected text token states
- `text_bottleneck_mask`: shape `(B, S_t)`
  - valid-token mask

These tensors come from the re-inspection module and are returned through
`ReInspectionOutput` only when `return_query_text_tensors=True`.

How it is computed:

1. mean-pool query states over the learned query slots
2. masked-mean pool text bottleneck states over valid tokens
3. L2-normalize both pooled representations
4. compute in-batch similarity matrix
5. apply symmetric cross-entropy in both directions

Formula:

```text
q_i = normalize(mean_queries_i)
t_i = normalize(masked_mean_text_i)

logits_ij = (q_i dot t_j) / temperature

L_qt = 0.5 * [ CE(logits, labels=i) + CE(logits^T, labels=i) ]
```

Edge case:

- if batch size is `< 2`, the implementation returns zero, because in-batch
  contrastive learning needs negatives

Default status:

- disabled by default in all backend YAMLs via `stage1_use_query_text_infonce: false`

---

## Stage 1 by backend

The current defaults and effective losses are:

| Backend | `stage1_use_attn_loss` | `stage1_attn_loss_weight` | `stage1_use_grounding_loss` | `stage1_grounding_loss_weight` | `stage1_use_query_text_infonce` | `stage1_query_text_infonce_weight` | Effective default Stage-1 objective |
|---------|------------------------|---------------------------|-----------------------------|--------------------------------|----------------------------------|-------------------------------------|-------------------------------------|
| InternVL3 | `true` | `0.5` | `false` | `2.0` | `false` | `0.1` | `L_ce + 0.5 * L_attn` |
| Qwen2.5-VL | `true` | `0.5` | `true` | `2.0` | `false` | `0.1` | effectively `L_ce + 0.5 * L_attn` |
| Gemma 4 | `true` | `0.5` | `true` | `2.0` | `false` | `0.1` | effectively close to `L_ce` unless Stage-1 supervision paths are fixed |

Why Gemma's attention term is ineffective:

- the Stage-1 Gemma dataset path currently sets `attn_target_mask` to an empty
  tensor in `src/data/refcoco.py`
- the trainer pads that target to the attention width, and the KL helper then
  clamps and normalizes it, which effectively turns the target into a near-
  uniform distribution rather than a box-localized supervision signal
- this is a code-path mismatch worth fixing if Gemma Stage 1 is expected to use
  attention supervision

Why Qwen and Gemma grounding are ineffective in the current trainer:

- their Stage-1 config enables grounding
- but their Stage-1 model setup does not call `_attach_bbox_head(...)`
- `_stage1_grounding_loss(...)` returns zero when the model has no `bbox_head`

---

## Stage 2

Stage 2 uses standard spatial VQA instruction tuning.

### Total Stage-2 objective

```text
L_stage2 = L_ce
```

There are no Stage-2 auxiliary losses in the current trainer.

What `L_ce` supervises:

- answer generation for Stage-2 spatial VQA examples
- only assistant answer tokens contribute
- prompt tokens remain masked with `-100`

The Stage-2 dataset loader is `src/data/spatial_dataset.py`, which constructs
labels exactly this way for all three backends.

Stage-2 label construction follows the same basic pattern as Stage 1:

1. tokenize the prompt-only conversation
2. tokenize the full prompt-plus-answer conversation
3. mask prompt positions to `-100`
4. compute causal next-token cross-entropy on the answer tokens

There is no attention target, box target, or query-text contrastive target in
the Stage-2 dataset path.

### What is trainable in Stage 2

Stage 2 is not just "plain SFT on the whole model." The loss is still only
cross-entropy, but the trainable parameters are restricted:

- the re-inspection module remains trainable
- the base model stays frozen
- LoRA adapters are attached to the language model and trained alongside the
  re-inspection module

So Stage 2 changes the **trainable parameter set**, not the loss formula.

---

## Reduction and optimization details

Some trainer behavior matters when interpreting reported loss values.

### Gradient accumulation

The trainer computes the full scalar loss for the current batch, then divides it
by `grad_accum` before `backward()`:

```text
micro_loss = L_total / grad_accum
```

This means:

- logged `loss`, `ce_loss`, `attn_loss`, `grounding_loss`, and
  `query_text_infonce_loss` are the **pre-division** values
- only the tensor sent into backprop is scaled by gradient accumulation

### Validation reduction

Validation accumulates already-reduced scalar losses and averages them over the
number of finite validation batches:

```text
metric = sum(batch_metric) / num_finite_batches
```

If distributed training is active, these sums are all-reduced across ranks
before the final average is computed.

### Non-finite loss handling

For both training and validation:

- the trainer checks `torch.isfinite(total_loss)`
- if any rank sees a non-finite loss, the batch is skipped
- skipped batches do not contribute to epoch averages

---

## Validation losses

Validation uses the same formulas as training:

- same `L_ce`
- same optional Stage-1 auxiliary terms
- same weighting

One difference:

- validation forces grounding warmup to full strength by passing a very large
  `global_step`, so the validation grounding term reflects the fully warmed-up
  loss rather than the early-training scaled version

---

## Logging

The trainer logs individual loss components and combined loss terms.

Stage 1 can log:

- `loss`
- `ce_loss`
- `attn_loss`
- `grounding_loss`
- `query_text_infonce_loss`
- `stage1_supervision_loss`

where:

```text
stage1_supervision_loss =
    stage1_attn_loss_weight * attn_loss
  + stage1_grounding_loss_weight * grounding_loss
  + stage1_query_text_infonce_weight * query_text_infonce_loss
```

Stage 2 logs only:

- `loss`
- `ce_loss`

---

## Current caveats

These are important if you are comparing experiments:

1. `stage1_attn_loss_type` exists in backend YAMLs but is not read by the
   current trainer. Effective behavior is KL attention supervision only.
2. The grounding head is only attached in the InternVL Stage-1 setup, so Qwen
   and Gemma currently do not get a non-zero grounding term even if configured.
3. Gemma Stage-1 currently does not build a real `attn_target_mask`, so its
   attention loss path is not meaningfully supervised as written.
4. Query-text InfoNCE is implemented end to end, but disabled by default.

---

## Summary

- **Stage 1** is designed as `CE + optional auxiliary losses`
- the auxiliary losses currently implemented are:
  - attention KL
  - box grounding (`L1 + GIoU`, with warmup)
  - query-text InfoNCE
- **Stage 2** uses only causal LM cross-entropy
- the current code has a few wiring mismatches, so configured losses and
  effective losses are not identical for every backend
