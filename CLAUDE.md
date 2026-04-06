# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Learnable Re-Inspection: a two-stage cross-attention module that injects task-conditioned "R tokens" into a vision-language model (Qwen3-VL-8B) to improve spatial reasoning. The module operates in a bottleneck dimension (d_r=256) and is inserted between the user message and assistant response in the chat template.

## Commands

### PoC (synthetic grid)
```bash
pip install -r poc/requirements.txt
python -m poc.run --model both --epochs 30 --seeds 42 123 456
bash poc/scripts/run_poc.sh
```

### Unified VLM package (`reinspection_vlm/`)
Implementation is consolidated in `reinspection_vlm/`: shared `ReInspectionModule`, `train_common.py`, `evaluate.py`, and backend-specific wrappers under `reinspection_vlm/backends/` (`qwen3vl.py`, `internvl3.py`). Training and eval use **Hydra**: compose `reinspection_vlm/configs/config.yaml` with config groups `backend/` (`qwen3vl`, `internvl3`) and `stage/` (`stage1`, `stage2`, `eval`). DeepSpeed JSON configs remain under `reinspection_vlm/configs/{qwen3vl,internvl3}/`.

```bash
pip install -r reinspection_vlm/requirements.txt

# Training / eval (Hydra overrides after the script name)
bash reinspection_vlm/scripts/run_stage1.sh
bash reinspection_vlm/scripts/run_stage2.sh
bash reinspection_vlm/scripts/run_eval.sh

# Examples
deepspeed --module reinspection_vlm.train stage=stage1 backend=qwen3vl data_root=/path/to/data
deepspeed --module reinspection_vlm.train stage=stage1 backend=internvl3_legacy   # older InternVL checkpoint geometry
# Reference copies: reinspection_vlm/config_archive/internvl3/stage1_legacy.yaml, stage2_legacy.yaml
deepspeed --module reinspection_vlm.train stage=stage2 stage1_checkpoint=models/internvl3/stage1/epoch_5/reinspection_module.pt
python -m reinspection_vlm.evaluate stage=eval checkpoint_dir=models/internvl3/stage2/epoch_4 data_root=/path/to/data

# Attention visualization (Qwen)
python -m reinspection_vlm.visualize_attention \
    --checkpoint_dir models/qwen3vl/stage2/epoch_10 \
    --image path/to/image.jpg --question "Where is the cat?"
```

Training uses DeepSpeed (`deepspeed --module reinspection_vlm.train ...`) for multi-GPU runs. Pass paths and W&B names as **Hydra overrides** (e.g. `data_root=...`, `output_dir=...`, `wandb_run_name=...`, `stage1_checkpoint=...`) rather than a separate argparse CLI.

## Architecture

### ReInspectionModule (`reinspection_vlm/reinspection_module.py`)
Two-stage cross-attention in bottleneck d_r=256:
- **Stage 1 (task conditioning):** `CrossAttn(Q^0, W_down_t(T))` → Q^1
- **Stage 2 (visual re-inspection):** `CrossAttn(Q^1, W_down_v(V))` → R_r
- `R = W_up(R_r)` projected back to d_model=4096

Key components: `W_down_t`, `W_down_v` (4096→256), `W_up` (256→4096), N_q=32 learnable queries, 4 attention heads. Returns R tokens plus attention maps (A_task, A_vis) for interpretability and L_attn supervision.

### Qwen3VLWithReInspection (`reinspection_vlm/backends/qwen3vl.py`)
Wraps `Qwen3VLForConditionalGeneration` (not subclassed). Forward flow:
1. Embed input_ids, encode vision features via ViT, scatter into embeddings
2. Find last `<|im_start|>` position (before assistant turn)
3. Extract V (vision tokens) and T (text tokens before insert point)
4. Run `ReInspectionModule(V, T)` → R tokens
5. Insert R tokens at the found position, extending attention_mask, position_ids (MRoPE: text dim incrementing, spatial dims=0), and labels (-100 for R positions)
6. Forward through LLM, compute CE loss

The `generate()` method follows the same injection during prefill.

### Training pipeline (`reinspection_vlm/train.py` + Hydra)
- **Stage 1:** Freeze backbone; train reinspection (and optionally InternVL projector). Loss = L_CE + λ·L_attn. Qwen uses focal loss with diversified bbox targets (`stage1_attn_loss_type: focal`); InternVL defaults to KL (`stage1_attn_loss_type: kl`).
- **Stage 2:** Load Stage 1 checkpoint. Add LoRA (r=16, q_proj+v_proj) to LLM. Train reinspection + LoRA. Loss = L_CE only.
- DeepSpeed (ZeRO per `reinspection_vlm/configs/*/deepspeed_*.json`). Checkpoints save under `{output_dir}/{backend}/stage{N}/epoch_{E}/` (default `output_dir=models`): `reinspection_module.pt` and optionally `lora_weights/`.

### Data (`reinspection_vlm/data/`)
- `RefCOCODataset`: Stage 1; `backend='qwen3vl'|'internvl3'` selects processor/chat and bbox→patch supervision.
- `SpatialVQADataset` / `build_spatial_dataset()`: Stage 2 spatial VQA; same `backend` flag.
- Qwen chat helpers: `data/utils.py`; InternVL: `data/chat_template.py`.

### Config (`reinspection_vlm/config.py`)
Single `ReInspectionConfig` dataclass (union of Qwen + InternVL fields). Defaults come from Hydra (`configs/config.yaml` + `backend/*.yaml` + `stage/*.yaml`); override on the command line (`key=value`) or add YAML under those groups.

## Deployment

- Docker: `Dockerfile` for the shared Qwen/InternVL VLM image, `Dockerfile.poc` for the PoC image
- Kubernetes: `k8s/stage1.yaml`, `k8s/stage2.yaml`, `k8s/eval.yaml` (invoke `reinspection_vlm/scripts/run_*.sh`; adjust image/workdir for your cluster)
- Secrets (HF_TOKEN, WANDB_API_KEY) via k8s Secrets; see `k8s/secrets.example.yaml`

## Conventions

- Python 4-space tabbed indent, `snake_case` functions, `PascalCase` classes
- Shell scripts use `set -euo pipefail`
- Commit messages: scoped imperative style, e.g. `qwen3vl: fix stage2 checkpoint loading`
- Env vars for paths/credentials, never hardcoded
