# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Learnable Re-Inspection: a two-stage cross-attention module that injects task-conditioned "R tokens" into a vision-language model to improve spatial reasoning. The module operates in a bottleneck dimension and is inserted between the user message and assistant response in the chat template. Supported VLM backends: Qwen3-VL-8B, Qwen2.5-VL-7B, InternVL3-8B, and Gemma 4 (12B).

## Commands

### PoC (synthetic grid)
```bash
pip install -r poc/requirements.txt
python -m poc.run --model both --epochs 30 --seeds 42 123 456
bash poc/scripts/run_poc.sh
```

### Unified VLM package (`src/`)
Implementation is consolidated in `src/`: shared `ReInspectionModule`, `train_common.py`, `evaluate.py`, and backend-specific wrappers under `src/backends/` (`qwen3vl.py`, `qwen25vl.py`, `internvl3.py`, `gemma4.py`). Training and eval use **Hydra**: compose `src/configs/config.yaml` with config groups `backend/` (`qwen3vl`, `qwen25vl`, `internvl3`, `gemma4`) and `stage/` (`stage1`, `stage2`, `eval`). DeepSpeed JSON configs are under `src/configs/`.

```bash
pip install -r src/requirements.txt

# Training / eval (Hydra overrides after the script name)
bash src/scripts/run_stage1.sh
bash src/scripts/run_stage2.sh
bash src/scripts/run_eval.sh

# Examples
deepspeed --module src.train stage=stage1 backend=qwen3vl data_root=/path/to/data
deepspeed --module src.train stage=stage1 backend=qwen25vl data_root=/path/to/data
deepspeed --module src.train stage=stage1 backend=gemma4 data_root=/path/to/data
deepspeed --module src.train stage=stage1 backend=internvl3_legacy   # older InternVL checkpoint geometry
# Reference copies: src/config_archive/internvl3/stage1_legacy.yaml, stage2_legacy.yaml
deepspeed --module src.train stage=stage2 stage1_checkpoint=models/internvl3/stage1/epoch_5/reinspection_module.pt
python -m src.evaluate stage=eval backend=gemma4 checkpoint_dir=models/gemma4/stage2/epoch_4 data_root=/path/to/data
python -m src.evaluate stage=eval checkpoint_dir=models/internvl3/stage2/epoch_4 data_root=/path/to/data

# Attention visualization (Qwen)
python -m src.visualize_attention \
    --checkpoint_dir models/qwen3vl/stage2/epoch_10 \
    --image path/to/image.jpg --question "Where is the cat?"
```

Training uses DeepSpeed (`deepspeed --module src.train ...`) for multi-GPU runs. Pass paths and W&B names as **Hydra overrides** (e.g. `data_root=...`, `output_dir=...`, `wandb_run_name=...`, `stage1_checkpoint=...`) rather than a separate argparse CLI.

## Architecture

### ReInspectionModule (`src/reinspection_module.py`)
Two-stage cross-attention in bottleneck d_r=256:
- **Stage 1 (task conditioning):** `CrossAttn(Q^0, W_down_t(T))` → Q^1
- **Stage 2 (visual re-inspection):** `CrossAttn(Q^1, W_down_v(V))` → R_r
- `R = W_up(R_r)` projected back to d_model=4096

Key components: `W_down_t`, `W_down_v` (d_model→d_bottleneck), `W_up` (d_bottleneck→d_model), N_q learnable queries, multi-head attention. Returns R tokens plus attention maps (A_task, A_vis) for interpretability and L_attn supervision.

### Backend wrappers (`src/backends/`)
Each backend wraps its base HF model (not subclassed) and follows the same flow:
1. Embed input_ids, encode vision features via ViT, scatter into embeddings
2. Find the insert point (before assistant turn marker)
3. Extract V (vision tokens) and T (text tokens before insert point)
4. Run `ReInspectionModule(V, T)` → R tokens
5. Insert R tokens, extending attention_mask, position_ids, and labels
6. Forward through LLM, compute loss

| Backend | Wrapper class | d_model | Position encoding | Chat template |
|---|---|---|---|---|
| `qwen3vl` | `Qwen3VLWithReInspection` | 4096 | 3D MRoPE | `<\|im_start\|>` / `<\|im_end\|>` |
| `qwen25vl` | `Qwen25VLWithReInspection` | 3584 | 3D MRoPE | `<\|im_start\|>` / `<\|im_end\|>` |
| `internvl3` | `InternVL3WithReInspection` | 4096 | 1D RoPE | InternVL chat template |
| `gemma4` | `Gemma4WithReInspection` | 3840 | 1D RoPE + 2D vision | `<start_of_turn>` / `<end_of_turn>` |

Qwen backends use a forward pre-hook for `generate()` (placeholder → R replacement). InternVL3 and Gemma4 pre-build `inputs_embeds` and pass them directly.

### Training pipeline (`src/train.py` + Hydra)
- **Stage 1:** Freeze backbone; train reinspection (and optionally InternVL projector). Loss = L_CE + λ·L_attn. Qwen uses focal loss with diversified bbox targets (`stage1_attn_loss_type: focal`); InternVL defaults to KL (`stage1_attn_loss_type: kl`).
- **Stage 2:** Load Stage 1 checkpoint. Add LoRA (r=16, q_proj+v_proj) to LLM. Train reinspection + LoRA. Loss = L_CE only.
- DeepSpeed (ZeRO per `src/configs/*/deepspeed_*.json`). Checkpoints save under `{output_dir}/{backend}/stage{N}/epoch_{E}/` (default `output_dir=models`): `reinspection_module.pt` and optionally `lora_weights/`.

### Data (`src/data/`)
- `RefCOCODataset`: Stage 1; `backend='qwen3vl'|'qwen25vl'|'internvl3'|'gemma4'` selects processor/chat and bbox→patch supervision.
- `SpatialVQADataset` / `build_spatial_dataset()`: Stage 2 spatial VQA; same `backend` flag.
- Qwen chat helpers: `data/utils.py`; InternVL: `data/chat_template.py`; Gemma4: `data/gemma4_chat.py`.

### Config (`src/config.py`)
Single `ReInspectionConfig` dataclass (union of Qwen + InternVL fields). Defaults come from Hydra (`configs/config.yaml` + `backend/*.yaml` + `stage/*.yaml`); override on the command line (`key=value`) or add YAML under those groups.

## Deployment

- Docker: `Dockerfile` for the shared Qwen/InternVL VLM image, `Dockerfile.poc` for the PoC image
- Kubernetes: `k8s/stage1.yaml`, `k8s/stage2.yaml`, `k8s/eval.yaml` (invoke `src/scripts/run_*.sh`; adjust image/workdir for your cluster)
- Secrets (HF_TOKEN, WANDB_API_KEY) via k8s Secrets; see `k8s/secrets.example.yaml`

## Conventions

- Python 4-space tabbed indent, `snake_case` functions, `PascalCase` classes
- Shell scripts use `set -euo pipefail`
- Commit messages: scoped imperative style, e.g. `qwen3vl: fix stage2 checkpoint loading`
- Env vars for paths/credentials, never hardcoded
