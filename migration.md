# DeepSpeed ZeRO-3 Migration

## What changed

| File | Change |
|------|--------|
| `src/configs/qwen3vl/deepspeed_z3.json` | New ZeRO-3 config |
| `src/configs/internvl3/deepspeed_z3.json` | New ZeRO-3 config |
| `src/train_common.py` | `_save_checkpoint` uses `GatheredParameters` for ZeRO-3 |
| `src/scripts/run_stage1.sh` | Opt-in `DEEPSPEED_CONFIG` env var |
| `src/scripts/run_stage2.sh` | Opt-in `DEEPSPEED_CONFIG` env var |
| `k8s/stage1.yaml`, `k8s/stage2.yaml` | `DEEPSPEED_CONFIG` set to qwen3vl z3 config |
| `k8s/internvl_stage1.yaml`, `k8s/internvl_stage2.yaml` | `DEEPSPEED_CONFIG` set to internvl3 z3 config |

## Why ZeRO-3 vs ZeRO-2

| | ZeRO-2 | ZeRO-3 |
|-|--------|--------|
| Model params | Replicated (full copy per GPU) | **Sharded** across GPUs |
| Gradients | Sharded | Sharded |
| Optimizer states | Sharded | Sharded |
| Per-GPU VRAM | ~full model | ~1/N of model |
| Throughput overhead | Low | ~5-10% (gather ops) |

With 2×H100 and Qwen3-VL-8B (bf16 ≈ 16 GB weights), ZeRO-3 frees ~8 GB per GPU compared to ZeRO-2.

## Key code change: checkpoint saving

ZeRO-3 partitions model parameters across GPUs. Calling `.state_dict()` on rank 0 alone yields only the local shard. The fix uses `deepspeed.zero.GatheredParameters` — a collective that gathers the full tensor on rank 0 while all other ranks participate:

```python
with deepspeed.zero.GatheredParameters(reinsp_params, modifier_rank=0):
    if is_main_process():
        torch.save(unwrapped.reinspection.state_dict(), path)
```

All ranks must enter the `with` block (collective op). Only rank 0 writes to disk.

## Usage

### Local / interactive
```bash
# Qwen Stage 1 with ZeRO-3
DEEPSPEED_CONFIG=src/configs/qwen3vl/deepspeed_z3.json \
  bash src/scripts/run_stage1.sh

# InternVL Stage 2 with ZeRO-3
BACKEND=internvl3 \
DEEPSPEED_CONFIG=src/configs/internvl3/deepspeed_z3.json \
  bash src/scripts/run_stage2.sh

# Without DeepSpeed (plain DDP, unchanged behaviour)
bash src/scripts/run_stage1.sh
```

### Kubernetes
The k8s YAMLs already set `DEEPSPEED_CONFIG` — just apply as usual:
```bash
kubectl apply -f k8s/stage1.yaml
```

## Verification checklist
1. `nvidia-smi` during training: per-GPU VRAM should be ~8 GB lower than plain DDP
2. After epoch 1: `models/<backend>/stage1/epoch_1/reinspection_module.pt` (or your `output_dir`) exists and loads cleanly:
   ```python
   import torch
   torch.load("models/internvl3/stage1/epoch_1/reinspection_module.pt", map_location="cpu")
   ```
3. Stage 2 loads the stage 1 checkpoint without error (`_load_stage1_weights` runs before DeepSpeed init — no changes needed there)
