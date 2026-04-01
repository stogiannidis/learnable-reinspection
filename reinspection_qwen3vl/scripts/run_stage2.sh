#!/bin/bash
# Stage 2: Spatial reasoning fine-tuning with LoRA
# Run with: bash scripts/run_stage2.sh

set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(dirname $(dirname $(realpath $0)))/../../"

DATA_ROOT="${DATA_ROOT:-/data/datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen3vl}"
NUM_GPUS="${NUM_GPUS:-2}"
STAGE1_CKPT="${STAGE1_CKPT:-outputs/qwen3vl/stage1/epoch_5/reinspection_module.pt}"

WANDB_PROJECT="${WANDB_PROJECT:-reinspection-qwen3vl}"
WANDB_ARGS=()
[[ -n "${WANDB_PROJECT:-}" ]] && WANDB_ARGS+=(--wandb_project "$WANDB_PROJECT")
[[ -n "${WANDB_RUN_NAME:-}" ]] && WANDB_ARGS+=(--wandb_run_name "$WANDB_RUN_NAME")

torchrun --nproc_per_node=${NUM_GPUS} \
    -m reinspection_vlm.train \
    --backend qwen3vl \
    --stage 2 \
    --data_root ${DATA_ROOT} \
    --output_dir ${OUTPUT_DIR} \
    --config reinspection_vlm/configs/qwen3vl/stage2.yaml \
    --stage1_checkpoint ${STAGE1_CKPT} \
    "${WANDB_ARGS[@]}"
