#!/bin/bash
# Stage 1: Grounding warm-up on RefCOCO
# Run with: bash scripts/run_stage1.sh

set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(dirname $(dirname $(realpath $0)))/../../"

DATA_ROOT="${DATA_ROOT:-/data/datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/internvl}"
NUM_GPUS="${NUM_GPUS:-2}"

WANDB_PROJECT="${WANDB_PROJECT:-reinspection-internvl3}"
WANDB_ARGS=()
[[ -n "${WANDB_PROJECT:-}" ]] && WANDB_ARGS+=(--wandb_project "$WANDB_PROJECT")
[[ -n "${WANDB_RUN_NAME:-}" ]] && WANDB_ARGS+=(--wandb_run_name "$WANDB_RUN_NAME")

torchrun --nproc_per_node=${NUM_GPUS} \
    -m reinspection_vlm.train \
    --backend internvl3 \
    --stage 1 \
    --data_root ${DATA_ROOT} \
    --output_dir ${OUTPUT_DIR} \
    --config reinspection_vlm/configs/internvl3/stage1.yaml \
    "${WANDB_ARGS[@]}"
