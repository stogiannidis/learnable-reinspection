#!/bin/bash
# Evaluation on spatial reasoning benchmarks (InternVL3)
#
# Runs all three conditions (frozen baseline, lora_only, reinspection) by default.
# Set EVAL_CONDITION to run a single condition instead.
#
# Run with: bash reinspection_internvl3/scripts/run_eval.sh

set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(dirname $(dirname $(realpath $0)))/../../"

DATA_ROOT="${DATA_ROOT:-/data/datasets}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-outputs/internvl3/stage2/epoch_10}"
OUTPUT_FILE="${OUTPUT_FILE:-outputs/internvl3/eval_results.json}"
WANDB_PROJECT="${WANDB_PROJECT:-reinspection-internvl3}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"

# Set EVAL_CONDITION to "frozen", "lora_only", or "reinspection" to run a
# single condition. Leave empty (default) to run --compare (all three).
EVAL_CONDITION="${EVAL_CONDITION:-}"

CONDITION_ARGS=""
if [ -n "${EVAL_CONDITION}" ]; then
    CONDITION_ARGS="--condition ${EVAL_CONDITION}"
else
    CONDITION_ARGS="--compare"
fi

WANDB_ARGS="--wandb_project ${WANDB_PROJECT}"
if [ -n "${WANDB_RUN_NAME}" ]; then
    WANDB_ARGS="${WANDB_ARGS} --wandb_run_name ${WANDB_RUN_NAME}"
fi

CONFIG="${CONFIG:-reinspection_vlm/configs/internvl3/stage2.yaml}"

python -m reinspection_vlm.evaluate \
    --backend internvl3 \
    --config ${CONFIG} \
    --data_root ${DATA_ROOT} \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --output_file ${OUTPUT_FILE} \
    --benchmarks vsr whatsup gqa_spatial spatialbench \
    ${CONDITION_ARGS} \
    ${WANDB_ARGS}
