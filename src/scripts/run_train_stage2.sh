#!/usr/bin/env bash
# Stage-2 training entrypoint. Defaults to the newest checkpoint from Stage 1.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/train_common.sh"

STAGE1_EXPERIMENT="${REINSPECTION_EXPERIMENT:-s1_qwen}"
STAGE2_EXPERIMENT="${REINSPECTION_EXPERIMENT_STAGE2:-s2_qwen}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-$(latest_stage1_checkpoint "$STAGE1_EXPERIMENT")}"

echo "Using Stage-1 checkpoint: ${STAGE1_CHECKPOINT}"
train "$STAGE2_EXPERIMENT" "stage1_checkpoint=${STAGE1_CHECKPOINT}" "$@" \
  2>&1 | tee "logs/${STAGE2_EXPERIMENT}.log"
