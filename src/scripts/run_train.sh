#!/usr/bin/env bash
# Full training pipeline: Stage 1, then Stage 2 with the newest Stage-1 checkpoint.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/train_common.sh"

STAGE1_EXPERIMENT="${REINSPECTION_EXPERIMENT:-s1_qwen}"
# Unset → default s2_qwen. Explicitly empty → Stage-1 only (see k8s/stage1_*.yaml).
if [[ -z "${REINSPECTION_EXPERIMENT_STAGE2+x}" ]]; then
  STAGE2_EXPERIMENT="s2_qwen"
elif [[ -z "${REINSPECTION_EXPERIMENT_STAGE2}" ]]; then
  STAGE2_EXPERIMENT=""
else
  STAGE2_EXPERIMENT="${REINSPECTION_EXPERIMENT_STAGE2}"
fi
PIPELINE_LOG="logs/pipeline_${STAGE1_EXPERIMENT}__${STAGE2_EXPERIMENT:-stage1_only}.log"
: > "$PIPELINE_LOG"

{ echo "=== Stage 1: +experiment=${STAGE1_EXPERIMENT} ==="; \
  "${SCRIPT_DIR}/run_train_stage1.sh" "$@"; } \
  2>&1 | tee -a "$PIPELINE_LOG"

if [[ -z "${STAGE2_EXPERIMENT}" ]]; then
  echo "=== Stage 2 skipped (REINSPECTION_EXPERIMENT_STAGE2 is empty) ===" | tee -a "$PIPELINE_LOG"
  exit 0
fi

CHECKPOINT=$(latest_stage1_checkpoint "$STAGE1_EXPERIMENT")

{ echo "=== Stage 2: +experiment=${STAGE2_EXPERIMENT} stage1_checkpoint=${CHECKPOINT} ==="; \
  REINSPECTION_EXPERIMENT_STAGE2="${STAGE2_EXPERIMENT}" \
  STAGE1_CHECKPOINT="$CHECKPOINT" "${SCRIPT_DIR}/run_train_stage2.sh" "$@"; } \
  2>&1 | tee -a "$PIPELINE_LOG"
