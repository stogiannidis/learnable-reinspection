#!/usr/bin/env bash
# Stage-2 training entrypoint. Defaults to the newest checkpoint from Stage 1.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/train_common.sh"

STAGE2_EXPERIMENT="${REINSPECTION_EXPERIMENT_STAGE2:-s2_qwen}"
if [[ -z "${STAGE1_CHECKPOINT:-}" ]]; then
  if [[ -n "${REINSPECTION_EXPERIMENT:-}" ]]; then
    STAGE1_CHECKPOINT="$(latest_stage1_checkpoint "$REINSPECTION_EXPERIMENT")"
  else
    STAGE1_CHECKPOINT="$(configured_stage1_checkpoint "$STAGE2_EXPERIMENT")"
    [[ -n "$STAGE1_CHECKPOINT" ]] || {
      echo "run_train: ${STAGE2_EXPERIMENT} does not define stage1_checkpoint" >&2
      exit 1
    }
  fi
fi

echo "Using Stage-1 checkpoint: ${STAGE1_CHECKPOINT}"
train "$STAGE2_EXPERIMENT" "stage1_checkpoint=${STAGE1_CHECKPOINT}" "$@" \
  2>&1 | tee "logs/${STAGE2_EXPERIMENT}.log"
