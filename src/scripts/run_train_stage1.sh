#!/usr/bin/env bash
# Stage-1 training entrypoint.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/train_common.sh"

EXPERIMENT="${REINSPECTION_EXPERIMENT:-s1_qwen}"

train "$EXPERIMENT" "$@" 2>&1 | tee "logs/${EXPERIMENT}.log"
