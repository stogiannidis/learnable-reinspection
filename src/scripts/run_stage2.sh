#!/usr/bin/env bash
# Stage 2 training (Hydra only). Overrides: key=value after the script name.
# Example: bash run_stage2.sh stage1_checkpoint=models/internvl3/stage1/epoch_5/reinspection_module.pt data_root=/data/datasets
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
deepspeed --module reinspection_vlm.train stage=stage2 "$@"
