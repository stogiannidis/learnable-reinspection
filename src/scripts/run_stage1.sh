#!/usr/bin/env bash
# Stage 1 training (Hydra only). Overrides: key=value after the script name.
# Examples:
#   bash run_stage1.sh data_root=/data/datasets
#   bash run_stage1.sh backend=qwen3vl wandb_run_name=s1-qwen
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
deepspeed --module src.train stage=stage1 "$@"
