#!/usr/bin/env bash
# Evaluation (Hydra only; default stage group is eval). Overrides: key=value.
# Example: bash run_eval.sh data_root=/data/datasets checkpoint_dir=models/internvl3/stage2/epoch_4 eval_compare=true
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
python -m reinspection_vlm.evaluate stage=eval "$@"
