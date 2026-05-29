#!/usr/bin/env bash
# Launch the per-token text->image attention visualizer.
#
# Sets PYTHONPATH / HF_HOME (matching the other launchers) and forwards all
# arguments straight to scripts/analysis/visualize_token_attention.py.
#
# Example:
#   bash src/scripts/run_token_attention.sh \
#     --backend internvl3 \
#     --image /data/datasets/vsr/images/000000000142.jpg \
#     --question "Is the cat to the left of the laptop?" \
#     --checkpoint_dir models/internvl3/stage1/epoch_5 \
#     --lora_checkpoint_dir models/internvl3/gqa_sg/stage2/epoch_1 \
#     --output_dir outputs/token_attn/demo
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/data/Huggingface}"
export PYTHONDONTWRITEBYTECODE=1
export HYDRA_FULL_ERROR=1

python scripts/analysis/visualize_token_attention.py "$@"
