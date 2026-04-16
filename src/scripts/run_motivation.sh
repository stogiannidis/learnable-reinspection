#!/usr/bin/env bash
# Motivation experiment: spatial minimal-pair flip consistency + attention divergence.
#
# Base VLM only (demonstrates the problem):
#   bash src/scripts/run_motivation.sh backend=qwen3vl
#   bash src/scripts/run_motivation.sh backend=internvl3
#
# Base + Re-Inspection (demonstrates the fix):
#   bash src/scripts/run_motivation.sh backend=qwen3vl checkpoint_dir=models/qwen3vl/stage2/epoch_10
#
# Quick test (first 50 pairs):
#   bash src/scripts/run_motivation.sh backend=qwen3vl max_samples=50
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
python -m src.motivation stage=motivation "$@"
