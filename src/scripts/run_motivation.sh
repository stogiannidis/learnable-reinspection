#!/usr/bin/env bash
# Motivation experiment: spatial minimal-pair flip consistency + attention divergence.
#
# All cluster / default Hydra overrides live in this file. Pass extra Hydra args
# after -- or as additional arguments; later keys override earlier ones.
#
# Environment overrides (optional):
#   BACKEND           default internvl3
#   DATA_ROOT         default /data/datasets
#   OUTPUT_FILE       default outputs/motivation/internvl3_motivation_results.json
#   CKPT_DIR          If unset: default InternVL stage-2 path below. If set but empty:
#                     omit checkpoint_dir (frozen base only). If set to a path: use that checkpoint.
#   LORA_CKPT_DIR     Optional split checkpoint for LoRA only.
#   MOTIVATION_LOG    Log file for tee (default: logs/motivation_${BACKEND}.log)
#   MAX_SAMPLES       If set, passed as max_samples=N (-1 = all pairs)
#   MOTIVATION_PROFILE  Optional: gemma4 → BACKEND=gemma4, OUTPUT_FILE for Gemma4,
#                     CKPT_DIR cleared (frozen-only). Any other value is ignored.
#
# Examples (local):
#   bash src/scripts/run_motivation.sh
#   MOTIVATION_PROFILE=gemma4 bash src/scripts/run_motivation.sh
#   bash src/scripts/run_motivation.sh backend=qwen3vl
#   bash src/scripts/run_motivation.sh backend=qwen3vl checkpoint_dir=models/qwen3vl/stage2/epoch_4
#   bash src/scripts/run_motivation.sh max_samples=50
#
# Kubernetes: only run this script from the job workingDir, e.g.
#   command: ["bash", "src/scripts/run_motivation.sh"]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

# Preset profiles (optional). Keeps k8s jobs free of Hydra-style env vars.
if [[ "${MOTIVATION_PROFILE:-}" == "gemma4" ]]; then
  : "${BACKEND:=gemma4}"
  : "${OUTPUT_FILE:=outputs/motivation/gemma4_motivation_results.json}"
  if [[ ! -v CKPT_DIR ]]; then
    CKPT_DIR=""
  fi
fi

# --- Defaults (change only here for cluster / standard runs) ---
# Use -v for CKPT_DIR so an empty value (e.g. frozen Gemma4 in k8s) does not
# fall back to the InternVL checkpoint path.
BACKEND="${BACKEND:-internvl3}"
DATA_ROOT="${DATA_ROOT:-/data/datasets}"
OUTPUT_FILE="${OUTPUT_FILE:-outputs/motivation/internvl3_motivation_results.json}"
if [[ ! -v CKPT_DIR ]]; then
  CKPT_DIR="models/internvl3/focal/stage2/epoch_1"
fi
MOTIVATION_LOG="${MOTIVATION_LOG:-logs/motivation_${BACKEND}.log}"

HYDRA_ARGS=(
  stage=motivation
  "backend=${BACKEND}"
  "data_root=${DATA_ROOT}"
  "output_file=${OUTPUT_FILE}"
)

if [[ -n "${CKPT_DIR}" ]]; then
  HYDRA_ARGS+=("checkpoint_dir=${CKPT_DIR}")
fi
if [[ -n "${LORA_CKPT_DIR:-}" ]]; then
  HYDRA_ARGS+=("lora_checkpoint_dir=${LORA_CKPT_DIR}")
fi
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  HYDRA_ARGS+=("max_samples=${MAX_SAMPLES}")
fi

mkdir -p logs outputs/motivation "$(dirname "${OUTPUT_FILE}")"

python -m src.motivation "${HYDRA_ARGS[@]}" "$@" 2>&1 | tee -a "${MOTIVATION_LOG}"
