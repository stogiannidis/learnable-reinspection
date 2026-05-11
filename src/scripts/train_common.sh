#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

OUTPUT_DIR="${OUTPUT_DIR:-models}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export HF_HOME="${HF_HOME:-/data/Huggingface}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p logs

train() {
  local experiment="$1"
  shift
  deepspeed --module src.train "+experiment=${experiment}" "$@"
}

latest_stage1_checkpoint() {
  local name="$1"
  local config="src/configs/experiment/${name}.yaml"
  [[ -f "$config" ]] || { echo "run_train: missing ${config}" >&2; return 1; }

  local backend experiment base best="" best_epoch=-1 dir epoch
  backend=$(awk '/override \/backend:/ {sub(/#.*/, ""); print $NF; exit}' "$config" | tr -d "\"'")
  experiment=$(awk '$1 == "experiment_name:" {sub(/#.*/, ""); print $2; exit}' "$config" | tr -d "\"'")
  [[ -n "$backend" ]] || { echo "run_train: cannot parse backend from ${config}" >&2; return 1; }

  base="${OUTPUT_DIR}/${backend}"
  [[ -n "$experiment" && "$experiment" != "null" ]] && base="${base}/${experiment}"
  base="${base}/stage1"

  shopt -s nullglob
  for dir in "${base}"/epoch_*; do
    [[ -f "${dir}/reinspection_module.pt" ]] || continue
    epoch="${dir##*/epoch_}"
    [[ "$epoch" =~ ^[0-9]+$ ]] || continue
    if (( 10#$epoch > best_epoch )); then
      best_epoch=$((10#$epoch))
      best="${dir}/reinspection_module.pt"
    fi
  done
  shopt -u nullglob

  [[ -n "$best" ]] || { echo "run_train: no checkpoint found under ${base}" >&2; return 1; }
  printf '%s' "$best"
}
