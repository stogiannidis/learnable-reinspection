#!/usr/bin/env bash
# Single entrypoint for all training runs.
# Edit the configuration block below to change what is trained, then:
#   bash src/scripts/run_train.sh
#
# Full pipeline (Stage 1 → Stage 2 in one job):
#   Set REINSPECTION_EXPERIMENT_STAGE2 to a Stage-2 experiment name. The script
#   runs Stage-1, finds the latest reinspection_module.pt under
#   models/…/stage1/epoch_*/ and passes it to Stage-2 automatically.
#   Leave REINSPECTION_EXPERIMENT_STAGE2 empty to run a single experiment only.
#
# Kubernetes: set REINSPECTION_EXPERIMENT (and optionally _STAGE2) as inline
#   env before calling: REINSPECTION_EXPERIMENT=s1_gemma4 bash src/scripts/run_train.sh
set -euo pipefail
set -o pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

# ── Configuration ─────────────────────────────────────────────────────────────
# Stage-1 experiment name (src/configs/experiment/<name>.yaml).
REINSPECTION_EXPERIMENT="${REINSPECTION_EXPERIMENT:-s1_qwen}"
# Stage-2 experiment name. Empty = run Stage-1 only.
REINSPECTION_EXPERIMENT_STAGE2="${REINSPECTION_EXPERIMENT_STAGE2:-s2_qwen}"
# Where checkpoints are saved (matches output_dir in Hydra config).
OUTPUT_DIR=models
# System
export OMP_NUM_THREADS=4
export HYDRA_FULL_ERROR=1
export HF_HOME=/data/Huggingface
# ──────────────────────────────────────────────────────────────────────────────

mkdir -p logs

# Discover the highest epoch_*/reinspection_module.pt saved by Stage-1.
# Reads backend and experiment_name from the Stage-1 experiment YAML to
# reconstruct the trainer's checkpoint path layout.
discover_stage1_checkpoint() {
  local s1_name="$1"
  local out_root="$2"
  local expf="src/configs/experiment/${s1_name}.yaml"
  [[ -f "$expf" ]] || { echo "run_train: missing ${expf}" >&2; return 1; }
  local backend
  backend=$(sed -n 's/^.*override \/backend:[[:space:]]*\([^#[:space:]]*\).*/\1/p' "$expf" | head -1)
  [[ -n "$backend" ]] || { echo "run_train: cannot parse backend from ${expf}" >&2; return 1; }
  local expn
  expn=$(sed -n 's/^experiment_name:[[:space:]]*\([^#[:space:]"]*\).*/\1/p' "$expf" | head -1)
  local base="${out_root}/${backend}"
  [[ -n "$expn" && "$expn" != "null" ]] && base="${base}/${expn}"
  base="${base}/stage1"
  local best="" best_n=-1 d n
  shopt -s nullglob
  for d in "${base}"/epoch_*; do
    [[ -d "$d" ]] || continue
    n="${d##*epoch_}"; n="${n//[^0-9]/}"
    [[ -n "$n" ]] || continue
    (( 10#${n} > best_n )) && { best_n=$((10#$n)); best="${d}/reinspection_module.pt"; }
  done
  shopt -u nullglob
  [[ -f "$best" ]] || { echo "run_train: no checkpoint found under ${base}" >&2; return 1; }
  printf '%s' "$best"
}

if [[ -n "$REINSPECTION_EXPERIMENT_STAGE2" ]]; then
  e1="$REINSPECTION_EXPERIMENT"
  e2="$REINSPECTION_EXPERIMENT_STAGE2"
  plog="logs/pipeline_${e1}__${e2}.log"
  { echo "=== Stage-1: +experiment=${e1} ==="
    deepspeed --module src.train "+experiment=${e1}" "$@"
  } 2>&1 | tee -a "$plog" "logs/${e1}.log"
  ckpt=$(discover_stage1_checkpoint "$e1" "$OUTPUT_DIR")
  { echo "=== Stage-2: +experiment=${e2} | stage1_checkpoint=${ckpt} ==="
    deepspeed --module src.train "+experiment=${e2}" "stage1_checkpoint=${ckpt}" "$@"
  } 2>&1 | tee -a "$plog" "logs/${e2}.log"
else
  deepspeed --module src.train "+experiment=${REINSPECTION_EXPERIMENT}" "$@" \
    2>&1 | tee "logs/${REINSPECTION_EXPERIMENT}.log"
fi
