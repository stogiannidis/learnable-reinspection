#!/usr/bin/env bash
set -euo pipefail

# Runs stage1 -> waits -> runs stage2 -> waits.
# By default, deletes each completed Job (garbage collection).
#
# Notes on aliases:
# - If you have an alias/function like `k` for kubectl in your shell config,
#   this script will try to use it after sourcing ~/.bashrc.
#
# Usage examples:
#   bash k8s/run_sequential.sh
#   NAMESPACE=eidf098ns TIMEOUT_STAGE1=48h TIMEOUT_STAGE2=72h bash k8s/run_sequential.sh
#   KEEP_JOBS=1 bash k8s/run_sequential.sh
#

shopt -s expand_aliases || true
[[ -f "${HOME}/.bashrc" ]] && source "${HOME}/.bashrc" || true

NS="${NAMESPACE:-eidf098ns}"
TIMEOUT_STAGE1="${TIMEOUT_STAGE1:-48h}"
TIMEOUT_STAGE2="${TIMEOUT_STAGE2:-72h}"
FOLLOW_LOGS="${FOLLOW_LOGS:-1}"
KEEP_JOBS="${KEEP_JOBS:-0}"
STACK="${STACK:-qwen}"

KUBECTL="${KUBECTL:-}"
if [[ -z "$KUBECTL" ]]; then
  if type -t k >/dev/null 2>&1; then
    KUBECTL="k"
  else
    KUBECTL="kubectl"
  fi
fi

die() {
  echo "ERROR: $*" >&2
  exit 1
}

need() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

need "$KUBECTL"

echo "Using kubectl command: ${KUBECTL}"
echo "Namespace: ${NS}"
echo "Stack: ${STACK}"

if ! ${KUBECTL} get secret reinspection-secrets -n "${NS}" >/dev/null 2>&1; then
  die "Secret 'reinspection-secrets' not found in namespace ${NS}. Create it first (see k8s/secrets.example.yaml)."
fi

case "${STACK}" in
  qwen)
    STAGE1_MANIFEST="k8s/stage1.yaml"
    STAGE2_MANIFEST="k8s/stage2.yaml"
    ;;
  internvl)
    STAGE1_MANIFEST="k8s/internvl_stage1.yaml"
    STAGE2_MANIFEST="k8s/internvl_stage2.yaml"
    ;;
  *)
    die "Unsupported STACK=${STACK}. Use qwen or internvl."
    ;;
esac

create_job() {
  local manifest="$1"
  ${KUBECTL} create -f "${manifest}" -n "${NS}" -o jsonpath='{.metadata.name}'
  echo
}

follow_logs_bg() {
  local job_name="$1"
  if [[ "${FOLLOW_LOGS}" != "1" ]]; then
    echo ""
    return 0
  fi
  # Follow logs in background; errors are ignored (pod may not exist yet).
  (${KUBECTL} logs -n "${NS}" -f "job/${job_name}" || true) &
  echo $!
}

wait_job() {
  local job_name="$1"
  local timeout="$2"
  ${KUBECTL} wait -n "${NS}" --for=condition=complete "job/${job_name}" --timeout="${timeout}"
}

gc_job() {
  local job_name="$1"
  if [[ "${KEEP_JOBS}" == "1" ]]; then
    return 0
  fi
  ${KUBECTL} delete -n "${NS}" "job/${job_name}" >/dev/null
}

echo "Creating ${STACK} Stage 1 job..."
JOB1="$(create_job "${STAGE1_MANIFEST}")"
echo "Stage 1 job: ${JOB1}"
LOGPID1="$(follow_logs_bg "${JOB1}")"
set +e
wait_job "${JOB1}" "${TIMEOUT_STAGE1}"
RC1=$?
set -e
[[ -n "${LOGPID1}" ]] && kill "${LOGPID1}" >/dev/null 2>&1 || true
[[ "${RC1}" -eq 0 ]] || die "Stage 1 failed or timed out (job=${JOB1})"
echo "Stage 1 complete."
gc_job "${JOB1}"

echo "Creating ${STACK} Stage 2 job..."
JOB2="$(create_job "${STAGE2_MANIFEST}")"
echo "Stage 2 job: ${JOB2}"
LOGPID2="$(follow_logs_bg "${JOB2}")"
set +e
wait_job "${JOB2}" "${TIMEOUT_STAGE2}"
RC2=$?
set -e
[[ -n "${LOGPID2}" ]] && kill "${LOGPID2}" >/dev/null 2>&1 || true
[[ "${RC2}" -eq 0 ]] || die "Stage 2 failed or timed out (job=${JOB2})"
echo "Stage 2 complete."
gc_job "${JOB2}"

echo "Done."
