#!/usr/bin/env bash
# OGBN-MAG(PV) gap fills for the paper table (NGSF r001 val; LF/NGLF/TC-NGSF r110).
# Even GPU spread: one job per GPU by default (override --gpu-ids / --max-per-gpu).
# Requires: conda env, wandb login (or WANDB_API_KEY). Commands include --use_wandb.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONDA_SH="${CONDA_SH:-/home/yxpeng/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-thgfm}"

MANIFEST_PATH="${SCRIPT_DIR}/manifests/task_pool_ogbn_mag_pv_gaps_s43.jsonl"
SUMMARY_CSV="${SCRIPT_DIR}/summaries/task_pool_ogbn_mag_pv_gaps_s43_best_test_summary.csv"
N_POOL="${N_POOL:-8}"
CPU_THREADS="${CPU_THREADS:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-THGFM-OGBN}"

# Default: all visible GPUs, one task per GPU, max_running = #GPUs.
# Override with space-separated list, e.g. export GPU_IDS="0 1 2 3"
if [[ -n "${GPU_IDS:-}" ]]; then
  read -r -a GPU_IDS <<< "${GPU_IDS}"
elif command -v nvidia-smi >/dev/null 2>&1; then
  mapfile -t GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d '\r' | awk 'NF{print $1}')
  if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
    GPU_IDS=(0 1 2 3)
  fi
else
  GPU_IDS=(0 1 2 3)
fi

MAX_PER_GPU="${MAX_PER_GPU:-1}"
MAX_RUNNING="${MAX_RUNNING:-${#GPU_IDS[@]}}"

usage() {
  cat <<EOF
Usage: $0 [options]

Environment overrides:
  CONDA_SH CONDA_ENV N_POOL CPU_THREADS WANDB_PROJECT WANDB_ENTITY WANDB_MODE
  GPU_IDS="0 1 2 3"   (space-separated string; default: all GPUs from nvidia-smi)
  MAX_PER_GPU MAX_RUNNING

Options (passed to this script):
  --dry-run
EOF
}

DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown: $1" >&2; usage >&2; exit 2 ;;
  esac
done

mkdir -p "${SCRIPT_DIR}/manifests" "${SCRIPT_DIR}/summaries" "${PROJECT_DIR}/logs"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${PROJECT_DIR}"

GEN_CMD=(
  python "./trainers/main/build_ogbn_mag_pv_gap_manifest.py"
  --output "${MANIFEST_PATH}"
  --n-pool "${N_POOL}"
  --cpu-threads "${CPU_THREADS}"
  --wandb-project "${WANDB_PROJECT}"
)
[[ -n "${WANDB_ENTITY:-}" ]] && GEN_CMD+=(--wandb-entity "${WANDB_ENTITY}")
[[ -n "${WANDB_MODE:-}" ]] && GEN_CMD+=(--wandb-mode "${WANDB_MODE}")

POOL_CMD=(
  python "./trainers/main/run_manifest_subprocess_pool.py"
  --manifest "${MANIFEST_PATH}"
  --gpu-ids "${GPU_IDS[@]}"
  --max-running "${MAX_RUNNING}"
  --max-per-gpu "${MAX_PER_GPU}"
)

COLLECT_CMD=(
  python "./trainers/main/collect_task_pool_best_test.py"
  --manifest "${MANIFEST_PATH}"
  --output "${SUMMARY_CSV}"
)

printf '[gap-pool] manifest: %q ' "${GEN_CMD[@]}"; printf '\n'
printf '[gap-pool] pool: %q ' "${POOL_CMD[@]}"; printf '\n'
printf '[gap-pool] collect: %q ' "${COLLECT_CMD[@]}"; printf '\n'

if [[ "${DRY_RUN}" -eq 1 ]]; then
  "${GEN_CMD[@]}"
  POOL_CMD+=(--dry-run)
  "${POOL_CMD[@]}"
  exit 0
fi

"${GEN_CMD[@]}"
"${COLLECT_CMD[@]}" --include-incomplete || true
"${POOL_CMD[@]}"
code=$?
"${COLLECT_CMD[@]}" || true
exit "${code}"
