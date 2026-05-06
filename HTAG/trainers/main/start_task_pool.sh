#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONDA_SH="${CONDA_SH:-${HOME}/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-thgfm}"

MAX_RUNNING=8
POLL_SECONDS=20
SEEDS=(43 44 45)
GPU_IDS=(0 1 2 3)
MAX_PER_GPU=2
N_POOL=8
CPU_THREADS=1
MAX_LOAD1=75
MAX_GPU_UTIL=0
MANIFEST_PATH=""
DRY_RUN=0
BACKGROUND=1

DATASETS=(dblp arxiv)
CONVS=(rtta ssta drsf drsf_tcngsf drsf_tcnglf)

MANIFEST_DIR="${SCRIPT_DIR}/manifests"
RUNTIME_DIR="${SCRIPT_DIR}/runtime"
PID_FILE="${RUNTIME_DIR}/task_pool.pid"
LEGACY_PID_FILE="${RUNTIME_DIR}/task_pool_scheduler.pid"
SCHEDULER_LOG="${RUNTIME_DIR}/task_pool_scheduler.log"

usage() {
  cat <<'EOF'
Usage:
  ./trainers/main/start_task_pool.sh [options]

Options:
  --manifest PATH               Use an existing manifest (skip auto-generate)
  --datasets D1 [D2 ...]        dblp / arxiv for auto-generate (default: both)
  --convs C1 [C2 ...]           Conv list for auto-generate
  --seeds S1 [S2 ...]           Seeds for auto-generate
  --max-running N               Max active tasks (default: 8)
  --poll-seconds N              Poll interval (default: 20)
  --n-pool N                    Per-task sampling workers (default: 8)
  --cpu-threads N               Per-task BLAS/OpenMP thread cap (default: 1)
  --gpu-ids G1 [G2 ...]         GPU IDs for scheduling (default: 0 1 2 3)
  --max-per-gpu N               Max concurrent tasks per GPU (default: 2)
  --max-load1 N                 Pause launches if load1 > N (default: 75)
  --max-gpu-util N              Pause launches if any gpu util% > N (default: 0, disabled)
  --foreground                  Run scheduler in foreground (default: background)
  --dry-run                     Print commands without running
  -h, --help                    Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) MANIFEST_PATH="$2"; shift 2 ;;
    --datasets)
      shift
      DATASETS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do DATASETS+=("$1"); shift; done
      ;;
    --convs)
      shift
      CONVS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do CONVS+=("$1"); shift; done
      ;;
    --seeds)
      shift
      SEEDS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do SEEDS+=("$1"); shift; done
      ;;
    --max-running) MAX_RUNNING="$2"; shift 2 ;;
    --poll-seconds) POLL_SECONDS="$2"; shift 2 ;;
    --n-pool) N_POOL="$2"; shift 2 ;;
    --cpu-threads) CPU_THREADS="$2"; shift 2 ;;
    --gpu-ids)
      shift
      GPU_IDS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do GPU_IDS+=("$1"); shift; done
      ;;
    --max-per-gpu) MAX_PER_GPU="$2"; shift 2 ;;
    --max-load1) MAX_LOAD1="$2"; shift 2 ;;
    --max-gpu-util) MAX_GPU_UTIL="$2"; shift 2 ;;
    --foreground) BACKGROUND=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "${#SEEDS[@]}" -eq 0 || "${#GPU_IDS[@]}" -eq 0 || "${#DATASETS[@]}" -eq 0 || "${#CONVS[@]}" -eq 0 ]]; then
  echo "--seeds/--gpu-ids/--datasets/--convs require at least one value" >&2
  exit 2
fi

mkdir -p "${MANIFEST_DIR}" "${RUNTIME_DIR}"

for pf in "${PID_FILE}" "${LEGACY_PID_FILE}"; do
  if [[ -f "${pf}" ]]; then
    old_pid="$(cat "${pf}" || true)"
    if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
      echo "[start] scheduler already running with pid=${old_pid} (${pf}). Stop it first." >&2
      exit 1
    fi
    rm -f "${pf}"
  fi
done

if [[ -z "${MANIFEST_PATH}" ]]; then
  ts="$(date +%Y%m%d_%H%M%S)"
  MANIFEST_PATH="${MANIFEST_DIR}/task_pool_htag_conv_ablation_${ts}.jsonl"
  GEN_CMD=(
    python "./trainers/main/build_task_pool_manifest.py"
    --datasets "${DATASETS[@]}"
    --convs "${CONVS[@]}"
    --seeds "${SEEDS[@]}"
    --output "${MANIFEST_PATH}"
    --n-pool "${N_POOL}"
    --cpu-threads "${CPU_THREADS}"
  )
else
  if [[ "${MANIFEST_PATH}" != /* ]]; then
    MANIFEST_PATH="${PROJECT_DIR}/${MANIFEST_PATH}"
  fi
  GEN_CMD=()
fi

POOL_CMD=(
  python "./trainers/main/run_systemd_task_pool.py"
  --manifest "${MANIFEST_PATH}"
  --max-running "${MAX_RUNNING}"
  --poll-seconds "${POLL_SECONDS}"
  --max-load1 "${MAX_LOAD1}"
  --max-gpu-util "${MAX_GPU_UTIL}"
  --gpu-ids "${GPU_IDS[@]}"
  --max-per-gpu "${MAX_PER_GPU}"
)
[[ "${DRY_RUN}" -eq 1 ]] && POOL_CMD+=(--dry-run)

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${PROJECT_DIR}"

if [[ "${#GEN_CMD[@]}" -gt 0 ]]; then
  printf '[start] manifest cmd: %q ' "${GEN_CMD[@]}"; printf '\n'
  [[ "${DRY_RUN}" -eq 0 ]] && "${GEN_CMD[@]}"
fi

printf '[start] pool cmd: %q ' "${POOL_CMD[@]}"; printf '\n'

if [[ "${DRY_RUN}" -eq 1 ]]; then
  exit 0
fi

if [[ "${BACKGROUND}" -eq 0 ]]; then
  echo "[start] running in foreground"
  "${POOL_CMD[@]}"
  exit $?
fi

echo "[start] running in background"
nohup bash -lc "
  set -euo pipefail
  source \"${CONDA_SH}\"
  conda activate \"${CONDA_ENV}\"
  cd \"${PROJECT_DIR}\"
  $(printf '%q ' "${POOL_CMD[@]}")
" >> "${SCHEDULER_LOG}" 2>&1 &

bg_pid=$!
echo "${bg_pid}" > "${PID_FILE}"
echo "[start] scheduler pid=${bg_pid}"
echo "[start] pid file: ${PID_FILE}"
echo "[start] log file: ${SCHEDULER_LOG}"
