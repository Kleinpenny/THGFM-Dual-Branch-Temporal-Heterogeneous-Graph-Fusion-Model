#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONDA_SH="${CONDA_SH:-${HOME}/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-thgfm}"

POOL_KIND="all"
MAX_RUNNING=12
POLL_SECONDS=20
SEEDS=(43)
GPU_IDS=(0 1 2 3)
MAX_PER_GPU=4
N_POOL=8
CPU_THREADS=1
MAX_LOAD1=75
MAX_GPU_UTIL=0
MANIFEST_PATH=""
DRY_RUN=0
SUMMARY_CSV_PATH=""
MANIFEST_DIR="${SCRIPT_DIR}/manifests"
SUMMARY_DIR="${SCRIPT_DIR}/summaries"

usage() {
  cat <<'EOF'
Usage:
  ./trainers/main/start_task_pool.sh [options]

Options:
  --pool-kind {ad|all|ta|missing} Which manifest to generate and run. Default: all
                                  ad = AD family, R111/S001 full set
                                  all = all 4 families, R111/S001 + R001/S001 full set
                                  ta  = all 4 families, TA-NGLF/TA-NGSF only (R111+R001)
                                  missing = only missing cells for table appendix-detailed-ablation-drsf-r111-s001
  --manifest PATH               Use a pre-generated manifest instead of auto-generating one
  --max-running N               Max active parent processes to keep running. Default: 12
  --poll-seconds N              Poll interval for checking finished jobs. Default: 20
  --max-load1 N                 Pause launching when 1-min system load exceeds N. Default: 75
  --max-gpu-util N              Pause launching when any target GPU util(%) exceeds N. Default: 0 (disabled)
  --seeds S1 [S2 ...]           Seeds used when auto-generating a manifest. Default: 43
  --n-pool N                    Per-task sampling worker count. Default: 8
  --cpu-threads N               Per-task BLAS/OpenMP/Torch thread cap. Default: 1
  --gpu-ids G1 [G2 ...]         GPU IDs to schedule on. Default: 0 1 2 3
  --max-per-gpu N               Max concurrent tasks per GPU. Default: 4
  --dry-run                     Only print the generated commands
  --summary-csv PATH            CSV output path for best_test(H/L) aggregation
  -h, --help                    Show this help message

Examples:
  ./trainers/main/start_task_pool.sh
  ./trainers/main/start_task_pool.sh --pool-kind ad --max-running 12 --max-load1 75 --max-gpu-util 0
  ./trainers/main/start_task_pool.sh --seeds 43 44 45 --n-pool 8 --cpu-threads 1 --gpu-ids 0 1 2 3 --max-per-gpu 4 --max-load1 75 --max-gpu-util 0
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pool-kind)
      POOL_KIND="$2"
      shift 2
      ;;
    --manifest)
      MANIFEST_PATH="$2"
      shift 2
      ;;
    --max-running)
      MAX_RUNNING="$2"
      shift 2
      ;;
    --poll-seconds)
      POLL_SECONDS="$2"
      shift 2
      ;;
    --max-load1)
      MAX_LOAD1="$2"
      shift 2
      ;;
    --max-gpu-util)
      MAX_GPU_UTIL="$2"
      shift 2
      ;;
    --seeds)
      shift
      SEEDS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        SEEDS+=("$1")
        shift
      done
      ;;
    --n-pool)
      N_POOL="$2"
      shift 2
      ;;
    --cpu-threads)
      CPU_THREADS="$2"
      shift 2
      ;;
    --gpu-ids)
      shift
      GPU_IDS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        GPU_IDS+=("$1")
        shift
      done
      ;;
    --max-per-gpu)
      MAX_PER_GPU="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --summary-csv)
      SUMMARY_CSV_PATH="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${POOL_KIND}" != "ad" && "${POOL_KIND}" != "all" && "${POOL_KIND}" != "ta" && "${POOL_KIND}" != "missing" ]]; then
  echo "--pool-kind must be 'ad', 'all', 'ta', or 'missing'" >&2
  exit 2
fi

if [[ "${#SEEDS[@]}" -eq 0 ]]; then
  echo "--seeds requires at least one value" >&2
  exit 2
fi

if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
  echo "--gpu-ids requires at least one value" >&2
  exit 2
fi
if [[ "${N_POOL}" -le 0 ]]; then
  echo "--n-pool must be positive" >&2
  exit 2
fi
if [[ "${CPU_THREADS}" -le 0 ]]; then
  echo "--cpu-threads must be positive" >&2
  exit 2
fi
if ! [[ "${MAX_LOAD1}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "--max-load1 must be a non-negative number" >&2
  exit 2
fi
if ! [[ "${MAX_GPU_UTIL}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "--max-gpu-util must be a non-negative number" >&2
  exit 2
fi

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${PROJECT_DIR}"
mkdir -p "${MANIFEST_DIR}" "${SUMMARY_DIR}"

if [[ -z "${MANIFEST_PATH}" ]]; then
  if [[ "${POOL_KIND}" == "ad" ]]; then
    MANIFEST_PATH="${MANIFEST_DIR}/task_pool.jsonl"
    FAMILIES=(ad)
    ABLATION_SET_ARG=()
    MANIFEST_PROFILE_ARG=()
  elif [[ "${POOL_KIND}" == "ta" ]]; then
    MANIFEST_PATH="${MANIFEST_DIR}/task_pool_ta.jsonl"
    FAMILIES=(ad pf-l1 pf-l2 pv)
    ABLATION_SET_ARG=(--ablation-set ta)
    MANIFEST_PROFILE_ARG=()
  elif [[ "${POOL_KIND}" == "missing" ]]; then
    MANIFEST_PATH="${MANIFEST_DIR}/task_pool_drsf_r111_table_missing.jsonl"
    FAMILIES=(ad)
    ABLATION_SET_ARG=()
    MANIFEST_PROFILE_ARG=(--manifest-profile drsf-r111-table-missing)
  else
    MANIFEST_PATH="${MANIFEST_DIR}/task_pool_all.jsonl"
    FAMILIES=(ad pf-l1 pf-l2 pv)
    ABLATION_SET_ARG=()
    MANIFEST_PROFILE_ARG=()
  fi

  GENERATE_CMD=(
    python "./trainers/main/build_task_pool_manifest.py"
    --families "${FAMILIES[@]}"
    --output "${MANIFEST_PATH}"
    --seeds "${SEEDS[@]}"
    --n-pool "${N_POOL}"
    --cpu-threads "${CPU_THREADS}"
    "${ABLATION_SET_ARG[@]}"
    "${MANIFEST_PROFILE_ARG[@]}"
  )

  echo "[start] generating manifest: ${MANIFEST_PATH}"
  printf '[start] manifest cmd: %q ' "${GENERATE_CMD[@]}"
  printf '\n'
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    "${GENERATE_CMD[@]}"
  fi
else
  if [[ "${MANIFEST_PATH}" != /* ]]; then
    MANIFEST_PATH="${PROJECT_DIR}/${MANIFEST_PATH}"
  fi
fi

if [[ -z "${SUMMARY_CSV_PATH}" ]]; then
  manifest_basename="$(basename "${MANIFEST_PATH}")"
  manifest_stem="${manifest_basename%.*}"
  SUMMARY_CSV_PATH="${SUMMARY_DIR}/${manifest_stem}_best_test_summary.csv"
elif [[ "${SUMMARY_CSV_PATH}" != /* ]]; then
  SUMMARY_CSV_PATH="${PROJECT_DIR}/${SUMMARY_CSV_PATH}"
fi

COLLECT_CMD=(
  python "./trainers/main/collect_task_pool_best_test.py"
  --manifest "${MANIFEST_PATH}"
  --output "${SUMMARY_CSV_PATH}"
)

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

if [[ "${DRY_RUN}" -eq 1 ]]; then
  POOL_CMD+=(--dry-run)
fi

echo "[start] pool kind: ${POOL_KIND}"
echo "[start] ablation scope: Fusion variants (SF/LF/CGSF/CGLF/NGSF/NGLF/TC-NGSF/TC-NGLF/TA-NGLF/TA-NGSF) x configs (R111/S001, R001/S001)"
echo "[start] manifest: ${MANIFEST_PATH}"
echo "[start] max_running: ${MAX_RUNNING}"
echo "[start] poll_seconds: ${POLL_SECONDS}"
echo "[start] max_load1: ${MAX_LOAD1}"
echo "[start] max_gpu_util: ${MAX_GPU_UTIL}"
echo "[start] seeds: ${SEEDS[*]}"
echo "[start] n_pool: ${N_POOL}"
echo "[start] cpu_threads: ${CPU_THREADS}"
echo "[start] gpu_ids: ${GPU_IDS[*]}"
echo "[start] max_per_gpu: ${MAX_PER_GPU}"
echo "[start] summary_csv: ${SUMMARY_CSV_PATH}"
printf '[start] pool cmd: %q ' "${POOL_CMD[@]}"
printf '\n'
printf '[start] collect cmd: %q ' "${COLLECT_CMD[@]}"
printf '\n'

if [[ "${DRY_RUN}" -eq 0 ]]; then
  echo "[start] collect existing completed best_test(H/L) before launch"
  "${COLLECT_CMD[@]}"
fi

set +e
"${POOL_CMD[@]}"
POOL_EXIT_CODE=$?
set -e

if [[ "${DRY_RUN}" -eq 0 ]]; then
  echo "[start] collect best_test(H/L) after pool run"
  "${COLLECT_CMD[@]}"
fi

exit "${POOL_EXIT_CODE}"
