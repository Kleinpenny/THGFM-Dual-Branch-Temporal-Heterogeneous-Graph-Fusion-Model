#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
USER_NAME="$(id -un)"

THRESHOLD=3
INTERVAL_SECONDS=60
COOLDOWN_SECONDS=300
MAX_RUNNING=15
DRY_RUN=0
ONCE=0

usage() {
  cat <<'EOF'
Usage:
  ./trainers/main/watch_task_pool_nohup.sh [options]

Options:
  --threshold N           Launch the pool when GPU python processes are below this number. Default: 3
  --interval-seconds N    Check interval. Default: 60
  --cooldown-seconds N    Minimum gap between relaunches. Default: 300
  --max-running N         Passed to start_task_pool.sh. Default: 15
  --once                  Run one check and exit
  --dry-run               Print actions without executing them
  -h, --help              Show this help message

Recommended background usage:
  nohup ./trainers/main/watch_task_pool_nohup.sh \
    > ./logs/nohup_task_pool_guard.out 2>&1 < /dev/null &
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --threshold)
      THRESHOLD="$2"
      shift 2
      ;;
    --interval-seconds)
      INTERVAL_SECONDS="$2"
      shift 2
      ;;
    --cooldown-seconds)
      COOLDOWN_SECONDS="$2"
      shift 2
      ;;
    --max-running)
      MAX_RUNNING="$2"
      shift 2
      ;;
    --once)
      ONCE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
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

if [[ "${THRESHOLD}" -lt 0 || "${INTERVAL_SECONDS}" -le 0 || "${COOLDOWN_SECONDS}" -lt 0 || "${MAX_RUNNING}" -le 0 ]]; then
  echo "threshold must be >= 0, interval/max-running must be > 0, cooldown must be >= 0" >&2
  exit 2
fi

cd "${PROJECT_DIR}"
mkdir -p "./logs"

timestamp() {
  date "+%F %T"
}

count_user_gpu_python_procs() {
  local count=0
  local raw_lines
  raw_lines="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>/dev/null || true)"

  while IFS=',' read -r pid proc_name; do
    pid="${pid//[[:space:]]/}"
    proc_name="${proc_name#"${proc_name%%[![:space:]]*}"}"
    if [[ -z "${pid}" || -z "${proc_name}" ]]; then
      continue
    fi
    if [[ "${proc_name}" != *python* ]]; then
      continue
    fi

    local owner
    owner="$(ps -o user= -p "${pid}" 2>/dev/null | tr -d '[:space:]' || true)"
    if [[ "${owner}" == "${USER_NAME}" ]]; then
      count=$((count + 1))
    fi
  done <<< "${raw_lines}"

  echo "${count}"
}

pool_manager_running() {
  pgrep -u "${USER_NAME}" -f "trainers/main/start_task_pool.sh|trainers/main/run_systemd_task_pool.py" >/dev/null
}

launch_pool() {
  local launch_cmd=(
    nohup ./trainers/main/start_task_pool.sh --max-running "${MAX_RUNNING}"
  )

  echo "[$(timestamp)] launching task pool"
  printf '[%s] command: ' "$(timestamp)"
  printf '%q ' "${launch_cmd[@]}"
  printf '> %q 2>&1 < /dev/null &\n' "./logs/nohup_task_pool.out"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi

  nohup ./trainers/main/start_task_pool.sh --max-running "${MAX_RUNNING}" \
    > ./logs/nohup_task_pool.out 2>&1 < /dev/null &
  echo "[$(timestamp)] launched pid=$!"
}

last_launch_epoch=0

while true; do
  current_count="$(count_user_gpu_python_procs)"
  echo "[$(timestamp)] gpu_python_processes=${current_count} threshold=${THRESHOLD}"

  if [[ "${current_count}" -lt "${THRESHOLD}" ]]; then
    if pool_manager_running; then
      echo "[$(timestamp)] pool manager already running, skip launch"
    else
      now_epoch="$(date +%s)"
      if (( now_epoch - last_launch_epoch < COOLDOWN_SECONDS )); then
        echo "[$(timestamp)] cooldown active, skip launch"
      else
        launch_pool
        last_launch_epoch="${now_epoch}"
      fi
    fi
  fi

  if [[ "${ONCE}" -eq 1 ]]; then
    break
  fi

  sleep "${INTERVAL_SECONDS}"
done
