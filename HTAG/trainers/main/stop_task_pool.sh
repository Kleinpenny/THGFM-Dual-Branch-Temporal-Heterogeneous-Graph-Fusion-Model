#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="${SCRIPT_DIR}/runtime"
PID_FILE="${RUNTIME_DIR}/task_pool.pid"
LEGACY_PID_FILE="${RUNTIME_DIR}/task_pool_scheduler.pid"
DRY_RUN=0
KILL_CHILDREN=0
STOP_UNITS=1

usage() {
  cat <<'EOF'
Usage:
  ./trainers/main/stop_task_pool.sh [options]

Options:
  --kill-children      Also kill leftover python trainer children by pattern
  --skip-units         Skip systemd stop/reset of task units
  --dry-run            Print actions only
  -h, --help           Show this help
EOF
}

run_cmd() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '[dry-run] %q ' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

stop_scheduler_pid_file() {
  local pf="$1"
  if [[ ! -f "${pf}" ]]; then
    return 0
  fi
  local pid
  pid="$(cat "${pf}" || true)"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    echo "[stop] terminating scheduler pid=${pid} (${pf})"
    run_cmd kill "${pid}" || true
    sleep 1
    if kill -0 "${pid}" 2>/dev/null; then
      echo "[stop] force killing scheduler pid=${pid}"
      run_cmd kill -9 "${pid}" || true
    fi
  else
    echo "[stop] pid file ${pf} exists but process already stopped"
  fi
  [[ "${DRY_RUN}" -eq 0 ]] && rm -f "${pf}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kill-children) KILL_CHILDREN=1; shift ;;
    --skip-units) STOP_UNITS=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

stop_scheduler_pid_file "${PID_FILE}"
stop_scheduler_pid_file "${LEGACY_PID_FILE}"

if [[ "${STOP_UNITS}" -eq 1 ]]; then
  PATTERNS=("thgfm-pv-dblp-*" "thgfm-pv-arxiv-*")
  for pattern in "${PATTERNS[@]}"; do
    mapfile -t units < <(systemctl list-units --all --type=service --no-legend "${pattern}" | awk '{print $1}')
    if [[ "${#units[@]}" -eq 0 ]]; then
      echo "[stop] no units matched ${pattern}"
      continue
    fi
    echo "[stop] matched ${#units[@]} units for ${pattern}"
    for unit in "${units[@]}"; do
      echo "  - ${unit}"
      run_cmd sudo systemctl stop "${unit}" || true
      run_cmd sudo systemctl reset-failed "${unit}" || true
    done
  done
fi

if [[ "${KILL_CHILDREN}" -eq 1 ]]; then
  patterns=(
    "Arxiv_PV/train_paper_venue_shared_task_pool.py"
    "DBLP_PV/train_paper_venue_shared_task_pool.py"
    "train_paper_venue_shared.py --task_name PV_HTAG_"
  )
  for pattern in "${patterns[@]}"; do
    echo "[kill] pkill -f ${pattern}"
    run_cmd pkill -f "${pattern}" || true
  done
fi

echo "[stop] remaining HTAG units snapshot:"
run_cmd systemctl list-units --all "thgfm-pv-dblp-*" "thgfm-pv-arxiv-*"
