#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="${SCRIPT_DIR}/runtime"
PID_FILE="${RUNTIME_DIR}/task_pool.pid"
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

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kill-children) KILL_CHILDREN=1; shift ;;
    --skip-units) STOP_UNITS=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -f "${PID_FILE}" ]]; then
  pid="$(cat "${PID_FILE}" || true)"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    echo "[stop] terminating scheduler pid=${pid}"
    run_cmd kill "${pid}" || true
    sleep 1
    if kill -0 "${pid}" 2>/dev/null; then
      echo "[stop] force killing scheduler pid=${pid}"
      run_cmd kill -9 "${pid}" || true
    fi
  else
    echo "[stop] pid file exists but process already stopped"
  fi
  [[ "${DRY_RUN}" -eq 0 ]] && rm -f "${PID_FILE}"
else
  echo "[stop] no pid file at ${PID_FILE}"
fi

if [[ "${STOP_UNITS}" -eq 1 ]]; then
  # list-units may prefix the UNIT column with a status glyph (e.g. ●); take the *.service token.
  mapfile -t units < <(
    systemctl list-units --all --type=service --no-legend "pv-ogbn-*" |
      awk 'NF { for (i = 1; i <= NF; i++) if ($i ~ /\.service$/) { print $i; break } }'
  )
  if [[ "${#units[@]}" -eq 0 ]]; then
    echo "[stop] no pv-ogbn-* systemd units found"
  else
    echo "[stop] matched ${#units[@]} pv-ogbn-* units"
    for unit in "${units[@]}"; do
      echo "  - ${unit}"
      run_cmd sudo systemctl stop "${unit}"
      run_cmd sudo systemctl reset-failed "${unit}"
    done
  fi
fi

if [[ "${KILL_CHILDREN}" -eq 1 ]]; then
  patterns=(
    "train_paper_venue_shared_task_pool.py"
    "train_paper_venue_shared.py --task_name PV_ogbn_mag_"
  )
  for pattern in "${patterns[@]}"; do
    echo "[kill] pkill -f ${pattern}"
    run_cmd pkill -f "${pattern}" || true
  done
fi

echo "[stop] remaining units snapshot:"
run_cmd systemctl list-units --all "pv-ogbn-*"
