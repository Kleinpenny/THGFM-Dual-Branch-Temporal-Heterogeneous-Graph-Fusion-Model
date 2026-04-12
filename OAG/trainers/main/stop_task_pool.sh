#!/usr/bin/env bash
set -euo pipefail

STOP_UNITS=1
KILL_CHILDREN=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  ./trainers/main/stop_task_pool.sh [options]

Options:
  --kill-children               Also kill leftover training child processes with pkill -f
  --skip-units                  Skip systemd stop/reset and only do child cleanup
  --dry-run                     Print actions without executing them
  -h, --help                    Show this help message
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
    --kill-children)
      KILL_CHILDREN=1
      shift
      ;;
    --skip-units)
      STOP_UNITS=0
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

PATTERNS=(
  "ad-*"
  "pf-l1-*"
  "pf-l2-*"
  "pv-*"
)

CHILD_PATTERNS=(
  "train_author_disambiguation_shared.py"
  "train_paper_field_shared.py --task_name PF_XLNet_L1_"
  "train_paper_field_shared.py --task_name PF_XLNet_L2_"
  "train_paper_venue_shared.py --task_name PV_XLNet_"
)

if [[ "${STOP_UNITS}" -eq 1 ]]; then
  for pattern in "${PATTERNS[@]}"; do
    mapfile -t units < <(systemctl list-units --all --type=service --no-legend "${pattern}" | awk '{print $1}')
    if [[ "${#units[@]}" -eq 0 ]]; then
      echo "[stop] no units matched ${pattern}"
      continue
    fi

    echo "[stop] matched ${#units[@]} units for ${pattern}"
    for unit in "${units[@]}"; do
      echo "  - ${unit}"
      run_cmd sudo systemctl stop "${unit}"
      run_cmd sudo systemctl reset-failed "${unit}"
    done
  done
fi

if [[ "${KILL_CHILDREN}" -eq 1 ]]; then
  for pattern in "${CHILD_PATTERNS[@]}"; do
    echo "[kill] pkill -f ${pattern}"
    run_cmd pkill -f "${pattern}" || true
  done
fi

echo "[stop] remaining units snapshot:"
run_cmd systemctl list-units --all "ad-*" "pf-l1-*" "pf-l2-*" "pv-*"
