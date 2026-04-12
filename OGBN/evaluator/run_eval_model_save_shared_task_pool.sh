#!/usr/bin/env bash
# Evaluate all model checkpoints under model_save_shared with task pool scheduling.
# Requirements implemented:
# - GPU 0 only
# - keep up to 10 processes
# - whenever running jobs < 6, refill immediately from task pool
# - evaluate with seeds 35 and 36
# - enable wandb project OGBN-EVL

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OGBN_ROOT="${OGBN_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

MAX_PARALLEL="${MAX_PARALLEL:-10}"
REFILL_THRESHOLD="${REFILL_THRESHOLD:-10}"
GPU_ID="${GPU_ID:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-OGBN-EVL}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"

MODEL_DIR="${OGBN_ROOT}/model_save_shared"
OUT_ROOT="${OGBN_ROOT}/evaluator/results"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${OUT_ROOT}/task_pool_eval_${TS}"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "${LOG_DIR}"

mapfile -t CKPTS < <(
  python - <<'PY'
import os

root = os.path.abspath("model_save_shared")
paths = []
for dpath, dnames, fnames in os.walk(root):
    dnames[:] = [d for d in dnames if d != "metrics"]
    for fn in fnames:
        if not fn.startswith("PV_ogbn_mag"):
            continue
        if fn.endswith((".log", ".json", ".jsonl", ".txt", ".csv", ".md", ".npy", ".npz", ".pt", ".pth")):
            continue
        paths.append(os.path.join(dpath, fn))

for p in sorted(paths):
    print(p)
PY
)

if [[ ${#CKPTS[@]} -eq 0 ]]; then
  echo "[error] no checkpoints found under ${MODEL_DIR}"
  exit 1
fi

declare -a SEEDS=(35 36)
declare -a SPLITS=(train valid test)
declare -a CMDS=()
declare -a TAGS=()

for ckpt in "${CKPTS[@]}"; do
  stem="$(basename "${ckpt}")"
  model_subdir="$(basename "$(dirname "$(dirname "${ckpt}")")")"
  for split in "${SPLITS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      run_name="eval_${stem}_${split}_seed${seed}"
      group="eval-ogbn-mag-${model_subdir}-seed${seed}"

      cmd=(
        "cd \"${OGBN_ROOT}\" &&"
        "CUDA_VISIBLE_DEVICES=${GPU_ID} python evaluator/eval_test_accuracy.py"
        "--checkpoint \"${ckpt}\""
        "--split \"${split}\""
        "--cuda 0"
        "--seed \"${seed}\""
        "--save_dir \"${RUN_ROOT}\""
        "--use_wandb"
        "--wandb_project \"${WANDB_PROJECT}\""
        "--wandb_mode \"${WANDB_MODE}\""
        "--wandb_group \"${group}\""
        "--wandb_run_name \"${run_name}\""
      )
      if [[ -n "${WANDB_ENTITY}" ]]; then
        cmd+=("--wandb_entity \"${WANDB_ENTITY}\"")
      fi
      cmd+=("--wandb_tags ogbn eval pv model_save_shared split_${split} seed${seed}")

      CMDS+=("$(printf '%s ' "${cmd[@]}")")
      TAGS+=("${run_name}")
    done
  done
done

total="${#CMDS[@]}"
if [[ "${total}" -eq 0 ]]; then
  echo "[error] task pool is empty."
  exit 1
fi

echo "[pool] total_tasks=${total} max_parallel=${MAX_PARALLEL} refill_threshold=${REFILL_THRESHOLD} gpu=${GPU_ID}"
echo "[pool] run_root=${RUN_ROOT}"
echo "[pool] seeds=${SEEDS[*]} splits=${SPLITS[*]} wandb_project=${WANDB_PROJECT}"

running=0
next_idx=0
fail=0
completed=0
declare -a RUNNING_PIDS=()
declare -a RUNNING_TAGS=()

launch_one() {
  local i="$1"
  local tag="${TAGS[$i]}"
  local log="${LOG_DIR}/${tag}.log"
  echo "[start] $((i + 1))/${total} ${tag}"
  bash -lc "${CMDS[$i]}" >"${log}" 2>&1 &
  RUNNING_PIDS+=("$!")
  RUNNING_TAGS+=("${tag}")
  running=$((running + 1))
}

refill_to_max_if_needed() {
  if [[ "${running}" -lt "${REFILL_THRESHOLD}" ]]; then
    while [[ "${next_idx}" -lt "${total}" && "${running}" -lt "${MAX_PARALLEL}" ]]; do
      launch_one "${next_idx}"
      next_idx=$((next_idx + 1))
    done
  fi
}

while [[ "${next_idx}" -lt "${total}" && "${running}" -lt "${MAX_PARALLEL}" ]]; do
  launch_one "${next_idx}"
  next_idx=$((next_idx + 1))
done

while [[ "${running}" -gt 0 ]]; do
  new_pids=()
  new_tags=()
  for i in "${!RUNNING_PIDS[@]}"; do
    pid="${RUNNING_PIDS[$i]}"
    tag="${RUNNING_TAGS[$i]}"
    if kill -0 "${pid}" 2>/dev/null; then
      new_pids+=("${pid}")
      new_tags+=("${tag}")
    else
      set +e
      wait "${pid}"
      rc=$?
      set -e
      if [[ "${rc}" -ne 0 ]]; then
        echo "[fail] ${tag} exited with rc=${rc}"
        fail=1
      else
        echo "[done] ${tag}"
      fi
      running=$((running - 1))
      completed=$((completed + 1))
    fi
  done
  RUNNING_PIDS=("${new_pids[@]:-}")
  RUNNING_TAGS=("${new_tags[@]:-}")
  refill_to_max_if_needed
  echo "[status] completed=${completed}/${total} running=${running} queued=$((total - completed - running))"
  if [[ "${running}" -gt 0 ]]; then
    sleep 2
  fi
done

echo "---"
echo "Logs: ${LOG_DIR}"
echo "Results root: ${RUN_ROOT}"
if [[ "${fail}" -ne 0 ]]; then
  echo "[pool] some tasks failed."
  exit 1
fi
echo "[pool] all tasks finished successfully."
