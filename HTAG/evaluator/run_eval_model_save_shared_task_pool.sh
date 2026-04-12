#!/usr/bin/env bash
# Evaluate all model_save_shared checkpoints on HTAG Arxiv/DBLP with task pool scheduling.
# - Uses only GPU 0.
# - Runs up to MAX_PARALLEL eval processes concurrently (default 10 via MAX_PARALLEL env).
# - Whenever running count drops below MAX_PARALLEL, launches from the pool until full or pool empty.
# - Evaluates both highest_valid_acc and lowest_valid_loss checkpoints.
# - Evaluates with seeds 35 and 36.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HTAG_ROOT="${HTAG_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
EVAL_ROUNDS="${1:-10}"
MAX_PARALLEL="${MAX_PARALLEL:-10}"
GPU_ID="${GPU_ID:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-HTAG-EVL}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"

MODEL_DIR="${HTAG_ROOT}/model_save_shared"
ARXIV_DATA="${HTAG_ROOT}/Arxiv_PV/data/shared_pv_base_HTAG_arxiv"
DBLP_DATA="${HTAG_ROOT}/DBLP_PV/data/shared_pv_base_HTAG_dblp"

OUT_ROOT="${HTAG_ROOT}/evaluator/results"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${OUT_ROOT}/task_pool_eval_${TS}"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "${LOG_DIR}"

shopt -s nullglob
declare -a CKPTS=("${MODEL_DIR}"/*highest_valid_acc.pt "${MODEL_DIR}"/*lowest_valid_loss.pt)
if [[ ${#CKPTS[@]} -eq 0 ]]; then
  echo "[error] no checkpoints found under ${MODEL_DIR}"
  exit 1
fi

declare -a SEEDS=(35 36)
declare -a CMDS=()
declare -a TAGS=()

for ckpt in "${CKPTS[@]}"; do
  base="$(basename "${ckpt}")"
  if [[ "${base}" == *"_Arxiv_"* ]]; then
    shared="${ARXIV_DATA}"
    task_subdir="Arxiv"
  elif [[ "${base}" == *"_DBLP_"* ]]; then
    shared="${DBLP_DATA}"
    task_subdir="DBLP"
  else
    echo "[skip] cannot infer dataset from ${base}"
    continue
  fi

  model_subdir="$(awk -F'_' '{print $4}' <<<"${base}")"
  model_name="${model_subdir}"
  ckpt_stem="${base%.pt}"

  for seed in "${SEEDS[@]}"; do
    out_json="${RUN_ROOT}/${task_subdir}/${model_subdir}/eval_${ckpt_stem}_seed${seed}.json"
    mkdir -p "$(dirname "${out_json}")"
    run_name="eval_${ckpt_stem}_seed${seed}"
    group="eval-${task_subdir,,}-${model_subdir}-seed${seed}"

    if [[ "${task_subdir}" == "Arxiv" ]]; then
      eval_script="evaluator/eval_pv_htag_arxiv.py"
    else
      eval_script="evaluator/eval_pv_htag_dblp.py"
    fi

    cmd=(
      "cd \"${HTAG_ROOT}\" &&"
      "CUDA_VISIBLE_DEVICES=${GPU_ID} exec -a \"HTAG_${run_name}\" python ${eval_script}"
      "--checkpoint \"${ckpt}\""
      "--shared_base_dir \"${shared}\""
      "--output_json \"${out_json}\""
      "--save_dir \"${OUT_ROOT}\""
      "--task_subdir \"${task_subdir}\""
      "--model_subdir \"${model_subdir}\""
      "--cuda 0"
      "--eval_rounds \"${EVAL_ROUNDS}\""
      "--batch_size 128"
      "--sample_depth 6"
      "--sample_width 128"
      "--n_hid 256"
      "--n_layers 3"
      "--n_heads 8"
      "--seed \"${seed}\""
      "--use_wandb"
      "--wandb_project \"${WANDB_PROJECT}\""
      "--wandb_mode \"${WANDB_MODE}\""
      "--wandb_group \"${group}\""
      "--wandb_run_name \"${run_name}\""
    )
    if [[ -n "${WANDB_ENTITY}" ]]; then
      cmd+=("--wandb_entity \"${WANDB_ENTITY}\"")
    fi
    cmd+=("--wandb_tags htag eval pv model_save_shared seed${seed}")
    CMDS+=("$(printf '%s ' "${cmd[@]}")")
    TAGS+=("${run_name}")
  done
done

total="${#CMDS[@]}"
if [[ "${total}" -eq 0 ]]; then
  echo "[error] task pool is empty."
  exit 1
fi

echo "[pool] total_tasks=${total} max_parallel=${MAX_PARALLEL} gpu=${GPU_ID}"
echo "[pool] run_root=${RUN_ROOT}"

running=0
next_idx=0
fail=0

launch_one() {
  local i="$1"
  local tag="${TAGS[$i]}"
  local log="${LOG_DIR}/${tag}.log"
  echo "[start] $((i + 1))/${total} ${tag}"
  bash -lc "${CMDS[$i]}" >"${log}" 2>&1 &
  running=$((running + 1))
}

while [[ "${next_idx}" -lt "${total}" && "${running}" -lt "${MAX_PARALLEL}" ]]; do
  launch_one "${next_idx}"
  next_idx=$((next_idx + 1))
done

while [[ "${running}" -gt 0 ]]; do
  set +e
  wait -n
  rc=$?
  set -e

  if [[ "${rc}" -ne 0 ]]; then
    echo "[fail] one task exited with rc=${rc}"
    fail=1
  fi
  running=$((running - 1))

  # Refill immediately up to MAX_PARALLEL whenever slots are free (not just one-at-a-time).
  while [[ "${next_idx}" -lt "${total}" && "${running}" -lt "${MAX_PARALLEL}" ]]; do
    launch_one "${next_idx}"
    next_idx=$((next_idx + 1))
  done
done

echo "---"
echo "Logs: ${LOG_DIR}"
echo "Results root: ${RUN_ROOT}"
if [[ "${fail}" -ne 0 ]]; then
  echo "[pool] some tasks failed."
  exit 1
fi
echo "[pool] all tasks finished successfully."
