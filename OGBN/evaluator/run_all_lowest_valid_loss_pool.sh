#!/usr/bin/env bash
# Evaluate all lowest_valid_loss checkpoints on valid + test splits.
# Task pool: max 4 concurrent jobs on GPU 0.
# Each job runs: valid split then test split sequentially for one checkpoint.
set -euo pipefail

OGBN_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$OGBN_ROOT"

LOG_DIR="${OGBN_ROOT}/evaluator/parallel_logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
SEED=43
MAX_JOBS=4
GPU=0

EVAL_PY=(python evaluator/eval_test_accuracy.py --seed "${SEED}" --cuda "${GPU}")

# ─── All lowest_valid_loss checkpoint paths ───────────────────────────────────
CHECKPOINTS=(
  "model_save_shared/drsf/nhid256_nlayers3_nepoch200/PV_ogbn_mag_drsf_s43_lowest_valid_loss_nhid256_nlayers3_nepoch200"
  "model_save_shared/drsf_cglf/nhid256_nlayers3_nepoch200/PV_ogbn_mag_lf_r001_s001_s43_lowest_valid_loss_nhid256_nlayers3_nepoch200"
  "model_save_shared/drsf_nglf/nhid256_nlayers3_nepoch200/PV_ogbn_mag_nglf_r001_s001_s43_lowest_valid_loss_nhid256_nlayers3_nepoch200"
  "model_save_shared/drsf_nglf_rota/nhid256_nlayers3_nepoch200/PV_ogbn_mag_nglf_r110_s110_s43_lowest_valid_loss_nhid256_nlayers3_nepoch200"
  "model_save_shared/drsf_nglf_rota/nhid256_nlayers3_nepoch200/PV_ogbn_mag_nglf_r111_s111_s43_lowest_valid_loss_nhid256_nlayers3_nepoch200"
  "model_save_shared/drsf_ngsf/nhid256_nlayers3_nepoch200/PV_ogbn_mag_ngsf_r001_s001_s43_lowest_valid_loss_nhid256_nlayers3_nepoch200"
  "model_save_shared/drsf_ngsf_rota/nhid256_nlayers3_nepoch200/PV_ogbn_mag_ngsf_r110_s110_s43_lowest_valid_loss_nhid256_nlayers3_nepoch200"
  "model_save_shared/drsf_ngsf_rota/nhid256_nlayers3_nepoch200/PV_ogbn_mag_ngsf_r111_s111_s43_lowest_valid_loss_nhid256_nlayers3_nepoch200"
  "model_save_shared/drsf_rota/nhid256_nlayers3_nepoch200/PV_ogbn_mag_lf_r110_s110_s43_lowest_valid_loss_nhid256_nlayers3_nepoch200"
  "model_save_shared/drsf_rota/nhid256_nlayers3_nepoch200/PV_ogbn_mag_lf_r111_s111_s43_lowest_valid_loss_nhid256_nlayers3_nepoch200"
  "model_save_shared/drsf_sf/nhid256_nlayers3_nepoch200/PV_ogbn_mag_drsf_sf_s43_lowest_valid_loss_nhid256_nlayers3_nepoch200"
  "model_save_shared/drsf_sf_rota/nhid256_nlayers3_nepoch200/PV_ogbn_mag_sf_r110_s110_s43_lowest_valid_loss_nhid256_nlayers3_nepoch200"
  "model_save_shared/drsf_sf_rota/nhid256_nlayers3_nepoch200/PV_ogbn_mag_sf_r111_s111_s43_lowest_valid_loss_nhid256_nlayers3_nepoch200"
  "model_save_shared/drsf_tc_nglf/nhid256_nlayers3_nepoch200/PV_ogbn_mag_tcnglf_r001_s001_s43_lowest_valid_loss_nhid256_nlayers3_nepoch200"
  "model_save_shared/drsf_tcnglf_rota/nhid256_nlayers3_nepoch200/PV_ogbn_mag_tcnglf_r110_s110_s43_lowest_valid_loss_nhid256_nlayers3_nepoch200"
  "model_save_shared/drsf_tcnglf_rota/nhid256_nlayers3_nepoch200/PV_ogbn_mag_tcnglf_r111_s111_s43_lowest_valid_loss_nhid256_nlayers3_nepoch200"
  "model_save_shared/drsf_tc_ngsf/nhid256_nlayers3_nepoch200/PV_ogbn_mag_drsf_tc_ngsf_s43_lowest_valid_loss_nhid256_nlayers3_nepoch200"
  "model_save_shared/drsf_tcngsf_rota/nhid256_nlayers3_nepoch200/PV_ogbn_mag_tcngsf_r110_s110_s43_lowest_valid_loss_nhid256_nlayers3_nepoch200"
  "model_save_shared/drsf_tcngsf_rota/nhid256_nlayers3_nepoch200/PV_ogbn_mag_tcngsf_r111_s111_s43_lowest_valid_loss_nhid256_nlayers3_nepoch200"
  "model_save_shared/rtta/nhid256_nlayers3_nepoch200/PV_ogbn_mag_rtta_s43_lowest_valid_loss_nhid256_nlayers3_nepoch200"
  "model_save_shared/rtta_rota/nhid256_nlayers3_nepoch200/PV_ogbn_mag_rtta_r110_s110_s43_lowest_valid_loss_nhid256_nlayers3_nepoch200"
  "model_save_shared/rtta_rota/nhid256_nlayers3_nepoch200/PV_ogbn_mag_rtta_r111_s111_s43_lowest_valid_loss_nhid256_nlayers3_nepoch200"
  "model_save_shared/ssta/nhid256_nlayers3_nepoch200/PV_ogbn_mag_ssta_s43_lowest_valid_loss_nhid256_nlayers3_nepoch200"
  "model_save_shared/ssta_rota/nhid256_nlayers3_nepoch200/PV_ogbn_mag_ssta_r110_s110_s43_lowest_valid_loss_nhid256_nlayers3_nepoch200"
  "model_save_shared/ssta_rota/nhid256_nlayers3_nepoch200/PV_ogbn_mag_ssta_r111_s111_s43_lowest_valid_loss_nhid256_nlayers3_nepoch200"
)

TOTAL=${#CHECKPOINTS[@]}
echo "Total checkpoints: ${TOTAL}"
echo "Task pool size: ${MAX_JOBS} (GPU ${GPU})"
echo "Logs: ${LOG_DIR}/"
echo "Timestamp prefix: ${TS}"
echo ""

# ─── Task pool bookkeeping ────────────────────────────────────────────────────
declare -a RUNNING_PIDS=()
declare -a RUNNING_TAGS=()

# Wait until the number of running jobs drops below MAX_JOBS
_wait_for_slot () {
  while (( ${#RUNNING_PIDS[@]} >= MAX_JOBS )); do
    local new_pids=()
    local new_tags=()
    for i in "${!RUNNING_PIDS[@]}"; do
      local pid="${RUNNING_PIDS[$i]}"
      local tag="${RUNNING_TAGS[$i]}"
      if kill -0 "$pid" 2>/dev/null; then
        new_pids+=("$pid")
        new_tags+=("$tag")
      else
        wait "$pid" && echo "[DONE  ] ${tag}" || echo "[FAILED] ${tag} (exit $?)"
      fi
    done
    RUNNING_PIDS=("${new_pids[@]+"${new_pids[@]}"}")
    RUNNING_TAGS=("${new_tags[@]+"${new_tags[@]}"}")
    if (( ${#RUNNING_PIDS[@]} >= MAX_JOBS )); then
      sleep 5
    fi
  done
}

# Wait for all remaining jobs
_wait_all () {
  for i in "${!RUNNING_PIDS[@]}"; do
    local pid="${RUNNING_PIDS[$i]}"
    local tag="${RUNNING_TAGS[$i]}"
    wait "$pid" && echo "[DONE  ] ${tag}" || echo "[FAILED] ${tag} (exit $?)"
  done
  RUNNING_PIDS=()
  RUNNING_TAGS=()
}

# ─── Submit jobs ──────────────────────────────────────────────────────────────
IDX=0
for CKPT_REL in "${CHECKPOINTS[@]}"; do
  IDX=$(( IDX + 1 ))
  CKPT="${OGBN_ROOT}/${CKPT_REL}"
  TAG="$(basename "${CKPT_REL}")"
  SAFE_TAG="${TAG//\//_}"
  LOG="${LOG_DIR}/${TS}_${IDX}_${SAFE_TAG}.log"

  _wait_for_slot

  echo "[START ] (${IDX}/${TOTAL}) ${TAG}"
  {
    echo "=== $(date -Is) START ${TAG}"
    CUDA_VISIBLE_DEVICES="${GPU}" "${EVAL_PY[@]}" --checkpoint "${CKPT}" --split valid
    CUDA_VISIBLE_DEVICES="${GPU}" "${EVAL_PY[@]}" --checkpoint "${CKPT}" --split test
    echo "=== $(date -Is) END ${TAG}"
  } >"${LOG}" 2>&1 &

  RUNNING_PIDS+=("$!")
  RUNNING_TAGS+=("${TAG}")
done

_wait_all

echo ""
echo "All ${TOTAL} checkpoints evaluated. Results in: ${OGBN_ROOT}/evaluator/results/"
