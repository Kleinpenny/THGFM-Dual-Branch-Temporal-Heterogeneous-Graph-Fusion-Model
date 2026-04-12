#!/usr/bin/env bash
# Run three OAG THGFM evaluations in parallel on one GPU (default: GPU 0).
# Override GPU with: GPU_ID=1 ./run_parallel_eval_4gpu.sh
#
# Defaults point at model_save_shared/rtta; override with env vars if needed.
#
# Logs: evaluator/logs_parallel/eval_<timestamp>_<job>.log

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OAG_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EVAL_DIR="${SCRIPT_DIR}"
LOG_DIR="${EVAL_DIR}/logs_parallel"
GPU_ID="${GPU_ID:-0}"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"

PV_CKPT="${PV_CKPT:-${OAG_ROOT}/model_save_shared/rtta}"
PF_L1_CKPT="${PF_L1_CKPT:-${OAG_ROOT}/model_save_shared/rtta}"
PF_L2_CKPT="${PF_L2_CKPT:-${OAG_ROOT}/model_save_shared/rtta}"

cd "${OAG_ROOT}"

PIDS=()
FAIL=0

run_job() {
  local name="$1"
  shift
  local log="${LOG_DIR}/eval_${TS}_${name}.log"
  echo "[launch] CUDA_VISIBLE_DEVICES=${GPU_ID} -> ${name} (log: ${log})"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "$@" >"${log}" 2>&1 &
  PIDS+=($!)
}

run_job pv_thgfm python "${EVAL_DIR}/eval_pv_test_ndcg_mrr.py" \
  --checkpoint "${PV_CKPT}" --cuda 0 --save_dir "${EVAL_DIR}/results"

run_job pf_l1_thgfm python "${EVAL_DIR}/eval_pf_l1_test_ndcg_mrr.py" \
  --checkpoint "${PF_L1_CKPT}" --cuda 0 --save_dir "${EVAL_DIR}/results"

run_job pf_l2_thgfm python "${EVAL_DIR}/eval_pf_l2_test_ndcg_mrr.py" \
  --checkpoint "${PF_L2_CKPT}" --cuda 0 --save_dir "${EVAL_DIR}/results"

for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    FAIL=1
  fi
done

if [[ "${FAIL}" -ne 0 ]]; then
  echo "[done] One or more jobs failed. Check logs under ${LOG_DIR}"
  exit 1
fi
echo "[done] All jobs finished. Logs: ${LOG_DIR}/eval_${TS}_*.log"
exit 0
