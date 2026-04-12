# `trainers/main`: PF-L1 shared pipeline

This document is for Paper-Field L1 (PF-L1) shared-memory training.

- Builder wrapper: `./trainers/main/build_shared_pf_base.py`
- Trainer wrapper: `./trainers/main/train_paper_field_shared.py`

## 1) Build PF-L1 shared dataset

Recommended path: reuse AD shared arrays and build PF-L1 task labels on top.

```bash
set -euo pipefail
source /home/yxpeng/anaconda3/etc/profile.d/conda.sh
conda activate thgfm
cd /home/yxpeng/Projects/Papers/2026/THGFM-Dual-Branch-Temporal-Heterogeneous-Graph-Fusion-Model/OAG

python "./trainers/main/build_shared_pf_base.py" \
  --field_level L1 \
  --source_shared_base_dir /home/yxpeng/Projects/Papers/2026/THGFM-Dual-Branch-Temporal-Heterogeneous-Graph-Fusion-Model/data/shared_ad_base_CS_XLNet \
  --out_dir /home/yxpeng/Projects/Papers/2026/THGFM-Dual-Branch-Temporal-Heterogeneous-Graph-Fusion-Model/data/shared_pf_base_CS_XLNet_L1
```

## 2) PF-L1 smoke test

```bash
conda activate thgfm
cd /home/yxpeng/Projects/Papers/2026/THGFM-Dual-Branch-Temporal-Heterogeneous-Graph-Fusion-Model/OAG

python "./trainers/main/train_paper_field_shared.py" \
  --shared_base_dir /home/yxpeng/Projects/Papers/2026/THGFM-Dual-Branch-Temporal-Heterogeneous-Graph-Fusion-Model/data/shared_pf_base_CS_XLNet_L1 \
  --field_level L1 \
  --model_dir ./model_save_shared \
  --task_name PF_XLNet_L1_shared_smoke \
  --conv_name rtta \
  --cuda 0 \
  --n_epoch 1 --n_batch 2 --repeat 1 --batch_size 128 \
  --sample_depth 4 --sample_width 64 --n_pool 2
```

## 3) PF-L1 ablation batch launch (`systemd-run`)

```bash
# Optional: cap memory for this user slice (run once as admin)
sudo mkdir -p /etc/systemd/system/python-yxpeng.slice.d
sudo tee /etc/systemd/system/python-yxpeng.slice.d/500-memory-limit.conf >/dev/null <<'EOF'
[Slice]
MemoryAccounting=yes
MemoryMax=500G
EOF
sudo systemctl daemon-reload
######################################################################

conda activate thgfm
cd /home/yxpeng/Projects/Papers/2026/THGFM-Dual-Branch-Temporal-Heterogeneous-Graph-Fusion-Model/OAG

PROJECT_DIR=/home/yxpeng/Projects/Papers/2026/THGFM-Dual-Branch-Temporal-Heterogeneous-Graph-Fusion-Model/OAG
CONDA_SH=/home/yxpeng/anaconda3/etc/profile.d/conda.sh
COMMON_WANDB_ARGS=(--use_wandb --wandb_project "${WANDB_PROJECT}")
if [[ -n "${WANDB_ENTITY}" ]]; then
  COMMON_WANDB_ARGS+=(--wandb_entity "${WANDB_ENTITY}")
fi
if [[ -n "${WANDB_MODE}" ]]; then
  COMMON_WANDB_ARGS+=(--wandb_mode "${WANDB_MODE}")
fi

launch_shared_unit() {
  local unit_name="$1"
  local log_file="$2"
  shift 2

  mkdir -p "$(dirname "$log_file")"
  sudo systemctl stop "${unit_name}.service" 2>/dev/null || true
  sudo systemctl reset-failed "${unit_name}.service" 2>/dev/null || true
  local -a unit_args=(
    sudo systemd-run
    --unit="$unit_name"
    --slice="python-yxpeng.slice"
    --property=User=yxpeng
    --property=WorkingDirectory="${PROJECT_DIR}"
    --property=OOMPolicy=kill
    --property=KillMode=control-group
    --property=MemoryAccounting=yes
  )
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    unit_args+=(--setenv="WANDB_API_KEY=${WANDB_API_KEY}")
  fi
  if [[ -n "${WANDB_MODE:-}" ]]; then
    unit_args+=(--setenv="WANDB_MODE=${WANDB_MODE}")
  fi
  "${unit_args[@]}" /bin/bash -lc "source '${CONDA_SH}' && conda activate thgfm && exec $* >> '$log_file' 2>&1"
}

PF_BASE=/home/yxpeng/Projects/Papers/2026/THGFM-Dual-Branch-Temporal-Heterogeneous-Graph-Fusion-Model/data/shared_pf_base_CS_XLNet_L1
CUDA=1
SEEDS=(43)  # add 44 45 if needed
declare -a ABLATIONS=(
  # ---- RTTA (Table order: 0/0/0 -> 0/0/1 -> 1/1/1 -> 1/1/0 -> 0/1/1 -> 0/1/0) ----
  "rtta_rte0|rtta_rte0|none|none|none|none|-1|-1"
  "rtta|rtta|none|none|none|none|-1|-1"
  "rtta_rota_qk_rte1|rtta_rota|qk|1|none|none|-1|-1"
  "rtta_rota_qk_rte0|rtta_rota|qk|0|none|none|-1|-1"
  "rtta_rota_k_rte1|rtta_rota|k|1|none|none|-1|-1"
  "rtta_rota_k_rte0|rtta_rota|k|0|none|none|-1|-1"

  # ---- SSTA (Table order: 0/0/0 -> 0/0/1 -> 1/1/1 -> 1/1/0 -> 0/1/1 -> 0/1/0) ----
  "ssta_rte0|ssta_rte0|none|none|none|none|-1|-1"
  "ssta|ssta|none|none|none|none|-1|-1"
  "ssta_rota_qk_rte1|ssta_rota|qk|1|none|none|-1|-1"
  "ssta_rota_qk_rte0|ssta_rota|qk|0|none|none|-1|-1"
  "ssta_rota_k_rte1|ssta_rota|k|1|none|none|-1|-1"
  "ssta_rota_k_rte0|ssta_rota|k|0|none|none|-1|-1"

  # ---- DRSF-LF (Table order) ----
  "drsf_rte0|drsf_rte0|none|none|none|none|-1|-1"
  "drsf|drsf|none|none|none|none|-1|-1"
  "drsf_rota_qk_rte1|drsf_rota|qk|1|none|none|-1|-1"
  "drsf_rota_qk_rte0|drsf_rota|qk|0|none|none|-1|-1"

  # ---- DRSF-GF (Table order) ----
  "drsf_gate|drsf_gate|none|none|none|none|-1|-1"
  "drsf_gate_rota_qk_rte1|drsf_gate_rota|qk|1|none|none|-1|-1"
  "drsf_gate_rota_qk_rte0|drsf_gate_rota|qk|0|none|none|-1|-1"
  "drsf_gate_rota_rttaK1_sstaQK1|drsf_gate_rota|qk|1|k|qk|1|1"
  "drsf_gate_rota_rttaQK1_sstaK1|drsf_gate_rota|qk|1|qk|k|1|1"
  "drsf_gate_rota_rttaK1_sstaQK0|drsf_gate_rota|qk|1|k|qk|1|0"
  "drsf_gate_rota_rttaQK0_sstaK1|drsf_gate_rota|qk|1|qk|k|0|1"
)

for SEED in "${SEEDS[@]}"; do
  for spec in "${ABLATIONS[@]}"; do
    IFS='|' read -r SUFFIX CONV ROTA_MODE ROTA_KEEP_RTE RTTA_MODE SSTA_MODE RTTA_KEEP SSTA_KEEP <<< "$spec"
    CMD=(
      python "./trainers/main/train_paper_field_shared.py"
      --shared_base_dir "${PF_BASE}"
      --field_level L1
      --model_dir ./model_save_shared
      --task_name "PF_XLNet_L1_${SUFFIX}_s${SEED}"
      --conv_name "${CONV}"
      --cuda "${CUDA}"
      --seed "${SEED}"
      --deterministic
      "${COMMON_WANDB_ARGS[@]}"
      --wandb_run_name "PF_XLNet_L1_${SUFFIX}_s${SEED}_shared"
      --wandb_group pf-l1-shared-ablation
    )
    if [[ "${ROTA_MODE}" != "none" ]]; then
      CMD+=(--rota_mode "${ROTA_MODE}" --rota_keep_rte "${ROTA_KEEP_RTE}")
    fi
    if [[ "${RTTA_MODE}" != "none" ]]; then
      CMD+=(--rtta_rota_mode "${RTTA_MODE}")
    fi
    if [[ "${SSTA_MODE}" != "none" ]]; then
      CMD+=(--ssta_rota_mode "${SSTA_MODE}")
    fi
    if [[ "${RTTA_KEEP}" != "-1" ]]; then
      CMD+=(--rtta_rota_keep_rte "${RTTA_KEEP}")
    fi
    if [[ "${SSTA_KEEP}" != "-1" ]]; then
      CMD+=(--ssta_rota_keep_rte "${SSTA_KEEP}")
    fi
    CMD_STR=$(printf '%q ' "${CMD[@]}")
    launch_shared_unit \
      "pf-l1-${SUFFIX}-s${SEED}" \
      "./logs/train_paper_field_l1_shared_${SUFFIX}_s${SEED}.log" \
      "${CMD_STR}"
  done
done
```

## 4) Monitor

```bash
systemctl list-units "pf-l1-*" --all
tail -f ./logs/train_paper_field_l1_shared_rtta_s43.log
```

## 5) Stop services

```bash
for u in $(systemctl list-units --all --type=service --no-legend "pf-l1-*" | awk '{print $1}'); do
  sudo systemctl stop "$u" || true
  sudo systemctl reset-failed "$u" || true
done
systemctl list-units "pf-l1-*" --all
```

Emergency kill (if a child process remains):

```bash
pkill -f "train_paper_field_shared.py --task_name PF_XLNet_L1_"
```
