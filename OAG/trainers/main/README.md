# OAG `trainers/main` Run Guide

This directory contains the actual launchers used for OAG experiments:

- THGFM variant ablations (manifest + task pool): `start_task_pool.sh`

Legacy `hgt` / `HGTConv` was removed from `THGFM`; use **`rtta`** or other supported `--conv_name` values.

## 0) Weights & Biases (wandb) Setup

Install and log in once (inside your training environment):

```bash
pip install -U wandb
wandb login
```

Optional environment variables (recommended):

```bash
export WANDB_PROJECT=THGFM
export WANDB_ENTITY=<your_team_or_username>   # optional
export WANDB_MODE=online                      # online | offline | disabled
```

Notes:

- THGFM task-pool manifests are generated with `--use_wandb` by default.

## 1) Environment Setup

```bash
cd /path/to/repo/OAG
```

## 2) THGFM Variant Ablations (Task Pool)

### 2.1 Recommended launch

```bash
./trainers/main/start_task_pool.sh --pool-kind all
```

`--pool-kind` choices:

- `ad`: AD family only
- `all`: AD + PF-L1 + PF-L2 + PV
- `ta`: TA-only subset (`TA-NGLF`, `TA-NGSF`)
- `missing`: only missing cells for the detailed ablation table

### 2.2 Resource-tuned launch (same setting used in scripts)

```bash
./trainers/main/start_task_pool.sh \
  --pool-kind all \
  --max-running 12 \
  --gpu-ids 0 1 2 3 \
  --max-per-gpu 4 \
  --max-load1 75 \
  --max-gpu-util 0 \
  --seeds 43 \
  --n-pool 8 \
  --cpu-threads 1
```

### 2.3 Use a pre-generated manifest

```bash
./trainers/main/start_task_pool.sh \
  --manifest ./trainers/main/manifests/task_pool_all.jsonl \
  --max-running 12 \
  --gpu-ids 0 1 2 3 \
  --max-per-gpu 4
```

### 2.4 Manual single-run example (without pool)

```bash
python ./trainers/main/train_paper_venue_shared.py \
  --shared_base_dir ./data/shared_pv_base_CS_XLNet \
  --model_dir ./model_save_shared \
  --conv_name drsf_tcnglf_rota \
  --task_name PV_XLNet_tcnglf_r111_s111_s43 \
  --cuda 0 \
  --n_pool 8 \
  --seed 43 \
  --deterministic \
  --use_wandb --wandb_project THGFM --wandb_group pv-shared-ablation
```

## 3) Background Launch / Monitoring / Stop

Start THGFM pool in background:

```bash
mkdir -p ./logs
nohup ./trainers/main/start_task_pool.sh --pool-kind all \
  > ./logs/nohup_task_pool.out 2>&1 < /dev/null &
```

Auto-watch and relaunch when GPU jobs are low:

```bash
nohup ./trainers/main/watch_task_pool_nohup.sh \
  > ./logs/nohup_task_pool_guard.out 2>&1 < /dev/null &
```

Stop task-pool units:

```bash
./trainers/main/stop_task_pool.sh
```

Stop units and also kill leftover child trainers:

```bash
./trainers/main/stop_task_pool.sh --kill-children
```

## 4) Dry Run

Print commands without executing:

```bash
./trainers/main/start_task_pool.sh --pool-kind all --dry-run
```

## 5) Output Paths

- THGFM manifests: `trainers/main/manifests/`
- THGFM summary CSV: `trainers/main/summaries/`
- Pool logs: `OAG/logs/`
- Model checkpoints and per-run metrics: under each model directory (for example `OAG/model_save_shared/.../metrics/`)
