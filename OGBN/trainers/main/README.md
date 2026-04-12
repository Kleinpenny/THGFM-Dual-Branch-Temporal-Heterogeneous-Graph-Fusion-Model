# OGBN paper->venue training (`trainers/main`)

This directory contains the runnable entry points used for:

- THGFM variant / ablation experiments on OGBN-MAG paper→venue (PV), single run and task-pool mode,
- task scheduling and result aggregation.

Legacy `hgt` / `HGTConv` was removed; default `--conv_name` is **`rtta`**.

## 0) Configure Weights & Biases (W&B) first

All training entry points support W&B flags (`--use_wandb`, `--wandb_project`, `--wandb_entity`, `--wandb_group`, `--wandb_run_name`, `--wandb_mode`).

```bash
# 1) Install once in your env
pip install wandb

# 2) Login (interactive)
wandb login

# Optional non-interactive login
# export WANDB_API_KEY=xxxxxxxxxxxxxxxx
# wandb login "$WANDB_API_KEY"
```

Common modes:

- `--wandb_mode online`: sync to W&B server (default),
- `--wandb_mode offline`: log locally, sync later,
- `--wandb_mode disabled`: keep `--use_wandb` off or disable logging explicitly.

## 1) Prerequisites

- Python environment with project dependencies (PyTorch, OGB package set, etc.).
- Built shared PV base (default path expected by trainers):
  - `OGBN/data/shared_pv_base_ogbn_mag_graph`
- Recommended launch location:

```bash
cd /path/to/THGFM-Dual-Branch-Temporal-Heterogeneous-Graph-Fusion-Model/OGBN
```

## 2) THGFM single-run training entry points

### 2.1 Standard shared trainer

Script: `trainers/main/train_paper_venue_shared.py`

```bash
python ./trainers/main/train_paper_venue_shared.py \
  --shared_base_dir ./data/shared_pv_base_ogbn_mag_graph \
  --model_dir ./model_save_shared \
  --task_name PV_ogbn_mag_rtta_s43 \
  --conv_name rtta \
  --cuda 0 \
  --seed 43 \
  --deterministic \
  --use_wandb --wandb_project THGFM-OGBN --wandb_group pv-ogbn-single
```

Key defaults in this script:

- `n_hid=256`, `n_layers=3`, `n_epoch=200`, `sample_width=128`, `n_pool=4`, `batch_size=256`, `repeat=2`.

## 3) THGFM variant / ablation task-pool (recommended for many conv variants)

Main scripts involved:

- `trainers/main/start_task_pool.sh` (end-to-end launcher)
- `trainers/main/build_task_pool_manifest.py` (manifest generator)
- `trainers/main/run_systemd_task_pool.py` (scheduler)
- `trainers/main/train_paper_venue_shared_task_pool.py` (trainer entry)
- `trainers/main/collect_task_pool_best_test.py` (summary collector)
- `trainers/main/stop_task_pool.sh` (stop helper)

### 3.1 Start task pool (auto-generate manifest)

```bash
./trainers/main/start_task_pool.sh \
  --conv-names rtta ssta drsf_sf drsf_tc_ngsf \
  --seeds 43 \
  --gpu-ids 0 1 2 3 \
  --max-running 8 \
  --max-per-gpu 2 \
  --n-pool 8
```

Behavior:

- Generates manifest under `trainers/main/manifests/` by default.
- Runs in background by default and writes PID/log under `trainers/main/runtime/`.
- Writes summary CSV under `trainers/main/summaries/`.

### 3.2 Stop task pool

```bash
./trainers/main/stop_task_pool.sh
```

Optional stronger cleanup:

```bash
./trainers/main/stop_task_pool.sh --kill-children
```

### 3.3 Build manifest manually (optional)

```bash
python ./trainers/main/build_task_pool_manifest.py \
  --output ./trainers/main/manifests/thgfm_custom.jsonl \
  --conv-names rtta ssta drsf_sf \
  --seeds 43 \
  --shared-base-dir ./data/shared_pv_base_ogbn_mag_graph \
  --wandb-project THGFM-OGBN
```

Then run:

```bash
./trainers/main/start_task_pool.sh --manifest ./trainers/main/manifests/thgfm_custom.jsonl
```

## 4) Outputs (logs, checkpoints, metrics)

For THGFM shared trainers, one run is stored under:

`{model_dir}/{conv_name}/nhid{n_hid}_nlayers{n_layers}_nepoch{n_epoch}/`

Typical files:

- `train_*.log`: full train log,
- `{task_name}_{tag}`: best-valid-acc checkpoint,
- `{task_name}_lowest_valid_loss_{tag}`: best-valid-loss checkpoint,
- `metrics/{task_name}.jsonl`: event-wise metrics (run start, per-epoch valid, final tests).

For task-pool runs, scheduler/runtime files are under:

- `trainers/main/runtime/`
- `trainers/main/manifests/`
- `trainers/main/summaries/`

## 5) Useful flags to remember

- `--shared_base_dir`: shared OGBN PV base path.
- `--model_dir`: root save dir.
- `--task_name`: metric/checkpoint naming prefix.
- `--conv_name`: model variant (`rtta`, `ssta`, `drsf_*`, `gcn`, `gat`, ...).
- `--seed`, `--deterministic`: reproducibility controls.
- `--n_pool`, `--n_batch`, `--batch_size`: sampling/training throughput knobs.
- `--use_wandb`, `--wandb_project`, `--wandb_entity`, `--wandb_group`, `--wandb_mode`: experiment tracking controls.

Use `-h` / `--help` on each script for the full and latest argument list.
