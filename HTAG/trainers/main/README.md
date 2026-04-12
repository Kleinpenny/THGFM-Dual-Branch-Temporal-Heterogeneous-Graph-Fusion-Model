# HTAG Main Trainers: THGFM

Training entry points for HTAG paper–venue experiments:

- `dblp` → `DBLP_PV/data/shared_pv_base_HTAG_dblp`
- `arxiv` → `Arxiv_PV/data/shared_pv_base_HTAG_arxiv`

Default convolution in code is **`rtta`** (legacy `hgt` / `HGTConv` was removed from `THGFM`).

---

## 1) Weights & Biases (wandb) Setup

THGFM trainers accept `--use_wandb` and related flags (see `train_paper_venue_shared.py --help`, `build_task_pool_manifest.py --wandb`, etc.).

### 1.1 Install and login

```bash
pip install wandb
wandb login
```

### 1.2 (Optional) export environment variables

```bash
export WANDB_API_KEY="your_api_key"
export WANDB_PROJECT="THGFM"
export WANDB_ENTITY="your_team_or_user"   # optional
export WANDB_MODE="online"                # online | offline | disabled
```

---

## 2) THGFM Variant Ablation on DBLP + Arxiv

### 2.1 Main scripts

- Manifest builder: `trainers/main/build_task_pool_manifest.py`
- Scheduler: `trainers/main/run_systemd_task_pool.py` (same pattern as OGBN `trainers/main`)
- Bash launcher: `trainers/main/start_task_pool.sh`
- Stop script: `trainers/main/stop_task_pool.sh`
- Task-pool worker entry (manifest uses these; thin wrappers over the shared trainers):
  - `DBLP_PV/train_paper_venue_shared_task_pool.py`
  - `Arxiv_PV/train_paper_venue_shared_task_pool.py`
- Underlying trainers (single runs or debugging):
  - `DBLP_PV/train_paper_venue_shared.py`
  - `Arxiv_PV/train_paper_venue_shared.py`

**GPU placement:** When the scheduler is started with `--gpu-ids` and `--max-per-gpu`, it **rewrites** each task’s `--cuda` argument before `systemd-run`. The `--cuda` value written into the manifest is only a placeholder (default `0` from `build_task_pool_manifest.py --cuda`). Do not run two schedulers against overlapping `unit_name` sets.

**W&B in manifests:** Add `--wandb` when generating the manifest, plus `--wandb-project`, `--wandb-group`, etc. as needed.

### 2.2 Supported conv variants

- `rtta`
- `ssta`
- `drsf`
- `drsf_tcngsf`
- `drsf_tcnglf`

### 2.3 Recommended one-command run (bash launcher)

From HTAG root:

```bash
bash ./trainers/main/start_task_pool.sh \
  --datasets dblp arxiv \
  --convs rtta ssta drsf drsf_tcngsf drsf_tcnglf \
  --seeds 43 44 45 \
  --gpu-ids 0 1 2 3 \
  --max-running 8 \
  --max-per-gpu 2 \
  --n-pool 8 \
  --cpu-threads 1 \
  --poll-seconds 20 \
  --max-load1 75 \
  --max-gpu-util 0
```

This command:

1. Generates a timestamped manifest JSONL under `trainers/main/manifests/`
2. Starts the systemd-backed task pool (`run_systemd_task_pool.py`) in the background
3. Records the scheduler PID in `trainers/main/runtime/task_pool.pid` and appends logs to `trainers/main/runtime/task_pool_scheduler.log`

Environment overrides (optional, same idea as OGBN):

- `CONDA_SH` — path to `conda.sh`
- `CONDA_ENV` — conda env name (default `thgfm`)

### 2.4 Foreground / dry-run examples

Foreground:

```bash
bash ./trainers/main/start_task_pool.sh --foreground
```

Dry run:

```bash
bash ./trainers/main/start_task_pool.sh --dry-run
```

Use an existing manifest (no auto-generate):

```bash
bash ./trainers/main/start_task_pool.sh --manifest ./trainers/main/manifests/your_run.jsonl
```

Generate a manifest manually (example with W&B):

```bash
python ./trainers/main/build_task_pool_manifest.py \
  --datasets dblp arxiv \
  --convs rtta drsf \
  --seeds 43 \
  --wandb \
  --wandb-group r001_s001_seed43 \
  --wandb-tags thgfm ablation pv \
  --output ./trainers/main/manifests/custom.jsonl
```

### 2.5 Stop running THGFM ablation jobs

```bash
bash ./trainers/main/stop_task_pool.sh
```

This stops the scheduler (if a pid file exists), then stops and `reset-failed` on `thgfm-pv-dblp-*` and `thgfm-pv-arxiv-*` systemd units.

Force cleanup leftover trainer processes (after units stop):

```bash
bash ./trainers/main/stop_task_pool.sh --kill-children
```

Only stop the scheduler, leave systemd units alone:

```bash
bash ./trainers/main/stop_task_pool.sh --skip-units
```

---

## 3) Direct Single-Run Commands (without pool)

These are useful for quick debugging.

### 3.1 DBLP THGFM variant (example: `drsf_tcngsf`)

```bash
python ./DBLP_PV/train_paper_venue_shared.py \
  --shared_base_dir ./DBLP_PV/data/shared_pv_base_HTAG_dblp \
  --model_dir ./model_save_shared \
  --task_name PV_HTAG_DBLP_drsf_tcngsf_r111_s111_s43 \
  --conv_name drsf_tcngsf \
  --cuda 0 \
  --n_pool 8 \
  --seed 43 \
  --deterministic
```

### 3.2 Arxiv THGFM variant (example: `drsf_tcngsf`)

```bash
python ./Arxiv_PV/train_paper_venue_shared.py \
  --shared_base_dir ./Arxiv_PV/data/shared_pv_base_HTAG_arxiv \
  --model_dir ./model_save_shared \
  --task_name PV_HTAG_Arxiv_drsf_tcngsf_r111_s111_s43 \
  --conv_name drsf_tcngsf \
  --cuda 1 \
  --n_pool 8 \
  --seed 43 \
  --deterministic
```

---

## 4) Output and Logs

- THGFM task-pool logs: `logs/task_pool/`
- Scheduler runtime files:
  - `trainers/main/runtime/task_pool_scheduler.log` (log tail from background scheduler)
  - `trainers/main/runtime/task_pool.pid` (current scheduler pid; legacy `task_pool_scheduler.pid` is still honored by `stop_task_pool.sh` if present)
- Model outputs are controlled by `--model_dir` (typical THGFM ablation root: `model_save_shared`)
