# `trainers/main`: task pool usage

This document is the operation guide for the ablation-study task pool.

The default setup is:

- keep up to `16` training parent processes active
- use dynamic GPU scheduling on `0 1 2 3` with `max 4` tasks per GPU
- launch the next pending ablation automatically when one task finishes
- stop only after the whole manifest is consumed

Current task counts:

- `AD`: `23`
- `PF-L1`: `23`
- `PF-L2`: `23`
- `PV`: `23`

## Run in Order

### 1) Enter the project and activate conda

```bash
source /home/yxpeng/anaconda3/etc/profile.d/conda.sh
conda activate thgfm
cd /home/yxpeng/Projects/Papers/2026/THGFM-Dual-Branch-Temporal-Heterogeneous-Graph-Fusion-Model/OAG
```

### 2) Optional: set wandb environment variables

```bash
export WANDB_API_KEY='your_wandb_api_key'
export WANDB_PROJECT=THGFM
export WANDB_ENTITY=''
export WANDB_MODE=online
```

### 3) Optional: dry run first

This prints the generated commands without actually submitting jobs.

```bash
./trainers/main/start_task_pool.sh --dry-run
```

### 4) Launch the full task pool

This is the default command you should use.

```bash
./trainers/main/start_task_pool.sh
```

### 5) Monitor services and logs

```bash
systemctl list-units --all "ad-*" "pf-l1-*" "pf-l2-*" "pv-*"
```

```bash
ps -ef | rg "train_author_disambiguation_shared.py|train_paper_field_shared.py|train_paper_venue_shared.py"
```

```bash
tail -f ./logs/train_author_disambiguation_xlnet_shared_rtta_s43.log
```

### 6) Stop everything

```bash
./trainers/main/stop_task_pool.sh
```

If needed, also kill leftover child processes:

```bash
./trainers/main/stop_task_pool.sh --kill-children
```


#############################################################
## Run with nohup
#############################################################


If you want the task pool to continue running after closing the terminal, use `nohup`.

### Test pool in background

```bash
nohup ./trainers/main/test/start_task_pool_test.sh --max-running 1 \
  > ./logs/task_pool_test/nohup_test.out 2>&1 < /dev/null &
echo $!
```

### Full pool in background

```bash
nohup ./trainers/main/start_task_pool.sh \
  > ./logs/nohup_task_pool.out 2>&1 < /dev/null &
echo $!
```

### Watch nohup output

```bash
tail -f ./logs/task_pool_test/nohup_test.out
```

```bash
tail -f ./logs/nohup_task_pool.out
```

#############################################################
临时的，只有当前GPU中跑的任务少于三个时，才会启动start_task_pool.sh
#############################################################


## Auto relaunch guard

If you want a background guard that periodically checks your GPU `python` process count and automatically restarts the full task pool when the count drops below `3`, use:

```bash
nohup ./trainers/main/watch_task_pool_nohup.sh \
  > ./logs/nohup_task_pool_guard.out 2>&1 < /dev/null &
echo $!
```

Watch the guard log:

```bash
tail -f ./logs/nohup_task_pool_guard.out
```

Run one check only:

```bash
./trainers/main/watch_task_pool_nohup.sh --once
```

Preview without actually launching the pool:

```bash
./trainers/main/watch_task_pool_nohup.sh --once --dry-run
```

Useful options:

```bash
./trainers/main/watch_task_pool_nohup.sh --threshold 3 --interval-seconds 60 --max-running 16
```

Notes:

- The guard checks the number of GPU `python` processes owned by `yxpeng`.
- When the count is below the threshold and no pool manager is already running, it launches:

```bash
nohup ./trainers/main/start_task_pool.sh \
  > ./logs/nohup_task_pool.out 2>&1 < /dev/null &
```

## Check sudo NOPASSWD

The task pool uses:

- `sudo systemd-run`
- `sudo systemctl`

So the important thing is whether these two commands are passwordless.

If you already added:

```text
yxpeng ALL=(root) NOPASSWD: /usr/bin/systemd-run, /usr/bin/systemctl
```

then verify it in a fresh session like this:

```bash
which systemd-run
which systemctl
sudo -k
sudo -n /usr/bin/systemctl --version >/dev/null && echo "systemctl NOPASSWD OK"
sudo -n /usr/bin/systemd-run --unit=nopasswd-check-$$ /bin/true >/dev/null && echo "systemd-run NOPASSWD OK"
```

You may still be asked for a password when running:

```bash
sudo su
```

This is normal. The `NOPASSWD` rule only applies to `/usr/bin/systemd-run` and `/usr/bin/systemctl`, not to `sudo su`.

## Common Commands

### Run only AD

```bash
./trainers/main/start_task_pool.sh --pool-kind ad --max-running 16
```

### Run with multiple seeds

```bash
./trainers/main/start_task_pool.sh --max-running 16 --seeds 43 44 45
```

### Override dynamic GPU scheduling

```bash
./trainers/main/start_task_pool.sh \
  --max-running 16 \
  --gpu-ids 0 1 2 3 \
  --max-per-gpu 4
```

### Use an existing manifest

```bash
./trainers/main/start_task_pool.sh \
  --manifest ./trainers/main/manifests/task_pool_all.jsonl \
  --max-running 16
```

## Manual Commands

If you want to generate the manifest separately:

```bash
python ./trainers/main/build_task_pool_manifest.py --write-defaults
```

If you want to generate a custom combined manifest:

```bash
python ./trainers/main/build_task_pool_manifest.py \
  --families ad pf-l1 pf-l2 pv \
  --seeds 43 \
  --output ./trainers/main/manifests/task_pool_all.custom.jsonl
```

If you want to run the scheduler directly:

```bash
python ./trainers/main/run_systemd_task_pool.py \
  --manifest ./trainers/main/manifests/task_pool_all.jsonl \
  --max-running 16 \
  --poll-seconds 20 \
  --gpu-ids 0 1 2 3 \
  --max-per-gpu 4
```

## Notes

- Manifests are stored under `trainers/main/manifests/`.
- `task_pool.jsonl` is the AD-only manifest.
- `task_pool_all.jsonl` contains `AD + PF-L1 + PF-L2 + PV`.
- Aggregation CSVs are stored under `trainers/main/summaries/`.
- The task pool only limits jobs launched by this scheduler. Other training jobs running outside this pool still count against your GPUs.
- If a task fails, the scheduler will report it in the final summary.
- Failed tasks are not retried automatically.
