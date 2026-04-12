# THGFM-Dual-Branch-Temporal-Heterogeneous-Graph-Fusion-Model

Implementation for **THGFM** (Dual-Branch Temporal Heterogeneous Graph Fusion Model).

## Overview

This codebase reproduces the experiments reported in the accompanying submission. **THGFM** targets **temporal heterogeneous graphs**, where node types, relation types, and timestamps jointly govern interaction patterns. The implementation couples a heterogeneous graph neural backbone (including temporal encoding options used in the paper) with a **dual-branch** fusion mechanism that separates complementary views of temporal structure; concrete layer variants and ablation presets are selectable via configuration (see `THGFM/` and trainer entry points).

The repository is **partitioned by benchmark family** so that data layout, training scripts, and evaluation protocols stay self-contained:

| Directory | Role |
|-----------|------|
| **HTAG** | HTAG-style paper–venue (and related) setups on ArXiv and DBLP-derived graphs. |
| **OAG** | Open Academic Graph (OAG) tasks, including paper–venue, paper–field (L1/L2), and author disambiguation. |
| **OGBN** | OGBN-MAG node property prediction under the same training/evaluation framework. |

Within each family, the **layout is parallel**:

1. **`data_preprocess/`** — builds shared graph caches and task-specific inputs from raw or preprocessed sources.
2. **`THGFM/`** — model definition, heterogeneous convolutions / fusion blocks, and data loading helpers.
3. **`trainers/main/`** — training loops, task-pool manifests, and multi-seed or distributed launch utilities where applicable.
4. **`evaluator/`** — standardized metrics and scripts for test-time evaluation and result aggregation.

Together, these pieces are intended to make **training, ablations, and evaluation** traceable without relying on undisclosed local paths or private assets (see **Setup** and **Datasets** for dependencies and public data sources).

**Double-blind review.** If this repository is shared with reviewers, avoid author names, affiliations, or other identifying metadata in the default branch, public issues, and commit messages.

## Setup

**Requirements (as pinned in the repo).** Python **3.11**, **PyTorch 2.4.0** with **CUDA 12.1**, **PyTorch Geometric 2.7.0**, **DGL 2.4.0+cu121**, and **OGB** (see `requirements.txt` / `environment.yml` for full pins and wheel index URLs).

**Install (from the repository root).**

```bash
# Recommended: CUDA 12.1 machine matching the PyTorch + PyG + DGL wheels in requirements.txt
python -m pip install -r requirements.txt
```

Conda alternative:

```bash
conda env create -f environment.yml
conda activate thgfm
```

**Optional.** [Weights & Biases](https://wandb.ai/) (`wandb`) is used by training scripts when `--use_wandb` is set; configure with `wandb login` or disable via flags documented in each benchmark’s `trainers/main/README.md`.

**Smoke check (used to validate this README).** After install, `python -c "import torch, torch_geometric"` should succeed, and the CLI entry points below should print help without error when run from the working directories indicated (e.g. `python DBLP_PV/train_paper_venue_shared.py --help` from `HTAG/`).

## Datasets and where to put downloads

All paths below are **relative to the repository root** unless noted.

### HTAG (paper → venue on ArXiv and DBLP)

| Step | Put files here | Scripts (run from noted cwd) |
|------|----------------|------------------------------|
| Upstream pickles | `data/HTAG/dblp/dblp.pkl` and `data/HTAG/arxiv/arxiv.pkl` (recommended), or paths you pass with `--input_pkl` | Download from [HTAG](https://github.com/Cloudy1225/HTAG/tree/main) or Hugging Face per `HTAG/*/data_preprocess/README.md`. |
| Intermediate graph + splits | `HTAG/DBLP_PV/data/` and `HTAG/Arxiv_PV/data/` | From `HTAG/DBLP_PV/data_preprocess/`: `build_dblp_graph_from_pkl.py`; from `HTAG/Arxiv_PV/data_preprocess/`: `build_arxiv_graph_from_pkl.py`. |
| Shared bases for training | `HTAG/DBLP_PV/data/shared_pv_base_HTAG_dblp`, `HTAG/Arxiv_PV/data/shared_pv_base_HTAG_arxiv` | Same folders: `build_shared_pv_base.py`. |

### OAG-CS (PV, PF L1/L2, author disambiguation)

| Step | Put files here | Scripts |
|------|----------------|---------|
| Raw MAG CS TSVs | `data/raw_data/MAG_0919_CS/` (or any dir you pass as `--input_dir`) | [OAG-CS Drive](https://drive.google.com/open?id=1yDdVaartOCOSsQlUZs8cJcAUhmvRiBSz). |
| Graph pickles | e.g. `data/graph_CS_XLNet.pk` via `--output_dir ../data` from `OAG/` | `OAG/data_preprocess/preprocess_OAG_XLNet.py` (or Qwen3 variants). |
| Shared bases | e.g. `data/shared_ad_base_CS_XLNet`, `data/shared_pv_base_CS_XLNet`, `data/shared_pf_base_CS_XLNet_L1`, … | `OAG/data_preprocess/build_shared_ad_base.py`, `build_shared_pv_base.py`, `build_shared_pf_base.py` (run from `OAG/`; see `OAG/data_preprocess/README.md`). |

### OGBN-MAG

| Step | Put files here | Scripts |
|------|----------------|---------|
| Official OGB extract | `data/ogbn_mag/` with `raw/`, `split/`, `mapping/` | [ogbn-mag](https://ogb.stanford.edu/docs/nodeprop/#ogbn-mag). |
| Large graph pickle | `OGBN/data/graph_ogbn_mag_w2v.pk` (default) | From repo root: `python OGBN/data_preprocess/preprocess_ogbn_mag_w2v.py --ogbn_mag_dir data/ogbn_mag --output_dir OGBN/data`. |
| Shared PV base | `OGBN/data/shared_pv_base_ogbn_mag_graph` | `python OGBN/data_preprocess/build_shared_pv_base_ogbn_mag.py` (see `OGBN/data_preprocess/README.md`). |

## Usage

Always `cd` into **`HTAG/`**, **`OAG/`**, or **`OGBN/`** before running Python entry points so that local `THGFM` packages and relative paths resolve as in the smoke tests.

**OAG paths.** Preprocess scripts in this repo typically write shared bases under the **repository root** `data/` (see the OAG-CS table above). After `cd OAG`, pass those bases as **`../data/...`** (not `./data/...`, unless you copy or symlink them under `OAG/data/`).

### Train

| Benchmark | Typical entry | More detail |
|-----------|---------------|-------------|
| **HTAG** | `DBLP_PV/train_paper_venue_shared.py`, `Arxiv_PV/train_paper_venue_shared.py`; task pools via `trainers/main/start_task_pool.sh` | `HTAG/trainers/main/README.md` |
| **OAG** | `trainers/main/train_paper_venue_shared.py`, `train_paper_field_shared.py`, `train_author_disambiguation_shared.py`; pools via `trainers/main/start_task_pool.sh` | `OAG/trainers/main/README.md` |
| **OGBN** | `trainers/main/train_paper_venue_shared.py`; pools via `trainers/main/start_task_pool.sh` | `OGBN/trainers/main/README.md` |

**Minimal single-run examples (after shared bases exist):**

```bash
# HTAG / DBLP
cd HTAG
python ./DBLP_PV/train_paper_venue_shared.py \
  --shared_base_dir ./DBLP_PV/data/shared_pv_base_HTAG_dblp \
  --model_dir ./model_save_shared \
  --task_name HTAG_DBLP_rtta_s43 \
  --conv_name rtta \
  --cuda 0 \
  --seed 43 --deterministic

# OAG / paper–venue (shared base at repo root data/; see "OAG paths" above)
cd OAG
python ./trainers/main/train_paper_venue_shared.py \
  --shared_base_dir ../data/shared_pv_base_CS_XLNet \
  --model_dir ./model_save_shared \
  --task_name PV_XLNet_rtta_s43 \
  --conv_name rtta \
  --cuda 0 \
  --seed 43 --deterministic

# OGBN-MAG
cd OGBN
python ./trainers/main/train_paper_venue_shared.py \
  --shared_base_dir ./data/shared_pv_base_ogbn_mag_graph \
  --model_dir ./model_save_shared \
  --task_name PV_ogbn_mag_rtta_s43 \
  --conv_name rtta \
  --cuda 0 \
  --seed 43 --deterministic
```

Use `-h` / `--help` on each script for the authoritative argument list (`--conv_name` accepts THGFM variants listed in the OAG/OGBN parsers).

### Evaluate

| Benchmark | Script | Notes |
|-----------|--------|--------|
| **HTAG** | `evaluator/eval_pv_htag_dblp.py`, `evaluator/eval_pv_htag_arxiv.py` | Require `--checkpoint` (`.pt`). |
| **OGBN** | `evaluator/eval_test_accuracy.py` | Official OGB `ogbn-mag` accuracy; use `--splits valid,test` for `valid_acc` + `test_acc` in one run. |
| **OAG** | `evaluator/eval_pv_test_ndcg_mrr.py`, `eval_pf_l1_test_ndcg_mrr.py`, `eval_pf_l2_test_ndcg_mrr.py`, `run_eval_all_oag_taskpool.py` | Full `nn.Sequential` (GNN + head) checkpoints only. |

**Examples:**

```bash
cd HTAG
python ./evaluator/eval_pv_htag_dblp.py \
  --checkpoint /path/to/your_checkpoint.pt \
  --cuda 0

cd OGBN
# Test split only (default when --splits is omitted)
python ./evaluator/eval_test_accuracy.py \
  --split test \
  --checkpoint /path/to/your_checkpoint.pt \
  --shared_base_dir ./data/shared_pv_base_ogbn_mag_graph \
  --batch_size 256

# Valid + test in one run (OGB official accuracy for each; stdout + JSON include valid_acc / test_acc)
python ./evaluator/eval_test_accuracy.py \
  --splits valid,test \
  --checkpoint /path/to/your_checkpoint.pt \
  --shared_base_dir ./data/shared_pv_base_ogbn_mag_graph \
  --batch_size 256 \
  --seed 43 \
  --cuda 0

cd OAG
python ./evaluator/eval_pv_test_ndcg_mrr.py \
  --checkpoint /path/to/PV_checkpoint_or_dir \
  --shared_base_dir ../data/shared_pv_base_CS_XLNet \
  --cuda 0
```

**Bundled OGBN-MAG weight (`Best_Model_Save/`).** To evaluate the shipped **PV OGBN-MAG** checkpoint on **valid** and **test** (full splits; long-running), from the repository root:

```bash
python OGBN/evaluator/eval_test_accuracy.py \
  --checkpoint ./Best_Model_Save/PV_OGBN_MAG_tcngsf_r111_s111_s43_lowest_valid_loss \
  --shared_base_dir ./OGBN/data/shared_pv_base_ogbn_mag_graph \
  --splits valid,test \
  --seed 43 \
  --batch_size 256 \
  --cuda 0 \
  --save_dir ./Best_Model_Save/eval_results/OGBN \
  --model_subdir PV_OGBN_MAG_tcngsf_r111_s111_s43_valid_test_full
```

Summary JSON is written under `Best_Model_Save/eval_results/OGBN/OGBN-MAG/<model_subdir>/` as `eval_valid_test_<checkpoint>_seed<seed>_*.json` with fields `valid_acc`, `test_acc`, `per_split`, and `accuracy` (equals test accuracy when `test` is included).

JSON outputs default under each `evaluator/results/` tree; see the per-benchmark `evaluator/README.md` where present.

#### Bundled best checkpoints: `scripts/eval_best_model_save.py`

The directory **`Best_Model_Save/`** holds six reference weights (HTAG Arxiv + DBLP PV, OAG PV + PF L1 + PF L2, OGBN-MAG PV). From the **repository root**, `scripts/eval_best_model_save.py` runs every matching evaluator as a subprocess, **in parallel** over all **(checkpoint × seed)** pairs, then prints a **summary**.

| Behavior | Notes |
|----------|--------|
| **Concurrency** | `ThreadPoolExecutor`; limit with `--max-workers` (default `6`). Many workers on **one** GPU share the same device and may contend or OOM—reduce `--max-workers` if needed. |
| **Multi-seed** | `--seeds 43,44,45` (comma-separated). If omitted, a single `--seed` is used (default `43`). |
| **Summary stats** | For each benchmark and metric, reports **mean ± (max − min)** over seeds when more than one seed is used; otherwise a single value. |
| **Outputs** | By default, JSON under `Best_Model_Save/eval_results/` (OAG / OGBN / HTAG subfolders). With multiple seeds, `--model_subdir` is `{checkpoint_stem}_seed{seed}` to avoid overwrites. HTAG **smoke** mode uses `--no_save` (stdout JSON only). **OGBN-MAG** jobs pass `--splits valid,test`, so each run reports **valid_acc** and **test_acc** (and writes one combined JSON per seed). |
| **Smoke** | `--smoke` sets `--max_test_rows 32` and HTAG `--eval_rounds 1 --no_save` for a quick wiring check. |
| **Evaluator flags** | Individual evaluators also accept `--max_test_rows` for partial splits; see `HTAG/OAG/OGBN/evaluator/` scripts. |
| **Live progress** | By default each subprocess uses a **pseudo-TTY** on Linux/macOS so **tqdm** bars stay enabled; output streams to the console (parallel jobs may interleave). Raw per-task copies go under `<save_root>/_eval_live_logs/<timestamp>/`. Use `--no-live-progress` to buffer until a job exits, or `--live-log-dir PATH` to choose the log directory. |

**Examples (repository root):**

```bash
# Quick smoke (small row cap)
python scripts/eval_best_model_save.py --smoke --seeds 43 --cuda 0 --max-workers 6

# Full evaluation, three seeds, high parallelism (adjust if GPU memory is tight)
python scripts/eval_best_model_save.py --seeds 43,44,45 --cuda 3 --max-workers 18

# Subset of jobs
python scripts/eval_best_model_save.py --seeds 43 --only htag_arxiv,oag_pv --cuda 0

# Print commands only
python scripts/eval_best_model_save.py --dry-run --seeds 43,44
```

`WANDB_MODE` defaults to `disabled` for these subprocesses unless you change the environment.
