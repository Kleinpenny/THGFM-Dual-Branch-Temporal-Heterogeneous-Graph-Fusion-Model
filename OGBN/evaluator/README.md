# OGBN Evaluator

This directory contains evaluation utilities for `ogbn-mag` `paper -> venue` models, using the official OGB `Evaluator(name="ogbn-mag")` accuracy.

## Files

- `eval_test_accuracy.py`  
  Loads a full `nn.Sequential` checkpoint (GNN + classifier) and runs batched inference over one or more splits.

## Evaluation Data Source

By default, the evaluator reads from the shared base:

- `OGBN/data/shared_pv_base_ogbn_mag_graph/meta.json`
- `labels/<split>__paper_ids.npy`, `__times.npy`, `__venue_ids.npy` for `train` / `valid` / `test`

Notes:

- Target nodes are defined by `paper_id + year(time)`.
- Labels are mapped from `venue_id` to class indices.
- The OGB evaluator computes `acc` from `y_true` / `y_pred` with shape `N x 1`.

## Quick Start

Run from `OGBN/`:

```bash
cd OGBN
python evaluator/eval_test_accuracy.py --split test --batch_size 256 --clip 0.25
```

Evaluate **valid** and **test** in one run (prints `valid_acc=` and `test_acc=`):

```bash
python evaluator/eval_test_accuracy.py --splits valid,test --batch_size 256
```

## Common Arguments

- `--checkpoint`: checkpoint path to evaluate  
  Default: `OGBN/trainers/main/model_save/rtta/PV_ogbn_mag`
- `--shared_base_dir`: shared base path  
  Default: `OGBN/data/shared_pv_base_ogbn_mag_graph`
- `--split`: single split (`test` / `valid` / `train`) when `--splits` is **not** set (default: `test`)
- `--splits`: comma-separated list, e.g. `valid,test`; when set, **overrides** `--split`
- `--max_test_rows`: evaluate only the first N rows **per split** (debug / smoke)
- `--batch_size`: evaluation batch size
- `--sample_depth`: subgraph sampling depth
- `--sample_width`: per-layer per-type sampling width
- `--cuda`: GPU index (`-1` for CPU)
- `--seed`: random seed (neighbor sampling only)
- `--clip`: training clip value (recorded in output JSON only)
- `--save_dir`: JSON output directory (default: `OGBN/evaluator/results`)

## Output Files

Each run writes one JSON under `--save_dir` (layout from `eval_result_layout.py`):

- Single split: `eval_<split>_<checkpoint_name>_seed<seed>_<timestamp>.json`
- Multiple splits: `eval_<split1>_<split2>_..._<checkpoint_name>_seed<seed>_<timestamp>.json`

### JSON Fields (multi-split)

- `splits`: list of evaluated splits
- `per_split`: per-split `accuracy`, `total_samples`, `full_split_size`
- `valid_acc` / `test_acc`: convenience fields (`null` if that split was not run)
- `accuracy`: same as test accuracy when `test` is included; otherwise the last / only split’s accuracy
- `checkpoint`, `shared_base_dir`, `batch_size`, `sample_depth`, `sample_width`, `seed`
- `n_hid` / `n_heads` / `n_layers`: extracted from the loaded model
- `clip`: clip value from CLI (can be null)
- `saved_at`, `results_dir`

## Repro Command (Current Setup)

```bash
cd OGBN
python evaluator/eval_test_accuracy.py --split test --batch_size 256 --clip 0.25
```
