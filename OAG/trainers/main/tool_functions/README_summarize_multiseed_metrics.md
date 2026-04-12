# README: `summarize_multiseed_metrics.py`

This document explains how to use `summarize_multiseed_metrics.py`.

## What this script does

`summarize_multiseed_metrics.py` aggregates metrics across multiple JSONL experiment files (typically different seeds).

It computes:

- **Mean ± std** for `ndcg`
- **Mean ± std** for `mrr`
- Per-run details (seed, ndcg, mrr, file path)

The script reads events from each JSONL file and selects one target event.

---

## Supported events

Use `--event` to choose one of:

- `best_test_(highest_valid_ndcg)` (default)
- `best_test_(lowest_valid_loss)`
- `last_test`

---

## Prerequisites

- Python 3
- JSONL metric files produced by your training scripts

Recommended run directory:

```bash
cd /home/yxpeng/Projects/Papers/2026/THGFM-Dual-Branch-Temporal-Heterogeneous-Graph-Fusion-Model/OAG
```

---

## Basic usage

```bash
python summarize_multiseed_metrics.py \
  --glob "model_save/drsf_gate/metrics/AD_XLNet_drsf_gate_s*.jsonl"
```

This uses the default event `best_test_(highest_valid_ndcg)`.

---

## Common examples

### 1) Aggregate AD (best by valid NDCG)

```bash
python summarize_multiseed_metrics.py \
  --glob "model_save/drsf_gate/metrics/AD_XLNet_drsf_gate_s*.jsonl" \
  --event "best_test_(highest_valid_ndcg)"
```

### 2) Aggregate PF-L2 (best by lowest valid loss)

```bash
python summarize_multiseed_metrics.py \
  --glob "model_save/drsf_gate/metrics/PF_XLNet_L2_drsf_gate_L2_s*.jsonl" \
  --event "best_test_(lowest_valid_loss)"
```

### 3) Aggregate PV (last test)

```bash
python summarize_multiseed_metrics.py \
  --glob "model_save/drsf_gate/metrics/PV_XLNet_drsf_gate_s*.jsonl" \
  --event "last_test"
```

---

## How seed is detected

The script tries, in order:

1. `run_start` event field `seed` in JSONL
2. Filename suffix pattern like `_s43.jsonl`
3. Fallback to `unknown`

---

## Error cases

- No files matched:
  - `No files matched: <glob>`
- Files matched but selected event missing:
  - `Matched files found, but no '<event>' entries were found.`

---

## Tips

- Keep naming consistent (e.g., `_s43`, `_s44`, `_s45`) to simplify aggregation.
- Use separate summaries for H/L selection criteria:
  - H: `best_test_(highest_valid_ndcg)`
  - L: `best_test_(lowest_valid_loss)`
