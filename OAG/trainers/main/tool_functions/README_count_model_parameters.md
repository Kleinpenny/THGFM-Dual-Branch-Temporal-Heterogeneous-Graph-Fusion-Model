# README: `count_model_parameters.py`

This document explains how to use `count_model_parameters.py`.

## What this script does

`count_model_parameters.py` reports model parameter counts in three ways:

1. **Method-level checkpoint comparison** (`SSTA / RTTA / DRSF`)
   - Loads representative trained checkpoints from `model_save/`
   - Prints total params, GNN-only params, and task-head params (in millions)

2. **Checkpoint-based counting by conv name**
   - Loads one checkpoint per conv family listed in the script
   - Prints total / GNN / head params + checkpoint file size

3. **Constructed GNN-only counts**
   - Builds `GNN` models directly (without task heads)
   - Reports comparable parameter counts for:
     - `rtta`, `rtta_rota`, `ssta`, `ssta_rota`, `drsf`, `drsf_rota`, `drsf_gate`, `drsf_gate_rota`
   - Uses fixed architecture constants in the script (`n_hid`, `n_layers`, `num_types`, etc.)

---

## Prerequisites

- Python environment with this repo dependencies installed
- `torch` available
- Run from the `OAG/` directory (or pass paths accordingly).

---

## How to run

```bash
cd OAG/trainers/main/tool_functions
python count_model_parameters.py
```

---

## Output format

You will see multiple sections, including:

- Method-level table with columns:
  - `Total(M)`
  - `GNN(M)`
  - `Head(M)`
  - checkpoint path
- Conv-by-conv listing for built GNN models
- A summary table comparing embedding settings

If a checkpoint path is missing, it prints `file not found` or `N/A`.

---

## Notes

- The script is **read-only** for your checkpoints (it only loads and counts).
- File size on disk is shown for reference and is not equal to parameter count.
- Paths and constants are defined at the top of the script:
  - `CHECKPOINTS`
  - `METHOD_CHECKPOINTS`
  - `IN_DIM_XLNET`, `IN_DIM_QWEN3`, etc.

If your checkpoint layout changes, update those dictionaries first.
