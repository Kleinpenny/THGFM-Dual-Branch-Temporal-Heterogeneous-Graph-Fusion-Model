# HTAG DBLP Data Preprocess

This directory is paired with `DBLP_PV/data/`.  
The scripts here read/write intermediate files and the shared base under `DBLP_PV/data/` by default.

## Scripts in this directory

- `build_dblp_graph_from_pkl.py`  
  Builds a `Graph` from `dblp.pkl`.  
  Default outputs:
  - `data/graph_dblp_HTAG.pk`
  - `data/dblp_splits/`
- `build_shared_pv_base.py`  
  Builds the shared Paper-Venue base from the graph and splits.  
  Default output:
  - `data/shared_pv_base_HTAG_dblp`

## Download HTAG data (Hugging Face)

If you execute commands inside `HTAG/DBLP_PV/data_preprocess`, you can directly download the released HTAG dataset with:

```python
from huggingface_hub import snapshot_download

# Download all
snapshot_download(
    repo_id="Cloudy1225/HTAG",
    repo_type="dataset",
    local_dir="../../../data/HTAG",
)
```

With this layout, DBLP data is expected at:

- `../../../data/HTAG/dblp/dblp.pkl` (relative to `HTAG/DBLP_PV/data_preprocess`)

## Input path behavior (actual code defaults)

In `build_dblp_graph_from_pkl.py`, the default `--input_pkl` points to:

- `../../data/HTAG/dblp/dblp.pkl` (relative to `HTAG/DBLP_PV`)

If your file is elsewhere, pass it explicitly:

```bash
python data_preprocess/build_dblp_graph_from_pkl.py --input_pkl /absolute/path/to/dblp.pkl
```

## 1) Build DBLP graph

Run inside `HTAG/DBLP_PV/data_preprocess`:

```bash
cd "/path/to/HTAG/DBLP_PV/data_preprocess"
python build_dblp_graph_from_pkl.py
```

If needed, override `--input_pkl`, `--out_graph`, and `--split_dir`.

## 2) Build shared PV base (DBLP)

```bash
cd "/path/to/HTAG/DBLP_PV/data_preprocess"
python build_shared_pv_base.py
```

If needed, override `--graph_file`, `--split_dir`, and `--out_dir`.

## Quick sanity check

```bash
cd "/path/to/HTAG/DBLP_PV"
python - <<'PY'
import json, os
base = os.path.abspath(os.path.join("data", "shared_pv_base_HTAG_dblp"))
meta = os.path.join(base, "meta.json")
print("meta exists:", os.path.exists(meta))
with open(meta, "r", encoding="utf-8") as f:
    m = json.load(f)
print("node_types:", m["node_types"])
print("feature_dims:", m["feature_dims"])
PY
```

## Note for HTAG ArXiv preprocess

The ArXiv pipeline is in `HTAG/Arxiv_PV/data_preprocess/`.

- `build_arxiv_graph_from_pkl.py` default input is:
  - `HTAG/Arxiv_PV/data/HTAG/arxiv/arxiv.pkl`
- If you downloaded to (from `HTAG/Arxiv_PV/data_preprocess`):
  - `../../../data/HTAG`
  then pass:

```bash
cd "/path/to/HTAG/Arxiv_PV/data_preprocess"
python build_arxiv_graph_from_pkl.py \
  --input_pkl "../../../data/HTAG/arxiv/arxiv.pkl"
```
