# HTAG ArXiv Data Preprocess

This directory is paired with `Arxiv_PV/data/`.  
The scripts here read/write intermediate files and the shared base under `Arxiv_PV/data/` by default.

## Scripts in this directory

- `build_arxiv_graph_from_pkl.py`  
  Builds a `Graph` from `arxiv.pkl`.  
  Default outputs:
  - `data/graph_arxiv_HTAG.pk`
  - `data/arxiv_splits/`
- `build_shared_pv_base.py`  
  Builds the shared Paper-Venue base from the graph and splits.  
  Default output:
  - `data/shared_pv_base_HTAG_arxiv`

## Download HTAG data (Hugging Face)

If you execute commands inside `HTAG/Arxiv_PV/data_preprocess`, you can directly download the released HTAG dataset with:

```python
from huggingface_hub import snapshot_download

# Download all
snapshot_download(
    repo_id="Cloudy1225/HTAG",
    repo_type="dataset",
    local_dir="../../../data/HTAG",
)
```

With this layout, ArXiv data is expected at:

- `../../../data/HTAG/arxiv/arxiv.pkl` (relative to `HTAG/Arxiv_PV/data_preprocess`)

## Input path behavior (actual code defaults)

In `build_arxiv_graph_from_pkl.py`, the default `--input_pkl` points to:

- `Arxiv_PV/data/HTAG/arxiv/arxiv.pkl`

If your file is elsewhere (including the repo-level `data/HTAG` path above), pass it explicitly:

```bash
python data_preprocess/build_arxiv_graph_from_pkl.py --input_pkl /absolute/path/to/arxiv.pkl
```

## 1) Build ArXiv graph

Run inside `HTAG/Arxiv_PV/data_preprocess`:

```bash
cd "/path/to/HTAG/Arxiv_PV/data_preprocess"
python build_arxiv_graph_from_pkl.py
```

Example with the repo-level downloaded data:

```bash
cd "/path/to/HTAG/Arxiv_PV/data_preprocess"
python build_arxiv_graph_from_pkl.py \
  --input_pkl "../../../data/HTAG/arxiv/arxiv.pkl"
```

If needed, also override `--out_graph` and `--split_dir`.

## 2) Build shared PV base (ArXiv)

```bash
cd "/path/to/HTAG/Arxiv_PV/data_preprocess"
python build_shared_pv_base.py
```

If needed, override `--graph_file`, `--split_dir`, and `--out_dir`.

## Quick sanity check

```bash
cd "/path/to/HTAG/Arxiv_PV"
python - <<'PY'
import json, os
base = os.path.abspath(os.path.join("data", "shared_pv_base_HTAG_arxiv"))
meta = os.path.join(base, "meta.json")
print("meta exists:", os.path.exists(meta))
with open(meta, "r", encoding="utf-8") as f:
    m = json.load(f)
print("node_types:", m["node_types"])
print("feature_dims:", m["feature_dims"])
PY
```
