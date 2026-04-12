
### ogbn-mag raw data From https://ogb.stanford.edu/docs/nodeprop/ 
# ogbn-mag data preprocessing scripts

This directory contains scripts that convert **OGB’s ogbn-mag** release into the format required for THGFM training. The task is **venue (conference/journal) classification with 349 classes**; splits follow the official OGB **time-based split** (`split/time/paper/`).

## Dependencies and environment

- Python 3; align with the project `requirements.txt` where possible (e.g. `numpy`, `pandas`, `tqdm`, `dill`, `torch`, `scipy`).
- **`preprocess_ogbn_mag_w2v.py`** and **`build_shared_pv_base_ogbn_mag.py`** prepend **`OAG/`** to `sys.path` and import **`from THGFM.data import ...`** using the **THGFM** package shipped with OAG. Run commands from the repository root and ensure `OAG/THGFM` exists.
- Deserializing `graph_ogbn_mag_w2v.pk` needs the same layout; otherwise `dill.load` may fail to resolve the `THGFM` module.

## Data layout

- The **extracted ogbn-mag root** should contain `raw/`, `split/`, `mapping/`, etc., matching OGB’s layout.
- A typical symlink under the repo root: `<repo_root>/data/ogbn_mag` → your actual dataset path. Adjust `--ogbn_mag_dir` in the examples below as needed.

## Scripts

| Script | Role |
|--------|------|
| `preprocess_ogbn_mag_w2v.py` | Builds a **THGFM `Graph`** from OGB raw files; papers use **128-d word2vec**, other types get `emb` via neighbor propagation; writes **`graph_ogbn_mag_w2v.pk`**. |
| `build_shared_pv_base_ogbn_mag.py` | Exports **shared PV base** (`features/`, `relations/`, `labels/`, `meta.json`) from **`graph_ogbn_mag_w2v.pk`**, matching OAG `build_shared_pv_base.py` feature layout and CSR format; **label splits** use official OGB CSVs. |
| `build_shared_ogbn_mag_pv_base.py` | **Without a Graph pickle**: builds the same shared base directly from OGB **raw CSV edges and features**; author–paper relations are unified as **`AP_write_other`** in the script (unlike the three-way first/last/other split in the graph preprocessing path). |

## Recommended pipeline (aligned with OAG)

To mirror `preprocess_OAG_XLNet.py` → `build_shared_pv_base.py` and keep full heterogeneous edge types (including `AP_write_first` / `AP_write_last` / `AP_write_other`), run in order:

### 1. Build the Graph pickle

```bash
cd <repo_root>
python3 OGBN/data_preprocess/preprocess_ogbn_mag_w2v.py \
  --ogbn_mag_dir data/ogbn_mag \
  --output_dir OGBN/data \
  --test_time_bar 2017
```

**Main arguments**

- `--ogbn_mag_dir`: Root of the extracted ogbn-mag dataset (default: `<repo_root>/data/ogbn_mag`).
- `--output_dir`: Output directory (default: `OGBN/data`); produces **`graph_ogbn_mag_w2v.pk`** (large, on the order of several GB).
- `--test_time_bar`: When aggregating paper embeddings to venue / field, only neighbors with **edge time ≤ this year** are used (default `2017`, consistent with the training period before “2018 validation”).

**Outputs**

- `graph_ogbn_mag_w2v.pk`: `dill`-serialized `THGFM.data.Graph` (`node_backward` removed; keeps `node_forward`, `node_feature`, `edge_list`, etc.).

### 2. Export shared PV base from the Graph

```bash
python3 OGBN/data_preprocess/build_shared_pv_base_ogbn_mag.py \
  --data_dir OGBN/data \
  --graph_file graph_ogbn_mag_w2v.pk \
  --ogbn_mag_dir data/ogbn_mag \
  --out_dir OGBN/data/shared_pv_base_ogbn_mag_graph
```

**Main arguments**

- `--data_dir`: Directory containing `--graph_file` (default `OGBN/data`).
- `--graph_file`: Pickle from the previous step (default `graph_ogbn_mag_w2v.pk`).
- `--ogbn_mag_dir`: Used to read **`split/time/paper/{train,valid,test}.csv`** and **`node_year.csv`** (`split_max_time`).
- `--out_dir`: Shared base output (default `OGBN/data/shared_pv_base_ogbn_mag_graph`).
- `--source_shared_base_dir` (optional): If a shared base already exists, symlink its **`features/`** and **`relations/`**, and only **rewrite `labels/`** and the task-related parts of `meta.json` from OGB splits.

**Output layout (same as OAG shared base)**

- `features/{paper,author,affiliation,field,venue}.npy`: Each row is **`[node_emb(400 zero pad) \| emb \| log10(citation+0.01)]`**, **529** dims with default 128-d `emb`.
- `relations/`: CSR shards per relation (`*__target_ids.npy`, `*__indptr.npy`, `*__source_ids.npy`, `*__edge_times.npy`).
- `labels/`: `candidate_venue_ids.npy` and per-split `*_paper_ids.npy`, `*_venue_ids.npy`, `*_times.npy`.
- `meta.json`: `schema_version`, `node_counts`, `feature_dims`, `relations`, `task`, `split_max_time`, etc.

## Optional: shared base from raw OGB only

No Graph pickle; faster, but **relation types differ** from the graph-preprocessing path (see table above).

```bash
cd <repo_root>
python3 OGBN/data_preprocess/build_shared_ogbn_mag_pv_base.py \
  --ogbn_mag_dir data/ogbn_mag \
  --out_dir <your_output_dir> \
  --feature_dim 128
```

**Note**: The script’s default `--ogbn_mag_dir` may point to an absolute path on the author’s machine; pass **`--ogbn_mag_dir`** explicitly in your environment.

**Outputs**

- Same shared base layout (`features/`, `relations/`, `labels/`, `meta.json`); under default settings features are **128-d per type** (no 400-d pad or citation concat unless you add them downstream).

## Choosing between the two shared bases

- For alignment with **OAG `Graph` / `sample_subgraph` / multi-type author edges**: use **`preprocess_ogbn_mag_w2v.py` + `build_shared_pv_base_ogbn_mag.py`**.
- For a **lightweight, reproducible OGB edge matrix** without a pickle dependency: use **`build_shared_ogbn_mag_pv_base.py`**.

## FAQ

- **`THGFM` not found when loading `graph_ogbn_mag_w2v.pk`**: ensure `OAG/` is on `sys.path` (same layout as when the file was written) and that the pickle was built with **`THGFM.data.Graph`** (not legacy aliases like `pyHGT.data`). Re-run `preprocess_ogbn_mag_w2v.py` if the file predates this layout.
- **`divide by zero` during propagation**: From `THGFM.utils.normalize` row-normalizing isolated nodes; similar to OAG preprocessing and usually safe to ignore.
- **Disk space**: Both `graph_ogbn_mag_w2v.pk` and `relations/` under the shared base are large; reserve enough space.
