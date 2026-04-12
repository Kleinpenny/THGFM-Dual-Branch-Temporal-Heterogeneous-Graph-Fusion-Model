# OAG data preprocessing (`data_preprocess/`)

Scripts in this folder match the current implementation. Run the examples from **`OAG/`**; paths below are relative to `OAG/`. The conda environment is defined at the repo root: `../environment.yml`.

## Scripts

| Script | Role |
|--------|------|
| `preprocess_OAG_XLNet.py` | XLNet paper embeddings → `graph{domain}_XLNet.pk` |
| `preprocess_OAG_Qwen3.py` | Qwen3-Embedding-8B → `graph{domain}_Qwen3.pk` |
| `preprocess_OAG_Qwen3_fp16.py` | Same model; embeddings stored as fp16 → `graph{domain}_Qwen3_fp16.pk` |
| `build_shared_ad_base.py` | AD shared read-only arrays from `.pk` (`meta.json`, `features/`, `relations/`, `groups/`) |
| `build_shared_pf_base.py` | Paper–Field shared base (can reuse `features/` and `relations/` from an AD shared base) |
| `build_shared_pv_base.py` | Paper–Venue shared base (same reuse pattern) |

## Graph build CLI (all three `preprocess_OAG_*.py`)

- `--input_dir`: raw TSV directory (default resolves to repo `../data/raw_data/MAG_0919_CS`)
- `--output_dir`: output directory (default `../data`)
- `--domain`: suffix such as `_CS`, `_Med`, or empty string (must match files like `Papers{domain}_20190919.tsv`)
- `--cuda`: GPU id; `-1` for CPU
- `--citation_bar`: citation lower-bound scale, default `1` (see `(2020 - year) * citation_bar` in code)

Outputs:

- `../data/graph{domain}_XLNet.pk`
- `../data/graph{domain}_Qwen3.pk`
- `../data/graph{domain}_Qwen3_fp16.pk`

## Optional `nohup` examples

From `OAG/`:

```bash
mkdir -p logs

nohup env CUDA_VISIBLE_DEVICES=0 python data_preprocess/preprocess_OAG_XLNet.py \
  --input_dir ../data/raw_data/MAG_0919_CS \
  --output_dir ../data \
  --domain _CS \
  --cuda 0 \
  > logs/preprocess_XLNet.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 python data_preprocess/preprocess_OAG_Qwen3_fp16.py \
  --input_dir ../data/raw_data/MAG_0919_CS \
  --output_dir ../data \
  --domain _CS \
  --cuda 0 \
  > logs/preprocess_Qwen3_fp16.log 2>&1 &
```

Tail logs: `tail -f logs/preprocess_XLNet.log` (adjust the log name as needed).

## Shared bases (`build_shared_*.py`)

Run from `OAG/`. Default `--data_dir` is repo `../data`. If `--graph_file` is set, it overrides the `graph{domain}.pk` name implied by `--domain`.

**1. AD base (author disambiguation)**

```bash
python data_preprocess/build_shared_ad_base.py \
  --data_dir ../data \
  --graph_file graph_CS_XLNet.pk \
  --out_dir ../data/shared_ad_base_CS_XLNet
```

**2. PF (L1 / L2) — prefer reusing large arrays from an AD shared base**

```bash
python data_preprocess/build_shared_pf_base.py \
  --field_level L1 \
  --source_shared_base_dir ../data/shared_ad_base_CS_XLNet \
  --out_dir ../data/shared_pf_base_CS_XLNet_L1

python data_preprocess/build_shared_pf_base.py \
  --field_level L2 \
  --source_shared_base_dir ../data/shared_ad_base_CS_XLNet \
  --out_dir ../data/shared_pf_base_CS_XLNet_L2
```

**3. PV — same `--source_shared_base_dir` recommendation**

```bash
python data_preprocess/build_shared_pv_base.py \
  --source_shared_base_dir ../data/shared_ad_base_CS_XLNet \
  --out_dir ../data/shared_pv_base_CS_XLNet
```

Each `--out_dir` contains `meta.json`; task labels live under `labels/` (PF/PV) or `groups/` (AD). Without `--source_shared_base_dir`, `build_shared_pf_base.py` and `build_shared_pv_base.py` can rebuild everything from a `.pk` using `--data_dir` plus `--domain` and/or `--graph_file`, as documented in those scripts.

## Linking to shared trainers

Shared training entry points live under `../trainers/main/` (e.g. `train_author_disambiguation_shared.py`, `train_paper_field_shared.py`, `train_paper_venue_shared.py`). Thin wrappers `build_shared_ad_base.py` and `build_shared_pv_base.py` in that folder delegate to the implementations here; for PF, run `build_shared_pf_base.py` in this directory directly. Point trainer arguments at the `../data/shared_*` directories you built above. See `../trainers/main/README.md`.
