# THGFM: Dual-Branch Temporal Heterogeneous Graph Fusion Model

**ISWC 2026 Research Track · Oral Presentation**

[![ISWC 2026](https://img.shields.io/badge/ISWC-2026%20Oral-2463a6)](https://iswc2026.semanticweb.org/)
[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2607.27303-b31b1b)](https://arxiv.org/abs/2607.27303)
[![Code](https://img.shields.io/badge/Code-PyTorch-orange)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)

**Yixin Peng¹, Diego Collarana², Er Jin¹, Stefan Decker¹˒²**  
¹ RWTH Aachen University · ² Fraunhofer FIT

**[Paper (arXiv)](https://arxiv.org/abs/2607.27303) · [PDF](https://arxiv.org/pdf/2607.27303) · [Repository](https://github.com/Kleinpenny/THGFM-Dual-Branch-Temporal-Heterogeneous-Graph-Fusion-Model) · [Citation](#citation)**

> **TL;DR:** THGFM learns from temporal heterogeneous graphs with **two complementary attention branches**: one shares information across node types, while the other preserves relation-specific semantics. **Rotary Temporal Attention (RoTA)** introduces relative time directly into query–key attention scoring, and **type-conditioned, non-competitive gates** combine the two views at every layer.

This is the **official implementation** of *THGFM: Dual-Branch Temporal Heterogeneous Graph Fusion Model*, presented at the **25th International Semantic Web Conference (ISWC 2026)**.

## Highlights

- **Complementary attention:** **Shared-Space Temporal Attention (SSTA)** enables cross-type transfer; **Relational Type-Partitioned Temporal Attention (RTTA)** models type- and relation-specific interactions.
- **Time-aware matching:** **Rotary Temporal Attention (RoTA)** incorporates pairwise relative time *inside* the attention score, rather than relying only on additive temporal features.
- **Adaptive, non-competitive fusion:** **Dual-Path Relational–Shared Fusion (DRSF)** combines shared and specialized evidence with independently gated, optionally target-type-conditioned branches.

**Experimental highlights** (under the settings reported in the paper and ISWC slides):

| +3.25% | +12.37% | 6 tasks / 4 graphs |
|:--:|:--:|:--:|
| Six-task mean relative gain **vs. HGT** for **TC-NGSF**, averaged over three temporal settings | Peak relative gain on **OAG-CS paper–venue** for **sum fusion with RoTA only** | OAG-CS, OGBN-MAG, HTAG-ArXiv and HTAG-DBLP |

> **Important:** The mean, peak, and per-task best results above are **different aggregations/configurations**, not metrics from one universal trained checkpoint. See [Main results](#main-results) and [Ablations](#what-the-ablations-show).

## Why THGFM?

Real-world heterogeneous graphs contain multiple **node types**, **relation types**, and **time-dependent interactions**. For example, an academic graph distinguishes papers, authors, venues, fields of study, and affiliations. A *cites* edge and a *written by* edge have different meanings, while observed relation timestamps may be missing for some edge types.

Two modeling requirements arise:

1. **Share what is transferable:** less frequent node types can benefit from a common representation space and cross-type knowledge transfer.
2. **Preserve what is different:** heterogeneous relations carry distinct semantics, and differences in event time should affect how neighbors are matched.

THGFM addresses both through **layer-wise dual-branch temporal attention and fusion**.

<p align="center">
  <img src="assets/THGFM%20Architecture.png" width="88%" alt="THGFM overall architecture showing SSTA, RTTA, DRSF and residual mixing" />
  <br />
  <em>Overall architecture. Each THGFM layer applies SSTA and RTTA, combines them with DRSF, and passes the result through residual mixing and normalization.</em>
</p>

## Method

### 1. Complementary temporal attention: SSTA + RTTA

| | **SSTA: shared space** | **RTTA: relation-specific space** |
|---|---|---|
| Purpose | Cross-type transfer; shared evidence | Preserve relation- and type-dependent semantics |
| Projections | Type adapters followed by shared Q/K/V projections | Type-specific Q/K/V projections with relation-specific attention/message transforms |
| Neighbor normalization | One softmax over all sampled neighbors | Separate softmax per source-node type |
| Benefit | Share statistical strength across types | Distinguish structurally different interactions |

Both branches operate **in parallel within every THGFM layer**; neither replaces the other.

<img width="1482" height="546" alt="image" src="https://github.com/user-attachments/assets/c2dfb01b-e71e-4f0a-995d-25f57b863659" />

### 2. Rotary Temporal Attention (RoTA): time inside attention

A key distinction is **where temporal information enters the model**. Traditional additive relative-time encoding (RTE) augments a source representation before projection; RoTA instead uses relative time to rotate the **query and key vectors used for attention matching**.

| | Additive RTE | **RoTA (ours)** |
|---|---|---|
| Time signal | Relative time difference | Relative time difference |
| Injection point | Source/node features before K/V projections | Q/K vectors before the attention dot product |
| Effect on matching | Time changes projected representations | Time explicitly changes the query–key alignment through rotation |

For an incoming edge from source node \(j\) to target node \(i\), the sampled temporal context defines

$$
\Delta t_{ji} = t_i - t_j + c,
$$

where \(t_i\) is the target's context time and \(c\) is the time offset used by the temporal encoding. If the relation carries an observed timestamp, that timestamp determines \(t_j\); otherwise, \(t_j\) inherits the target context time. In the latter case, \(\Delta t_{ji}=c\).

<img width="1488" height="540" alt="image" src="https://github.com/user-attachments/assets/b02086e9-d4af-4599-93fa-75485a0ec4dd" />

**Conceptually**, RoTA applies opposite half-angle rotations to the query and key:

$$
\widetilde{q}_{i\leftarrow j}=R\!\left(+\tfrac{\theta(\Delta t_{ji})}{2}\right)q_i,
\qquad
\widetilde{k}_{j\to i}=R\!\left(-\tfrac{\theta(\Delta t_{ji})}{2}\right)k_j,
$$

and computes attention from their rotated dot product. This makes the pairwise temporal gap part of the **matching operation**, not just an extra feature added upstream. See the [paper](https://arxiv.org/abs/2607.27303) for the full parameterization and implementation details.

**Why it matters:** two neighbors may occupy adjacent sampling positions but be separated by very different amounts of real time. RoTA conditions attention on the actual edge-associated time difference instead of treating position and elapsed time as equivalent.

**Implementation:** see the rotary temporal attention variants in [`OAG/THGFM/conv.py`](OAG/THGFM/conv.py), [`HTAG/THGFM/conv.py`](HTAG/THGFM/conv.py), and [`OGBN/THGFM/conv.py`](OGBN/THGFM/conv.py).

### 3. Dual-Path Relational–Shared Fusion (DRSF)

After the two branches produce their representations, DRSF combines them. In particular, **Type-Conditioned Non-Competitive Gated Sum Fusion (TC-NGSF)** uses **independent, feature-wise gates**, conditioned on the **target-node type**, for the shared and specialized outputs.

Unlike a two-way softmax that forces the branches to compete for a fixed total weight, independent sigmoid gates allow **both branches to be emphasized simultaneously** when both provide useful evidence.

<img width="1480" height="558" alt="image" src="https://github.com/user-attachments/assets/7885a173-5694-4732-a044-17a4cdf211c3" />

<p align="center">
  <img src="assets/DRSF%20Fusion.png" width="65%" alt="Dual-Path Relational-Shared Fusion (DRSF) operators" />
  <br />
  <em>DRSF includes linear/sum fusion and non-competitive gated variants, with optional target-type conditioning.</em>
</p>

The method also supports **single-branch and fusion ablations** to isolate the contributions of SSTA, RTTA, RoTA, and the fusion operator.

## Main results

THGFM was evaluated on **six supervised prediction tasks over four academic graphs**, with matched data features, sampling, and training pipelines for **HGT, CTRL, ieHGCN, and SeHGNN**. The reported study uses **three layers**, **256 hidden units**, and **nine runs**.

### Benchmark scale

| Graph | Nodes | Edges | Prediction tasks |
|---|---:|---:|---|
| OAG-CS | 11.73M | 107.26M | Paper–field (L1/L2), paper–venue |
| OGBN-MAG | 1.94M | 27.20M | Paper venue classification (349 classes) |
| HTAG-ArXiv | 0.23M | 2.08M | Paper subject area (40 classes) |
| HTAG-DBLP | 1.99M | 29.83M | Paper topic label (9 classes) |

### Results across six tasks

| Dataset / task | Primary metric | Strongest baseline | THGFM best | Relative gain | OAG MRR (THGFM best) |
|---|---|---:|---:|---:|---:|
| OAG-CS · Paper–Field L1 | NDCG | 0.8522 (CTRL) | **0.8603** | **+0.95%** | 0.8439 |
| OAG-CS · Paper–Field L2 | NDCG | 0.4375 (HGT) | **0.4528** | **+3.50%** | 0.5437 |
| OAG-CS · Paper–Venue | NDCG | 0.4878 (ieHGCN) | **0.5102** | **+4.59%** | 0.3378 |
| OGBN-MAG · Venue | Test accuracy | 0.4017 (CTRL) | **0.4150** | **+3.31%** | — |
| HTAG-ArXiv · Subject area | Test accuracy | 0.8443 (CTRL) | **0.8598** | **+1.84%** | — |
| HTAG-DBLP · Topic label | Test accuracy | 0.7540 (CTRL) | **0.7895** | **+4.71%** | — |

**Interpretation:** Values labeled **“THGFM best”** are the **best reported mean for each metric across the evaluated configurations**, as summarized in the ISWC presentation. They **do not correspond to a single common checkpoint or configuration** across all tasks. Relative gains here compare the primary metric to the strongest reported baseline on the same task. NDCG/MRR and accuracy are **not** interchangeable or averaged directly in this table.

### What the ablations show

**Temporal attention:** Across the top three configurations under each temporal setting, the six-task mean relative gains over HGT were:

| Temporal setting | Mean relative gain |
|---|---:|
| Additive RTE only | +2.83% |
| **RoTA only** | **+3.42%** |
| RTE + RoTA | +2.92% |

The **RoTA-only** setting exceeds **RTE-only** by **0.59 percentage points** under this aggregation, with the same parameter count in the reported comparison. The combination is **not universally additive**.

**Fusion:** Averaging across the three temporal settings, **TC-NGSF** attains **+3.25% six-task mean relative gain over HGT**. A separate **sum-fusion + RoTA-only** configuration achieves a **peak +12.37% relative gain** on OAG-CS paper–venue prediction. These should not be conflated with the per-task best-results table above.

<details>
<summary><strong>More fusion ablation results (six-task mean relative gain)</strong></summary>

| Gating scheme | Sum fusion | Linear fusion |
|---|---:|---:|
| No gates | +2.60% | +1.04% |
| Independent gates | +3.23% | +1.54% |
| **Type-conditioned independent gates** | **+3.25%** | **+1.87%** |

The strongest average fusion variant is **TC-NGSF** (type-conditioned non-competitive gated sum fusion).

</details>

<details>
<summary><strong>Shared preprocessing and training-system efficiency</strong></summary>

The experiments also use an improved **shared preprocessing pipeline** for THGFM **and the compared baselines**, rather than claiming that these are architectural speedups over every baseline:

| Measurement | Earlier pipeline | Improved shared pipeline | Reduction |
|---|---:|---:|---:|
| Peak RAM | 26.86 GB | 4.15 GB | 84.5% |
| Graph preparation | 14.8 s/epoch | 4.9 s/epoch | 66.9% |
| Epoch wall time | 219.4 s/epoch | 101.9 s/epoch | 53.6% |

These measurements describe the pipeline comparison reported in the ISWC slides, not independent estimates of RoTA's runtime savings.

</details>

## Get started

### Repository structure

```text
.
├── HTAG/                       # HTAG-ArXiv and HTAG-DBLP experiments
│   ├── Arxiv_PV/               # ArXiv paper prediction / preprocessing
│   ├── DBLP_PV/                # DBLP paper prediction / preprocessing
│   ├── THGFM/                  # Model and convolution modules
│   ├── trainers/main/          # Training pools and experiment guides
│   └── evaluator/              # HTAG evaluation
├── OAG/                        # OAG-CS PV and PF-L1/L2 experiments
│   ├── data_preprocess/
│   ├── THGFM/
│   ├── trainers/main/
│   └── evaluator/
├── OGBN/                       # OGBN-MAG classification
│   ├── data_preprocess/
│   ├── THGFM/
│   ├── trainers/main/
│   └── evaluator/
├── Best_Model_Save/            # Released task-specific trained checkpoints
├── scripts/                    # Unified checkpoint evaluation
├── assets/                     # Architecture and module diagrams
├── requirements.txt
└── environment.yml
```

### Environment

The provided environment targets:

- **Python 3.11**
- **PyTorch 2.4.0** with **CUDA 12.1**
- **PyTorch Geometric 2.7.0**
- **DGL 2.4.0+cu121**
- **OGB ≥ 1.3.6**

See [`requirements.txt`](requirements.txt) and [`environment.yml`](environment.yml) for exact dependency specifications. A compatible NVIDIA GPU and CUDA driver are recommended for full-scale experiments.

```bash
git clone https://github.com/Kleinpenny/THGFM-Dual-Branch-Temporal-Heterogeneous-Graph-Fusion-Model.git
cd THGFM-Dual-Branch-Temporal-Heterogeneous-Graph-Fusion-Model

# Option A: Conda (recommended)
conda env create -f environment.yml
conda activate thgfm
```

Alternatively, install with pip from a Python 3.11 environment (for example, a virtual environment):

```bash
# Option B: pip
pip install -r requirements.txt
```

Verify the installation:

```bash
python -c "import torch, torch_geometric, dgl; print('CUDA available:', torch.cuda.is_available())"
```

### Quick start: evaluate released checkpoints

Six **trained, task-specific** checkpoints are provided in [`Best_Model_Save/`](Best_Model_Save/): HTAG-ArXiv, HTAG-DBLP, OAG-CS PV, OAG-CS PF-L1, OAG-CS PF-L2, and OGBN-MAG. These are **supervised task checkpoints**, not self-supervised foundation-model weights.

After installing dependencies and preparing the corresponding benchmark data (see below), run from the **repository root**:

```bash
# Fast smoke test (one evaluation seed)
python scripts/eval_best_model_save.py --smoke --seeds 43 --cuda 0

# Evaluate all six released checkpoints
python scripts/eval_best_model_save.py --seeds 43,44,45 --cuda 0

# Preview commands without running them
python scripts/eval_best_model_save.py --dry-run --seeds 43,44
```

Evaluation results are written to `Best_Model_Save/eval_results/`. `--max-workers` controls concurrent evaluation jobs. **Note:** multiple `--seeds` here rerun evaluation using the **released checkpoints**; they are **not a substitute for independently retraining multiple model seeds** to reproduce paper-level uncertainty estimates.

## Dataset preparation

**All commands in this section assume you start from the repository root.** The benchmark-specific data generators build shared graph bases used for fair evaluation across methods. Large benchmark files and pretrained text embeddings may require additional disk space and downloads.

> **Feature consistency:** For fair comparisons, use the same feature extraction, graph construction, neighbor sampling, and training conditions for THGFM and baselines, as specified in the paper and the benchmark pipelines.

### HTAG-ArXiv and HTAG-DBLP

Download the upstream data from [HTAG](https://github.com/Cloudy1225/HTAG/tree/main) and follow its raw-pickle preparation guidance. Place the input pickles under the expected `data/HTAG/{arxiv,dblp}/` paths (or configure the preprocessing scripts accordingly).

```bash
(
  cd HTAG/DBLP_PV/data_preprocess
  python build_dblp_graph_from_pkl.py
  python build_shared_pv_base.py
)

(
  cd HTAG/Arxiv_PV/data_preprocess
  python build_arxiv_graph_from_pkl.py
  python build_shared_pv_base.py
)
```

### OAG-CS

Download the raw MAG CS TSVs from the [OAG dataset source](https://drive.google.com/open?id=1yDdVaartOCOSsQlUZs8cJcAUhmvRiBSz) into `data/raw_data/MAG_0919_CS/`, then generate XLNet features and build reusable graph bases. From the repository root:

```bash
(
  cd OAG

  python data_preprocess/preprocess_OAG_XLNet.py \
    --input_dir ../data/raw_data/MAG_0919_CS \
    --output_dir ../data --domain _CS --cuda 0

  python data_preprocess/build_shared_ad_base.py \
    --data_dir ../data --graph_file graph_CS_XLNet.pk \
    --out_dir ../data/shared_ad_base_CS_XLNet

  python data_preprocess/build_shared_pv_base.py \
    --source_shared_base_dir ../data/shared_ad_base_CS_XLNet \
    --out_dir ../data/shared_pv_base_CS_XLNet

  python data_preprocess/build_shared_pf_base.py \
    --field_level L1 \
    --source_shared_base_dir ../data/shared_ad_base_CS_XLNet \
    --out_dir ../data/shared_pf_base_CS_XLNet_L1

  python data_preprocess/build_shared_pf_base.py \
    --field_level L2 \
    --source_shared_base_dir ../data/shared_ad_base_CS_XLNet \
    --out_dir ../data/shared_pf_base_CS_XLNet_L2
)
```

The supported preprocessing options, including alternatives to XLNet, are documented in [`OAG/data_preprocess/README.md`](OAG/data_preprocess/README.md).

### OGBN-MAG

Download the official [ogbn-mag](https://ogb.stanford.edu/docs/nodeprop/#ogbn-mag) dataset into `data/ogbn_mag/`, then run:

```bash
python OGBN/data_preprocess/preprocess_ogbn_mag_w2v.py \
  --ogbn_mag_dir data/ogbn_mag --output_dir OGBN/data

python OGBN/data_preprocess/build_shared_pv_base_ogbn_mag.py
```

## Training

**Training examples are independent commands run from the repository root.** The first example uses dual-branch, type-conditioned fusion; the next two explicitly select a RoTA-enabled fusion variant. These are **example runs**, not a claim that one configuration produced every best value in the results table.

### HTAG-DBLP: dual-branch TC-NGSF

```bash
(
  cd HTAG
  python DBLP_PV/train_paper_venue_shared.py \
    --shared_base_dir ./DBLP_PV/data/shared_pv_base_HTAG_dblp \
    --model_dir ./model_save_shared \
    --task_name HTAG_DBLP_tcngsf_demo \
    --conv_name drsf_tcngsf \
    --cuda 0 --seed 43 --deterministic
)
```

### OAG-CS: RoTA-enabled TC-NGSF

```bash
(
  cd OAG
  python trainers/main/train_paper_venue_shared.py \
    --shared_base_dir ../data/shared_pv_base_CS_XLNet \
    --model_dir ./model_save_shared \
    --task_name PV_XLNet_tcngsf_rota_demo \
    --conv_name drsf_tcngsf_rota \
    --cuda 0 --seed 43 --deterministic
)
```

### OGBN-MAG: RoTA-enabled TC-NGSF

```bash
(
  cd OGBN
  python trainers/main/train_paper_venue_shared.py \
    --shared_base_dir ./data/shared_pv_base_ogbn_mag_graph \
    --model_dir ./model_save_shared \
    --task_name PV_ogbn_mag_tcngsf_rota_demo \
    --conv_name drsf_tcngsf_rota \
    --cuda 0 --seed 43 --deterministic
)
```

For other branches and ablations, inspect each script's `--help` and the benchmark-specific training guides:

- [`HTAG/trainers/main/README.md`](HTAG/trainers/main/README.md)
- [`OAG/trainers/main/README.md`](OAG/trainers/main/README.md)
- [`OGBN/trainers/main/`](OGBN/trainers/main/)

For example, `rtta` and `ssta` are **single-branch baselines**, whereas `drsf_tcngsf` and `drsf_tcngsf_rota` select fusion models. The repository also contains manifest-based multi-seed/task-pool launchers. [Weights & Biases](https://wandb.ai/) logging is available with `--use_wandb`.

## Evaluation

### Per-benchmark evaluators

If you want to inspect results from a specific experiment or checkpoint:

```bash
# HTAG-DBLP
(
  cd HTAG
  python evaluator/eval_pv_htag_dblp.py \
    --checkpoint /path/to/checkpoint.pt --cuda 0
)

# OAG-CS paper–venue (NDCG and MRR)
(
  cd OAG
  python evaluator/eval_pv_test_ndcg_mrr.py \
    --checkpoint /path/to/checkpoint \
    --shared_base_dir ../data/shared_pv_base_CS_XLNet --cuda 0
)

# OGBN-MAG (official OGB accuracy)
(
  cd OGBN
  python evaluator/eval_test_accuracy.py \
    --checkpoint /path/to/checkpoint \
    --shared_base_dir ./data/shared_pv_base_ogbn_mag_graph \
    --splits valid,test --seed 43 --batch_size 256 --cuda 0
)
```

For the released checkpoint names, benchmark-specific evaluator paths, and concurrent execution, see [`scripts/eval_best_model_save.py`](scripts/eval_best_model_save.py).

## Scope and future directions

THGFM is evaluated here as a **supervised temporal heterogeneous graph model** on academic graphs. The ISWC presentation also identifies ongoing research questions:

- **Temporal context consistency:** a global node may be assigned different induced context times in different sampled batches.
- **Cross-graph and cross-database generalization:** transferring learned representations across new relational domains remains an open direction.
- **Relational pretraining and graph–language integration:** extending the encoder toward reusable representations and broader relational intelligence is **future work**, not a demonstrated feature of the released checkpoints.

We welcome reproducibility reports, questions about RoTA, and research discussions on temporal relational representation learning via [GitHub Issues](https://github.com/Kleinpenny/THGFM-Dual-Branch-Temporal-Heterogeneous-Graph-Fusion-Model/issues).

## Citation

If you use **THGFM, RoTA, the fusion architecture, or this implementation**, please cite:

```bibtex
@misc{peng2026thgfm,
  title         = {THGFM: Dual-Branch Temporal Heterogeneous Graph Fusion Model},
  author        = {Peng, Yixin and Collarana, Diego and Jin, Er and Decker, Stefan},
  year          = {2026},
  eprint        = {2607.27303},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2607.27303},
  url           = {https://arxiv.org/abs/2607.27303},
  note          = {Accepted to ISWC 2026 Research Track (oral presentation)}
}
```

*This arXiv-based entry is provided until the verified proceedings metadata (e.g., LNCS volume, page range, and publication DOI) is available. Please replace it with the official published citation when released.*

## Acknowledgments

This implementation builds on the open-source ecosystems and benchmark resources provided by [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric), [DGL](https://github.com/dmlc/dgl), [Open Graph Benchmark (OGB)](https://ogb.stanford.edu/), and [HTAG](https://github.com/Cloudy1225/HTAG).

## License

This repository is distributed under the [MIT License](LICENSE).
