import argparse
import os
import pickle
import sys
from typing import Dict

import dill
import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm.auto import tqdm

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_DBLP_PV_DIR = os.path.abspath(os.path.join(_CUR_DIR, ".."))
_HTAG_DIR = os.path.abspath(os.path.join(_DBLP_PV_DIR, ".."))
if _HTAG_DIR not in sys.path:
    sys.path.insert(0, _HTAG_DIR)

from THGFM.data import Graph


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_dblp_payload(path: str) -> Dict[str, object]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    required = {
        "paper_labels",
        "paper_years",
        "paper-paper",
        "paper_feats",
        "paper-author",
        "paper-fos",
    }
    missing = required.difference(set(data.keys()))
    if missing:
        raise KeyError(f"Missing keys in dblp.pkl: {sorted(list(missing))}")
    return data


def _build_temporal_split(years: np.ndarray):
    """DBLP HTAG: train before 2010; valid 2011–2013; test from 2014 onward. Year 2010 is excluded."""
    train_ids = np.nonzero(years <= 2009)[0].astype(np.int64, copy=False)
    valid_ids = np.nonzero((years >= 2011) & (years <= 2013))[0].astype(np.int64, copy=False)
    test_ids = np.nonzero(years >= 2014)[0].astype(np.int64, copy=False)
    return train_ids, valid_ids, test_ids


def _normalize(mx: sp.coo_matrix):
    rowsum = np.array(mx.sum(1)).flatten()
    r_inv = np.zeros_like(rowsum, dtype=np.float32)
    nz = rowsum > 0
    r_inv[nz] = np.power(rowsum[nz], -1)
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(mx)


def _propagate_paper_emb_to_type(graph: Graph, node_type: str, paper_embeddings: np.ndarray) -> np.ndarray:
    node_pairs = []
    rel_names = list(graph.edge_list[node_type]["paper"])
    for rel_name in tqdm(rel_names, desc=f"collect {node_type}→paper edges", dynamic_ncols=True, leave=False):
        rel_map = graph.edge_list[node_type]["paper"][rel_name]
        for target_idx in rel_map:
            for source_idx in rel_map[target_idx]:
                node_pairs.append([int(target_idx), int(source_idx)])

    if len(node_pairs) == 0:
        return np.zeros((len(graph.node_backward[node_type]), paper_embeddings.shape[1]), dtype=np.float32)

    node_pairs = np.asarray(node_pairs, dtype=np.int64).T
    edge_count = int(node_pairs.shape[1])
    v = np.ones(edge_count, dtype=np.float32)
    m = _normalize(
        sp.coo_matrix(
            (v, node_pairs),
            shape=(len(graph.node_backward[node_type]), len(graph.node_backward["paper"])),
        )
    )
    out = m.dot(paper_embeddings).astype(np.float32, copy=False)
    return out


def _save_split_arrays(
    split_dir: str,
    split_name: str,
    paper_ids: np.ndarray,
    paper_years: np.ndarray,
    paper_labels: np.ndarray,
) -> None:
    _ensure_dir(split_dir)
    np.save(os.path.join(split_dir, f"{split_name}__paper_ids.npy"), paper_ids, allow_pickle=False)
    np.save(
        os.path.join(split_dir, f"{split_name}__times.npy"),
        paper_years[paper_ids].astype(np.int32, copy=False),
        allow_pickle=False,
    )
    np.save(
        os.path.join(split_dir, f"{split_name}__venue_ids.npy"),
        paper_labels[paper_ids].astype(np.int64, copy=False),
        allow_pickle=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build HTAG DBLP Graph from dblp.pkl")
    # Repo layout: .../THGFM-Dual-Branch-Temporal-Heterogeneous-Graph-Fusion-Model/{HTAG/DBLP_PV, data/HTAG/dblp/}
    _repo_data = os.path.abspath(os.path.join(_DBLP_PV_DIR, "..", "..", "data"))
    default_root = os.path.join(_DBLP_PV_DIR, "data")
    parser.add_argument(
        "--input_pkl",
        type=str,
        default=os.path.join(_repo_data, "HTAG", "dblp", "dblp.pkl"),
        help="Path to DBLP pkl file from HTAG release.",
    )
    parser.add_argument(
        "--out_graph",
        type=str,
        default=os.path.join(default_root, "graph_dblp_HTAG.pk"),
        help="Output path for serialized Graph object.",
    )
    parser.add_argument(
        "--split_dir",
        type=str,
        default=os.path.join(default_root, "dblp_splits"),
        help="Directory to save temporal split arrays.",
    )
    args = parser.parse_args()

    payload = _load_dblp_payload(args.input_pkl)
    paper_labels = np.asarray(payload["paper_labels"], dtype=np.int64)
    paper_years = np.asarray(payload["paper_years"], dtype=np.int32)
    paper_feats = np.asarray(payload["paper_feats"], dtype=np.float32)

    pp_src, pp_dst = payload["paper-paper"]
    pa_paper, pa_author = payload["paper-author"]
    pf_paper, pf_fos = payload["paper-fos"]
    pp_src = np.asarray(pp_src, dtype=np.int64)
    pp_dst = np.asarray(pp_dst, dtype=np.int64)
    pa_paper = np.asarray(pa_paper, dtype=np.int64)
    pa_author = np.asarray(pa_author, dtype=np.int64)
    pf_paper = np.asarray(pf_paper, dtype=np.int64)
    pf_fos = np.asarray(pf_fos, dtype=np.int64)

    n_paper = int(paper_feats.shape[0])
    feat_dim = int(paper_feats.shape[1])
    n_author = int(pa_author.max()) + 1
    n_field = int(pf_fos.max()) + 1
    n_venue = int(paper_labels.max()) + 1

    if len(paper_labels) != n_paper or len(paper_years) != n_paper:
        raise ValueError("paper_labels/paper_years length must match paper_feats rows.")

    print(f"[INFO] papers={n_paper} authors={n_author} fields={n_field} classes={n_venue}", flush=True)
    print(f"[INFO] paper feature dim={feat_dim}", flush=True)

    graph = Graph()
    graph.feature_mode = "htag"
    graph.feature_dim = feat_dim

    print("[INFO] Creating nodes...", flush=True)
    zero_emb = np.zeros((feat_dim,), dtype=np.float32).tolist()
    for pid in tqdm(range(n_paper), desc="paper nodes", dynamic_ncols=True):
        graph.add_node(
            {
                "id": int(pid),
                "type": "paper",
                "title": "",
                "time": int(paper_years[pid]),
                "label": int(paper_labels[pid]),
                "citation": 0,
                "emb": paper_feats[pid].astype(np.float32, copy=False).tolist(),
            }
        )
    for aid in tqdm(range(n_author), desc="author nodes", dynamic_ncols=True):
        graph.add_node({"id": int(aid), "type": "author", "citation": 0, "emb": zero_emb})
    for fid in tqdm(range(n_field), desc="field nodes", dynamic_ncols=True):
        graph.add_node({"id": int(fid), "type": "field", "citation": 0, "emb": zero_emb})
    for vid in tqdm(range(n_venue), desc="venue nodes", dynamic_ncols=True):
        graph.add_node({"id": int(vid), "type": "venue", "citation": 0, "emb": zero_emb})

    print("[INFO] Creating edges...", flush=True)
    for src, dst in tqdm(zip(pp_src, pp_dst), total=len(pp_src), desc="paper-paper", dynamic_ncols=True):
        src = int(src)
        dst = int(dst)
        graph.add_edge(
            {"id": src, "type": "paper"},
            {"id": dst, "type": "paper"},
            time=int(paper_years[src]),
            relation_type="PP_cite",
        )
    for pid, aid in tqdm(zip(pa_paper, pa_author), total=len(pa_paper), desc="author-paper", dynamic_ncols=True):
        pid = int(pid)
        aid = int(aid)
        graph.add_edge(
            {"id": aid, "type": "author"},
            {"id": pid, "type": "paper"},
            time=int(paper_years[pid]),
            relation_type="AP_write_other",
        )
    for pid, fid in tqdm(zip(pf_paper, pf_fos), total=len(pf_paper), desc="paper-field", dynamic_ncols=True):
        pid = int(pid)
        fid = int(fid)
        graph.add_edge(
            {"id": pid, "type": "paper"},
            {"id": fid, "type": "field"},
            time=int(paper_years[pid]),
            relation_type="PF_in_L2",
        )
    for pid in tqdm(range(n_paper), desc="paper-venue", dynamic_ncols=True):
        vid = int(paper_labels[pid])
        graph.add_edge(
            {"id": int(pid), "type": "paper"},
            {"id": vid, "type": "venue"},
            time=int(paper_years[pid]),
            relation_type="PV_Journal",
        )

    print("[INFO] Calculating citation attributes...", flush=True)
    pp_rel = graph.edge_list["paper"]["paper"]["PP_cite"]
    for paper_idx, paper_node in enumerate(tqdm(graph.node_backward["paper"], desc="paper citation", dynamic_ncols=True)):
        paper_node["citation"] = int(len(pp_rel.get(paper_idx, {})))

    author_rev = graph.edge_list["author"]["paper"]
    for author_idx, author_node in enumerate(tqdm(graph.node_backward["author"], desc="author citation", dynamic_ncols=True)):
        csum = 0
        for rel_name in author_rev.keys():
            for paper_idx in author_rev[rel_name].get(author_idx, {}):
                csum += int(graph.node_backward["paper"][int(paper_idx)]["citation"])
        author_node["citation"] = int(csum)

    field_rev = graph.edge_list["field"]["paper"]
    for field_idx, field_node in enumerate(tqdm(graph.node_backward["field"], desc="field citation", dynamic_ncols=True)):
        csum = 0
        for rel_name in field_rev.keys():
            for paper_idx in field_rev[rel_name].get(field_idx, {}):
                csum += int(graph.node_backward["paper"][int(paper_idx)]["citation"])
        field_node["citation"] = int(csum)

    venue_rev = graph.edge_list["venue"]["paper"]
    for venue_idx, venue_node in enumerate(tqdm(graph.node_backward["venue"], desc="venue citation", dynamic_ncols=True)):
        csum = 0
        for rel_name in venue_rev.keys():
            for paper_idx in venue_rev[rel_name].get(venue_idx, {}):
                csum += int(graph.node_backward["paper"][int(paper_idx)]["citation"])
        venue_node["citation"] = int(csum)

    print("[INFO] Propagating paper embeddings to textless node types...", flush=True)
    paper_df = pd.DataFrame(graph.node_backward["paper"])
    paper_embeddings = np.asarray(list(paper_df["emb"]), dtype=np.float32)
    for node_type in tqdm(("author", "field", "venue"), desc="propagate paper emb", dynamic_ncols=True):
        out_emb = _propagate_paper_emb_to_type(graph, node_type, paper_embeddings)
        for idx, vec in enumerate(
            tqdm(out_emb, desc=f"write {node_type} emb", dynamic_ncols=True, leave=False)
        ):
            graph.node_backward[node_type][idx]["emb"] = vec.tolist()

    print("[INFO] Building node_feature tables...", flush=True)
    graph.node_feature = {
        "paper": paper_df,
        "author": pd.DataFrame(graph.node_backward["author"]),
        "field": pd.DataFrame(graph.node_backward["field"]),
        "venue": pd.DataFrame(graph.node_backward["venue"]),
    }

    out_graph_dir = os.path.dirname(os.path.abspath(args.out_graph))
    _ensure_dir(out_graph_dir)
    with open(args.out_graph, "wb") as f:
        dill.dump(graph, f)
    print(f"[OK] graph saved: {args.out_graph}", flush=True)

    train_ids, valid_ids, test_ids = _build_temporal_split(paper_years)
    _ensure_dir(args.split_dir)
    _save_split_arrays(args.split_dir, "train", train_ids, paper_years, paper_labels)
    _save_split_arrays(args.split_dir, "valid", valid_ids, paper_years, paper_labels)
    _save_split_arrays(args.split_dir, "test", test_ids, paper_years, paper_labels)
    np.save(
        os.path.join(args.split_dir, "candidate_venue_ids.npy"),
        np.arange(n_venue, dtype=np.int64),
        allow_pickle=False,
    )
    print(
        f"[OK] split saved: train={len(train_ids)} valid={len(valid_ids)} test={len(test_ids)} at {args.split_dir}",
        flush=True,
    )
    n_gap = int(np.sum(paper_years == 2010))
    if n_gap > 0:
        print(
            f"[INFO] papers with year==2010 (excluded from train/valid/test by HTAG spec): {n_gap}",
            flush=True,
        )


if __name__ == "__main__":
    main()
