import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
from tqdm.auto import tqdm


NONE_TIME_SENTINEL = np.int32(np.iinfo(np.int32).min)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_vector_csv(path: str, dtype=np.int64) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=",", dtype=dtype)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def _load_edge_csv(path: str) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=",", dtype=np.int64)
    if arr.ndim == 1:
        arr = arr.reshape(1, 2)
    return arr


def _build_csr(
    target_ids: np.ndarray,
    source_ids: np.ndarray,
    edge_times: np.ndarray,
    target_count: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(target_ids) == 0:
        return (
            np.empty((0,), dtype=np.int64),
            np.zeros((1,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int32),
        )

    order = np.lexsort((source_ids, target_ids))
    tgt = target_ids[order].astype(np.int64, copy=False)
    src = source_ids[order].astype(np.int64, copy=False)
    tim = edge_times[order].astype(np.int32, copy=False)

    counts = np.bincount(tgt, minlength=target_count)
    nz_tgt = np.nonzero(counts)[0].astype(np.int64, copy=False)
    indptr = np.zeros((len(nz_tgt) + 1,), dtype=np.int64)
    indptr[1:] = np.cumsum(counts[nz_tgt], dtype=np.int64)
    return nz_tgt, indptr, src, tim


def _save_relation_arrays(
    out_dir: str,
    key: str,
    target_ids: np.ndarray,
    indptr: np.ndarray,
    source_ids: np.ndarray,
    edge_times: np.ndarray,
) -> Dict[str, str]:
    rel_dir = os.path.join(out_dir, "relations")
    _ensure_dir(rel_dir)
    base = os.path.join("relations", key)
    files = {
        "target_ids": f"{base}__target_ids.npy",
        "indptr": f"{base}__indptr.npy",
        "source_ids": f"{base}__source_ids.npy",
        "edge_times": f"{base}__edge_times.npy",
    }
    np.save(os.path.join(out_dir, files["target_ids"]), target_ids, allow_pickle=False)
    np.save(os.path.join(out_dir, files["indptr"]), indptr, allow_pickle=False)
    np.save(os.path.join(out_dir, files["source_ids"]), source_ids, allow_pickle=False)
    np.save(os.path.join(out_dir, files["edge_times"]), edge_times, allow_pickle=False)
    return files


def _add_relation_pair(
    out_dir: str,
    relation_entries: List[dict],
    meta_graph: List[List[str]],
    source_type: str,
    target_type: str,
    relation_name: str,
    source_ids: np.ndarray,
    target_ids: np.ndarray,
    edge_times: np.ndarray,
    node_count: Dict[str, int],
) -> None:
    # Forward: target_type <- source_type
    f_tgt, f_indptr, f_src, f_tim = _build_csr(
        target_ids=target_ids,
        source_ids=source_ids,
        edge_times=edge_times,
        target_count=node_count[target_type],
    )
    key_f = f"{target_type}__{source_type}__{relation_name}"
    files_f = _save_relation_arrays(out_dir, key_f, f_tgt, f_indptr, f_src, f_tim)
    relation_entries.append(
        {
            "target_type": target_type,
            "source_type": source_type,
            "relation_type": relation_name,
            **files_f,
        }
    )
    meta_graph.append([target_type, source_type, relation_name])

    # Reverse: source_type <- target_type
    rev_name = f"rev_{relation_name}"
    r_tgt, r_indptr, r_src, r_tim = _build_csr(
        target_ids=source_ids,
        source_ids=target_ids,
        edge_times=edge_times,
        target_count=node_count[source_type],
    )
    key_r = f"{source_type}__{target_type}__{rev_name}"
    files_r = _save_relation_arrays(out_dir, key_r, r_tgt, r_indptr, r_src, r_tim)
    relation_entries.append(
        {
            "target_type": source_type,
            "source_type": target_type,
            "relation_type": rev_name,
            **files_r,
        }
    )
    meta_graph.append([source_type, target_type, rev_name])


def _save_features(
    out_dir: str,
    node_count: Dict[str, int],
    paper_feat: np.ndarray,
    feat_dim: int,
) -> Tuple[Dict[str, str], Dict[str, int]]:
    feat_dir = os.path.join(out_dir, "features")
    _ensure_dir(feat_dir)
    feature_files = {}
    feature_dims = {}

    features = {
        "paper": paper_feat.astype(np.float32, copy=False),
        "author": np.zeros((node_count["author"], feat_dim), dtype=np.float32),
        "affiliation": np.zeros((node_count["affiliation"], feat_dim), dtype=np.float32),
        "field": np.zeros((node_count["field"], feat_dim), dtype=np.float32),
        "venue": np.zeros((node_count["venue"], feat_dim), dtype=np.float32),
    }
    for node_type, feat in features.items():
        rel_path = os.path.join("features", f"{node_type}.npy")
        np.save(os.path.join(out_dir, rel_path), feat, allow_pickle=False)
        feature_files[node_type] = rel_path
        feature_dims[node_type] = int(feat.shape[1])
    return feature_files, feature_dims


def _save_split_labels(
    out_dir: str,
    split_name: str,
    paper_ids: np.ndarray,
    paper_year: np.ndarray,
    paper_label_mapped: np.ndarray,
) -> Tuple[dict, dict]:
    venue_ids = paper_label_mapped[paper_ids]
    valid_mask = venue_ids >= 0
    paper_ids = paper_ids[valid_mask].astype(np.int64, copy=False)
    venue_ids = venue_ids[valid_mask].astype(np.int64, copy=False)
    times = paper_year[paper_ids].astype(np.int32, copy=False)

    label_dir = os.path.join(out_dir, "labels")
    _ensure_dir(label_dir)
    paper_ids_path = os.path.join("labels", f"{split_name}__paper_ids.npy")
    venue_ids_path = os.path.join("labels", f"{split_name}__venue_ids.npy")
    times_path = os.path.join("labels", f"{split_name}__times.npy")

    np.save(os.path.join(out_dir, paper_ids_path), paper_ids, allow_pickle=False)
    np.save(os.path.join(out_dir, venue_ids_path), venue_ids, allow_pickle=False)
    np.save(os.path.join(out_dir, times_path), times, allow_pickle=False)

    return (
        {
            "paper_ids": paper_ids_path,
            "venue_ids": venue_ids_path,
            "times": times_path,
        },
        {"paper_count": int(len(paper_ids))},
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build THGFM shared PV base from extracted ogbn_mag files."
    )
    # One level up: OGBN repo root (not the parent monorepo), so default --out_dir stays under OGBN/data/.
    default_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parser.add_argument(
        "--ogbn_mag_dir",
        type=str,
        default=os.path.join(default_project_root, "data", "ogbn_mag"),
        help="Path to extracted ogbn_mag directory.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(default_project_root, "data", "shared_pv_base_ogbn_mag"),
        help="Output directory in THGFM shared-base format.",
    )
    parser.add_argument(
        "--feature_dim",
        type=int,
        default=128,
        help="Feature dimension for all node types. Paper uses first feature_dim dims.",
    )
    args = parser.parse_args()

    _ensure_dir(args.out_dir)
    ogbn_mag_dir = os.path.abspath(args.ogbn_mag_dir)
    print(f"[INFO] ogbn_mag_dir: {ogbn_mag_dir}", flush=True)
    print(f"[INFO] out_dir: {args.out_dir}", flush=True)

    num_node_path = os.path.join(ogbn_mag_dir, "raw", "num-node-dict.csv")
    num_node_row = _load_vector_csv(num_node_path, dtype=str)
    if num_node_row.ndim == 1:
        # Shape like ["paper","author","institution","field_of_study"]
        header = num_node_row
        values = _load_vector_csv(num_node_path, dtype=np.int64,)[1]  # unreachable for np.loadtxt with dtype=str
        raise RuntimeError("Unexpected num-node-dict.csv parsing result.")
    # Robust parse with plain text fallback for 2-row csv.
    with open(num_node_path, "r", encoding="utf-8") as f:
        lines = [x.strip() for x in f.readlines() if x.strip()]
    keys = lines[0].split(",")
    vals = [int(x) for x in lines[1].split(",")]
    raw_counts = dict(zip(keys, vals))

    paper_count = int(raw_counts["paper"])
    author_count = int(raw_counts["author"])
    affiliation_count = int(raw_counts["institution"])
    field_count = int(raw_counts["field_of_study"])

    paper_feat_path = os.path.join(ogbn_mag_dir, "raw", "node-feat", "paper", "node-feat.csv")
    paper_feat = np.loadtxt(paper_feat_path, delimiter=",", dtype=np.float32)
    if paper_feat.ndim == 1:
        paper_feat = paper_feat.reshape(1, -1)
    if paper_feat.shape[0] != paper_count:
        raise ValueError(f"paper feature row mismatch: {paper_feat.shape[0]} vs {paper_count}")
    if args.feature_dim <= paper_feat.shape[1]:
        paper_feat = paper_feat[:, : args.feature_dim]
    else:
        pad = np.zeros((paper_feat.shape[0], args.feature_dim - paper_feat.shape[1]), dtype=np.float32)
        paper_feat = np.concatenate([paper_feat, pad], axis=1)

    paper_year_path = os.path.join(ogbn_mag_dir, "raw", "node-feat", "paper", "node_year.csv")
    paper_year = _load_vector_csv(paper_year_path, dtype=np.int32)
    if paper_year.shape[0] != paper_count:
        raise ValueError(f"paper year row mismatch: {paper_year.shape[0]} vs {paper_count}")

    paper_label_path = os.path.join(ogbn_mag_dir, "raw", "node-label", "paper", "node-label.csv")
    paper_label_raw = _load_vector_csv(paper_label_path, dtype=np.int64)
    if paper_label_raw.shape[0] != paper_count:
        raise ValueError(f"paper label row mismatch: {paper_label_raw.shape[0]} vs {paper_count}")

    valid_label_mask = paper_label_raw >= 0
    unique_labels = np.sort(np.unique(paper_label_raw[valid_label_mask])).astype(np.int64, copy=False)
    label_to_vidx = {int(lbl): int(i) for i, lbl in enumerate(unique_labels)}
    paper_label_mapped = np.full((paper_count,), -1, dtype=np.int64)
    for lbl, vidx in label_to_vidx.items():
        paper_label_mapped[paper_label_raw == lbl] = vidx
    venue_count = int(len(unique_labels))
    candidate_venue_ids = np.arange(venue_count, dtype=np.int64)

    node_count = {
        "paper": paper_count,
        "author": author_count,
        "affiliation": affiliation_count,
        "field": field_count,
        "venue": venue_count,
    }
    node_types = ["paper", "author", "affiliation", "field", "venue"]

    feature_files, feature_dims = _save_features(
        out_dir=args.out_dir,
        node_count=node_count,
        paper_feat=paper_feat,
        feat_dim=args.feature_dim,
    )

    relation_entries: List[dict] = []
    meta_graph: List[List[str]] = []

    print("[INFO] Building relation arrays...", flush=True)
    # author -> affiliation  (relation: in), no reliable time, use sentinel.
    e_ai = _load_edge_csv(
        os.path.join(
            ogbn_mag_dir,
            "raw",
            "relations",
            "author___affiliated_with___institution",
            "edge.csv",
        )
    )
    t_ai = np.full((e_ai.shape[0],), NONE_TIME_SENTINEL, dtype=np.int32)
    _add_relation_pair(
        out_dir=args.out_dir,
        relation_entries=relation_entries,
        meta_graph=meta_graph,
        source_type="author",
        target_type="affiliation",
        relation_name="in",
        source_ids=e_ai[:, 0],
        target_ids=e_ai[:, 1],
        edge_times=t_ai,
        node_count=node_count,
    )

    # author -> paper (relation: AP_write_other), time from paper year.
    e_ap = _load_edge_csv(
        os.path.join(
            ogbn_mag_dir,
            "raw",
            "relations",
            "author___writes___paper",
            "edge.csv",
        )
    )
    t_ap = paper_year[e_ap[:, 1]].astype(np.int32, copy=False)
    _add_relation_pair(
        out_dir=args.out_dir,
        relation_entries=relation_entries,
        meta_graph=meta_graph,
        source_type="author",
        target_type="paper",
        relation_name="AP_write_other",
        source_ids=e_ap[:, 0],
        target_ids=e_ap[:, 1],
        edge_times=t_ap,
        node_count=node_count,
    )

    # paper -> paper (relation: PP_cite), time from citing paper (source) year.
    e_pp = _load_edge_csv(
        os.path.join(
            ogbn_mag_dir,
            "raw",
            "relations",
            "paper___cites___paper",
            "edge.csv",
        )
    )
    t_pp = paper_year[e_pp[:, 0]].astype(np.int32, copy=False)
    _add_relation_pair(
        out_dir=args.out_dir,
        relation_entries=relation_entries,
        meta_graph=meta_graph,
        source_type="paper",
        target_type="paper",
        relation_name="PP_cite",
        source_ids=e_pp[:, 0],
        target_ids=e_pp[:, 1],
        edge_times=t_pp,
        node_count=node_count,
    )

    # paper -> field (relation: PF_in_L2), time from paper year.
    e_pf = _load_edge_csv(
        os.path.join(
            ogbn_mag_dir,
            "raw",
            "relations",
            "paper___has_topic___field_of_study",
            "edge.csv",
        )
    )
    t_pf = paper_year[e_pf[:, 0]].astype(np.int32, copy=False)
    _add_relation_pair(
        out_dir=args.out_dir,
        relation_entries=relation_entries,
        meta_graph=meta_graph,
        source_type="paper",
        target_type="field",
        relation_name="PF_in_L2",
        source_ids=e_pf[:, 0],
        target_ids=e_pf[:, 1],
        edge_times=t_pf,
        node_count=node_count,
    )

    # paper -> venue (relation: PV_Journal), time from paper year.
    labeled_papers = np.nonzero(paper_label_mapped >= 0)[0].astype(np.int64, copy=False)
    e_pv_src = labeled_papers
    e_pv_tgt = paper_label_mapped[labeled_papers].astype(np.int64, copy=False)
    t_pv = paper_year[labeled_papers].astype(np.int32, copy=False)
    _add_relation_pair(
        out_dir=args.out_dir,
        relation_entries=relation_entries,
        meta_graph=meta_graph,
        source_type="paper",
        target_type="venue",
        relation_name="PV_Journal",
        source_ids=e_pv_src,
        target_ids=e_pv_tgt,
        edge_times=t_pv,
        node_count=node_count,
    )

    label_dir = os.path.join(args.out_dir, "labels")
    _ensure_dir(label_dir)
    candidate_path = os.path.join("labels", "candidate_venue_ids.npy")
    np.save(os.path.join(args.out_dir, candidate_path), candidate_venue_ids, allow_pickle=False)

    split_files = {}
    split_stats = {}
    split_ids = {}
    for split_name in ("train", "valid", "test"):
        split_path = os.path.join(ogbn_mag_dir, "split", "time", "paper", f"{split_name}.csv")
        split_ids[split_name] = _load_vector_csv(split_path, dtype=np.int64)
        files, stats = _save_split_labels(
            out_dir=args.out_dir,
            split_name=split_name,
            paper_ids=split_ids[split_name],
            paper_year=paper_year,
            paper_label_mapped=paper_label_mapped,
        )
        split_files[split_name] = files
        split_stats[split_name] = stats

    split_max_time = {
        "train": int(np.max(paper_year[split_ids["train"]])),
        "valid": int(np.max(paper_year[split_ids["valid"]])),
        "test": int(np.max(paper_year[split_ids["test"]])),
    }

    task_info = {
        "name": "paper_venue",
        "relation_type": "PV_Journal",
        "reverse_relation_type": "rev_PV_Journal",
        "candidate_ids": candidate_path,
        "splits": split_files,
        "stats": split_stats,
    }

    meta = {
        "schema_version": 1,
        "source_graph": ogbn_mag_dir,
        "none_time_sentinel": int(NONE_TIME_SENTINEL),
        "node_types": node_types,
        "feature_files": feature_files,
        "feature_dims": feature_dims,
        "node_counts": {k: int(v) for k, v in node_count.items()},
        "relations": relation_entries,
        "meta_graph": meta_graph,
        "split_max_time": split_max_time,
        "task": task_info,
    }

    meta_path = os.path.join(args.out_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=True, indent=2)

    print("[OK] Build done.", flush=True)
    print(f"[OK] meta: {meta_path}", flush=True)
    print(
        "[OK] split sizes: "
        + ", ".join([f"{k}={split_stats[k]['paper_count']}" for k in ("train", "valid", "test")]),
        flush=True,
    )
    print(f"[OK] venue classes: {venue_count}", flush=True)


if __name__ == "__main__":
    main()
