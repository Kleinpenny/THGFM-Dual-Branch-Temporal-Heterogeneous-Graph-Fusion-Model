import argparse
import json
import os
import sys

import numpy as np
from tqdm.auto import tqdm


_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_ARXIV_PV_DIR = os.path.abspath(os.path.join(_CUR_DIR, ".."))
_HTAG_DIR = os.path.abspath(os.path.join(_ARXIV_PV_DIR, ".."))
if _HTAG_DIR not in sys.path:
    sys.path.insert(0, _HTAG_DIR)

from THGFM.data import renamed_load  # noqa: E402


NONE_TIME_SENTINEL = np.int32(np.iinfo(np.int32).min)


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _load_graph(graph_path: str):
    with open(graph_path, "rb") as f:
        return renamed_load(f)


def _save_relation_arrays(out_dir: str, key: str, target_ids, indptr, source_ids, edge_times):
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


def _build_features_emb_only(graph, out_dir: str):
    feat_dir = os.path.join(out_dir, "features")
    _ensure_dir(feat_dir)
    feature_files = {}
    feature_dims = {}
    node_counts = {}

    for node_type in tqdm(graph.get_types(), desc="Build features (emb only)", dynamic_ncols=True):
        df = graph.node_feature[node_type].sort_index()
        node_ids = df.index.to_numpy(dtype=np.int64)
        expected = np.arange(len(node_ids), dtype=np.int64)
        if not np.array_equal(node_ids, expected):
            raise ValueError(
                f"Node ids for type={node_type} are not contiguous [0..N-1]. "
                "Shared-base builder expects contiguous ids."
            )
        if "emb" not in df:
            raise KeyError(f"Missing required column 'emb' for node type '{node_type}'.")
        feat = np.asarray(list(df["emb"]), dtype=np.float32)

        rel_path = os.path.join("features", f"{node_type}.npy")
        np.save(os.path.join(out_dir, rel_path), feat, allow_pickle=False)
        feature_files[node_type] = rel_path
        feature_dims[node_type] = int(feat.shape[1])
        node_counts[node_type] = int(feat.shape[0])

    return feature_files, feature_dims, node_counts


def _build_relations(graph, out_dir: str):
    relation_entries = []
    meta_graph = []

    target_types = sorted(list(graph.edge_list.keys()))
    for target_type in tqdm(target_types, desc="Build relation arrays", dynamic_ncols=True):
        source_types = sorted(list(graph.edge_list[target_type].keys()))
        for source_type in source_types:
            relation_types = sorted(list(graph.edge_list[target_type][source_type].keys()))
            for relation_type in relation_types:
                rel_dict = graph.edge_list[target_type][source_type][relation_type]
                target_ids = np.array(sorted(rel_dict.keys()), dtype=np.int64)
                indptr = np.zeros(len(target_ids) + 1, dtype=np.int64)
                source_buf = []
                time_buf = []

                for i, tid in enumerate(target_ids):
                    src_time_items = sorted(rel_dict[tid].items(), key=lambda x: x[0])
                    if src_time_items:
                        src_ids = np.fromiter((x[0] for x in src_time_items), dtype=np.int64)
                        times = np.fromiter(
                            (
                                int(x[1]) if isinstance(x[1], (int, np.integer)) else int(NONE_TIME_SENTINEL)
                                for x in src_time_items
                            ),
                            dtype=np.int32,
                            count=len(src_time_items),
                        )
                        source_buf.append(src_ids)
                        time_buf.append(times)
                    indptr[i + 1] = indptr[i] + len(src_time_items)

                if len(source_buf) > 0:
                    source_ids = np.concatenate(source_buf).astype(np.int64, copy=False)
                    edge_times = np.concatenate(time_buf).astype(np.int32, copy=False)
                else:
                    source_ids = np.empty((0,), dtype=np.int64)
                    edge_times = np.empty((0,), dtype=np.int32)

                key = f"{target_type}__{source_type}__{relation_type}"
                files = _save_relation_arrays(
                    out_dir,
                    key,
                    target_ids=target_ids,
                    indptr=indptr,
                    source_ids=source_ids,
                    edge_times=edge_times,
                )
                relation_entries.append(
                    {
                        "target_type": target_type,
                        "source_type": source_type,
                        "relation_type": relation_type,
                        **files,
                    }
                )
                meta_graph.append([target_type, source_type, relation_type])

    return relation_entries, meta_graph


def _save_label_split(out_dir: str, split_name: str, paper_ids, venue_ids, times):
    label_dir = os.path.join(out_dir, "labels")
    _ensure_dir(label_dir)
    paper_ids_path = os.path.join("labels", f"{split_name}__paper_ids.npy")
    venue_ids_path = os.path.join("labels", f"{split_name}__venue_ids.npy")
    times_path = os.path.join("labels", f"{split_name}__times.npy")
    np.save(os.path.join(out_dir, paper_ids_path), paper_ids.astype(np.int64, copy=False), allow_pickle=False)
    np.save(os.path.join(out_dir, venue_ids_path), venue_ids.astype(np.int64, copy=False), allow_pickle=False)
    np.save(os.path.join(out_dir, times_path), times.astype(np.int32, copy=False), allow_pickle=False)
    return {
        "paper_ids": paper_ids_path,
        "venue_ids": venue_ids_path,
        "times": times_path,
    }, {
        "paper_count": int(len(paper_ids)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build shared read-only Paper-Venue base for HTAG ArXiv"
    )
    parser.add_argument(
        "--graph_file",
        type=str,
        default=os.path.join(_ARXIV_PV_DIR, "data", "graph_arxiv_HTAG.pk"),
        help="Path to graph_arxiv_HTAG.pk",
    )
    parser.add_argument(
        "--split_dir",
        type=str,
        default=os.path.join(_ARXIV_PV_DIR, "data", "arxiv_splits"),
        help="Directory with train/valid/test split npy files.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(_ARXIV_PV_DIR, "data", "shared_pv_base_HTAG_arxiv"),
        help="Output directory for shared base files.",
    )
    args = parser.parse_args()

    _ensure_dir(args.out_dir)
    print(f"[INFO] Loading graph: {args.graph_file}", flush=True)
    graph = _load_graph(args.graph_file)

    feature_files, feature_dims, node_counts = _build_features_emb_only(graph, args.out_dir)
    relation_entries, meta_graph = _build_relations(graph, args.out_dir)

    candidate_split_path = os.path.join(args.split_dir, "candidate_venue_ids.npy")
    if os.path.exists(candidate_split_path):
        candidate_ids = np.load(candidate_split_path, mmap_mode="r").astype(np.int64, copy=False)
    else:
        candidate_ids = np.array(sorted(list(graph.edge_list["venue"]["paper"]["PV_Journal"].keys())), dtype=np.int64)
    label_dir = os.path.join(args.out_dir, "labels")
    _ensure_dir(label_dir)
    candidate_path = os.path.join("labels", "candidate_venue_ids.npy")
    np.save(os.path.join(args.out_dir, candidate_path), candidate_ids, allow_pickle=False)

    split_files = {}
    split_stats = {}
    split_max_time = {}
    for split_name in ("train", "valid", "test"):
        paper_ids = np.load(
            os.path.join(args.split_dir, f"{split_name}__paper_ids.npy"),
            mmap_mode="r",
        ).astype(np.int64, copy=False)
        venue_ids = np.load(
            os.path.join(args.split_dir, f"{split_name}__venue_ids.npy"),
            mmap_mode="r",
        ).astype(np.int64, copy=False)
        times = np.load(
            os.path.join(args.split_dir, f"{split_name}__times.npy"),
            mmap_mode="r",
        ).astype(np.int32, copy=False)
        split_files[split_name], split_stats[split_name] = _save_label_split(
            args.out_dir, split_name, paper_ids, venue_ids, times
        )
        split_max_time[split_name] = int(times.max()) if len(times) > 0 else 0

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
        "source_graph": args.graph_file,
        "none_time_sentinel": int(NONE_TIME_SENTINEL),
        "node_types": graph.get_types(),
        "feature_files": feature_files,
        "feature_dims": feature_dims,
        "node_counts": node_counts,
        "relations": relation_entries,
        "meta_graph": meta_graph,
        "split_max_time": split_max_time,
        "task": task_info,
    }

    meta_path = os.path.join(args.out_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=True, indent=2)
    print(f"[OK] Shared PV base written to: {args.out_dir}", flush=True)
    print(f"[OK] Metadata: {meta_path}", flush=True)
    print(
        "[OK] split sizes: "
        + ", ".join([f"{k}={split_stats[k]['paper_count']}" for k in ("train", "valid", "test")]),
        flush=True,
    )


if __name__ == "__main__":
    main()
