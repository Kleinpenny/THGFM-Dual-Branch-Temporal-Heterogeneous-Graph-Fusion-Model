"""
Build shared read-only Paper–Venue base from preprocess_ogbn_mag_w2v.py output (graph_ogbn_mag_w2v.pk).

Mirrors OAG ``build_shared_pv_base.py``:
  - features: concat(node_emb or zeros(400), emb, log10(citation + 0.01))
  - relations: CSR arrays from ``graph.edge_list``
  - task metadata and label .npy files

Differs from OAG in **label splits**: uses OGB official ``split/time/paper/{train,valid,test}.csv``
(not ``_split_by_time`` 2015/2016 cutoffs), consistent with ``build_shared_ogbn_mag_pv_base.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Dict, List, Tuple

import numpy as np
from tqdm.auto import tqdm

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_OGBN_DIR = os.path.abspath(os.path.join(_CUR_DIR, ".."))
_PROJECT_ROOT = os.path.abspath(os.path.join(_OGBN_DIR, ".."))
_OAG_DIR = os.path.join(_PROJECT_ROOT, "OAG")
if _OAG_DIR not in sys.path:
    sys.path.insert(0, _OAG_DIR)

from THGFM.data import renamed_load  # noqa: E402

NONE_TIME_SENTINEL = np.int32(np.iinfo(np.int32).min)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _resolve_graph_path(data_dir: str, graph_file: str) -> str:
    if os.path.isabs(graph_file):
        candidates = [graph_file]
    else:
        candidates = [
            os.path.join(data_dir, graph_file),
            os.path.join(_OGBN_DIR, "data", graph_file),
            os.path.join(_PROJECT_ROOT, "data", graph_file),
        ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Graph file not found. Checked:\n- " + "\n- ".join(candidates))


def _load_graph(graph_path: str):
    with open(graph_path, "rb") as f:
        return renamed_load(f)


def _load_split_ids(ogbn_mag_dir: str, split_name: str) -> np.ndarray:
    path = os.path.join(ogbn_mag_dir, "split", "time", "paper", f"{split_name}.csv")
    arr = np.loadtxt(path, delimiter=",", dtype=np.int64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def _build_features(graph, out_dir: str):
    feat_dir = os.path.join(out_dir, "features")
    _ensure_dir(feat_dir)
    feature_files = {}
    feature_dims = {}
    node_counts = {}

    for node_type in tqdm(graph.get_types(), desc="Build features", dynamic_ncols=True):
        df = graph.node_feature[node_type].sort_index()
        node_ids = df.index.to_numpy(dtype=np.int64)
        expected = np.arange(len(node_ids), dtype=np.int64)
        if not np.array_equal(node_ids, expected):
            raise ValueError(
                f"Node ids for type={node_type} are not contiguous [0..N-1]. "
                "Current shared-base builder expects contiguous ids."
            )

        if "emb" not in df:
            raise KeyError(f"Missing required column 'emb' for node type '{node_type}'.")
        emb = np.asarray(list(df["emb"]), dtype=np.float32)
        citation = np.asarray(list(df["citation"]), dtype=np.float32).reshape(-1, 1)
        # OGBN setting: drop propagated node_emb; keep 128-d emb + 1-d citation feature.
        feat = np.concatenate((emb, np.log10(citation + 0.01)), axis=1).astype(
            np.float32, copy=False
        )

        rel_path = os.path.join("features", f"{node_type}.npy")
        np.save(os.path.join(out_dir, rel_path), feat, allow_pickle=False)
        feature_files[node_type] = rel_path
        feature_dims[node_type] = int(feat.shape[1])
        node_counts[node_type] = int(feat.shape[0])

    return feature_files, feature_dims, node_counts


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


def _find_relation_entry(meta, target_type: str, source_type: str, relation_type: str):
    for rel in meta.get("relations", []):
        if (
            rel.get("target_type") == target_type
            and rel.get("source_type") == source_type
            and rel.get("relation_type") == relation_type
        ):
            return rel
    raise KeyError(
        f"Relation not found in shared meta: ({target_type}, {source_type}, {relation_type})"
    )


def _link_shared_subdir(source_shared_dir: str, out_dir: str, subdir: str):
    src_dir = os.path.join(source_shared_dir, subdir)
    dst_dir = os.path.join(out_dir, subdir)
    _ensure_dir(dst_dir)
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"Missing source subdir: {src_dir}")
    for name in os.listdir(src_dir):
        src_file = os.path.join(src_dir, name)
        if not os.path.isfile(src_file):
            continue
        dst_file = os.path.join(dst_dir, name)
        if os.path.lexists(dst_file):
            os.remove(dst_file)
        rel_src = os.path.relpath(src_file, os.path.dirname(dst_file))
        try:
            os.symlink(rel_src, dst_file)
        except OSError:
            shutil.copy2(src_file, dst_file)


def _paper_venue_from_rev_csr(
    rev_entry: dict,
    source_shared_dir: str,
) -> Dict[int, Tuple[int, int]]:
    """Map paper index -> (venue_id, edge_time) using rev_PV_Journal CSR files."""
    target_ids = np.load(os.path.join(source_shared_dir, rev_entry["target_ids"]))
    indptr = np.load(os.path.join(source_shared_dir, rev_entry["indptr"]))
    source_ids = np.load(os.path.join(source_shared_dir, rev_entry["source_ids"]))
    edge_times = np.load(os.path.join(source_shared_dir, rev_entry["edge_times"]))
    out: Dict[int, Tuple[int, int]] = {}
    for i, paper_id in enumerate(target_ids):
        st = int(indptr[i])
        ed = int(indptr[i + 1])
        if ed > st:
            out[int(paper_id)] = (int(source_ids[st]), int(edge_times[st]))
    return out


def _save_pv_split_labels(out_dir: str, split_name: str, split_map: Dict[int, Tuple[int, int]]):
    label_dir = os.path.join(out_dir, "labels")
    _ensure_dir(label_dir)

    paper_ids_sorted = np.array(sorted(split_map.keys()), dtype=np.int64)
    venue_ids = np.zeros(len(paper_ids_sorted), dtype=np.int64)
    times = np.zeros(len(paper_ids_sorted), dtype=np.int32)
    for i, paper_id in enumerate(paper_ids_sorted):
        venue_id, t = split_map[int(paper_id)]
        venue_ids[i] = int(venue_id)
        times[i] = int(t)

    paper_ids_path = os.path.join("labels", f"{split_name}__paper_ids.npy")
    venue_ids_path = os.path.join("labels", f"{split_name}__venue_ids.npy")
    times_path = os.path.join("labels", f"{split_name}__times.npy")
    np.save(os.path.join(out_dir, paper_ids_path), paper_ids_sorted, allow_pickle=False)
    np.save(os.path.join(out_dir, venue_ids_path), venue_ids, allow_pickle=False)
    np.save(os.path.join(out_dir, times_path), times, allow_pickle=False)

    return (
        {
            "paper_ids": paper_ids_path,
            "venue_ids": venue_ids_path,
            "times": times_path,
        },
        {"paper_count": int(len(paper_ids_sorted))},
    )


def _build_ogb_task_labels_from_graph(graph, ogbn_mag_dir: str, out_dir: str):
    rel = "PV_Journal"
    rev_rel = "rev_PV_Journal"
    if (
        "venue" not in graph.edge_list
        or "paper" not in graph.edge_list["venue"]
        or rel not in graph.edge_list["venue"]["paper"]
    ):
        raise KeyError(f"Relation {rel} not found in graph.edge_list['venue']['paper'].")
    if (
        "paper" not in graph.edge_list
        or "venue" not in graph.edge_list["paper"]
        or rev_rel not in graph.edge_list["paper"]["venue"]
    ):
        raise KeyError(f"Relation {rev_rel} not found in graph.edge_list['paper']['venue'].")

    candidate_ids = np.array(
        sorted(graph.edge_list["venue"]["paper"][rel].keys()),
        dtype=np.int64,
    )
    label_dir = os.path.join(out_dir, "labels")
    _ensure_dir(label_dir)
    candidate_path = os.path.join("labels", "candidate_venue_ids.npy")
    np.save(os.path.join(out_dir, candidate_path), candidate_ids, allow_pickle=False)

    rev_map = graph.edge_list["paper"]["venue"][rev_rel]
    split_files = {}
    split_stats = {}
    for split_name in ("train", "valid", "test"):
        split_map: Dict[int, Tuple[int, int]] = {}
        for pid in _load_split_ids(ogbn_mag_dir, split_name):
            pid = int(pid)
            if pid not in rev_map:
                continue
            vm = rev_map[pid]
            if not vm:
                continue
            venue_id, t_raw = next(iter(vm.items()))
            split_map[pid] = (int(venue_id), int(t_raw))
        split_files[split_name], split_stats[split_name] = _save_pv_split_labels(
            out_dir=out_dir,
            split_name=split_name,
            split_map=split_map,
        )

    return {
        "name": "paper_venue",
        "relation_type": rel,
        "reverse_relation_type": rev_rel,
        "candidate_ids": candidate_path,
        "splits": split_files,
        "stats": split_stats,
    }


def _build_ogb_task_labels_from_shared_meta(
    source_shared_dir: str,
    ogbn_mag_dir: str,
    out_dir: str,
    source_meta: dict,
):
    rel = "PV_Journal"
    rev_rel = "rev_PV_Journal"
    rev_entry = _find_relation_entry(source_meta, "paper", "venue", rev_rel)
    cand_entry = _find_relation_entry(source_meta, "venue", "paper", rel)

    label_dir = os.path.join(out_dir, "labels")
    _ensure_dir(label_dir)
    cand_ids = np.load(os.path.join(source_shared_dir, cand_entry["target_ids"]))
    candidate_path = os.path.join("labels", "candidate_venue_ids.npy")
    np.save(
        os.path.join(out_dir, candidate_path),
        cand_ids.astype(np.int64, copy=False),
        allow_pickle=False,
    )

    paper_to_vt = _paper_venue_from_rev_csr(rev_entry, source_shared_dir)

    split_files = {}
    split_stats = {}
    for split_name in ("train", "valid", "test"):
        split_map: Dict[int, Tuple[int, int]] = {}
        for pid in _load_split_ids(ogbn_mag_dir, split_name):
            pid = int(pid)
            if pid in paper_to_vt:
                split_map[pid] = paper_to_vt[pid]
        split_files[split_name], split_stats[split_name] = _save_pv_split_labels(
            out_dir=out_dir,
            split_name=split_name,
            split_map=split_map,
        )

    return {
        "name": "paper_venue",
        "relation_type": rel,
        "reverse_relation_type": rev_rel,
        "candidate_ids": candidate_path,
        "splits": split_files,
        "stats": split_stats,
    }


def _split_max_time_from_ogb(ogbn_mag_dir: str, paper_year: np.ndarray) -> dict:
    out = {}
    for name in ("train", "valid", "test"):
        ids = _load_split_ids(ogbn_mag_dir, name)
        out[name] = int(np.max(paper_year[ids]))
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Build shared PV base from graph_ogbn_mag_w2v.pk (OGB splits for labels)."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(_OGBN_DIR, "data"),
        help="Directory to search for --graph_file",
    )
    parser.add_argument(
        "--graph_file",
        type=str,
        default="graph_ogbn_mag_w2v.pk",
        help="Preprocessed graph pickle (from preprocess_ogbn_mag_w2v.py).",
    )
    parser.add_argument(
        "--ogbn_mag_dir",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "data", "ogbn_mag"),
        help="Extracted ogbn-mag root (for split/time/paper/*.csv).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(_OGBN_DIR, "data", "shared_pv_base_ogbn_mag_graph"),
        help="Output directory for shared base files.",
    )
    parser.add_argument(
        "--source_shared_base_dir",
        type=str,
        default="",
        help="Optional: reuse features/relations from an existing shared base (symlink); "
        "rebuild OGB label .npy from ogbn_mag_dir.",
    )
    args = parser.parse_args()

    _ensure_dir(args.out_dir)
    ogbn_mag_dir = os.path.abspath(args.ogbn_mag_dir)

    if args.source_shared_base_dir:
        source_shared_dir = os.path.abspath(args.source_shared_base_dir)
        source_meta_path = os.path.join(source_shared_dir, "meta.json")
        with open(source_meta_path, "r", encoding="utf-8") as f:
            source_meta = json.load(f)
        _link_shared_subdir(source_shared_dir, args.out_dir, "features")
        _link_shared_subdir(source_shared_dir, args.out_dir, "relations")
        task_info = _build_ogb_task_labels_from_shared_meta(
            source_shared_dir=source_shared_dir,
            ogbn_mag_dir=ogbn_mag_dir,
            out_dir=args.out_dir,
            source_meta=source_meta,
        )
        paper_year = np.loadtxt(
            os.path.join(ogbn_mag_dir, "raw", "node-feat", "paper", "node_year.csv"),
            delimiter=",",
            dtype=np.int32,
        )
        split_max_time = _split_max_time_from_ogb(ogbn_mag_dir, paper_year)
        meta = {
            "schema_version": 1,
            "source_graph": source_meta.get("source_graph", ""),
            "none_time_sentinel": int(source_meta["none_time_sentinel"]),
            "node_types": source_meta["node_types"],
            "feature_files": source_meta["feature_files"],
            "feature_dims": source_meta["feature_dims"],
            "node_counts": source_meta["node_counts"],
            "relations": source_meta["relations"],
            "meta_graph": source_meta["meta_graph"],
            "split_max_time": split_max_time,
            "task": task_info,
        }
    else:
        graph_path = _resolve_graph_path(args.data_dir, args.graph_file)
        print(f"[INFO] Loading graph: {graph_path}", flush=True)
        graph = _load_graph(graph_path)

        feature_files, feature_dims, node_counts = _build_features(graph, args.out_dir)
        relation_entries, meta_graph = _build_relations(graph, args.out_dir)
        task_info = _build_ogb_task_labels_from_graph(graph, ogbn_mag_dir, args.out_dir)

        paper_year = np.loadtxt(
            os.path.join(ogbn_mag_dir, "raw", "node-feat", "paper", "node_year.csv"),
            delimiter=",",
            dtype=np.int32,
        )
        split_max_time = _split_max_time_from_ogb(ogbn_mag_dir, paper_year)

        meta = {
            "schema_version": 1,
            "source_graph": graph_path,
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
    for k in ("train", "valid", "test"):
        print(f"[OK] split {k}: {meta['task']['stats'][k]['paper_count']} papers", flush=True)


if __name__ == "__main__":
    main()
