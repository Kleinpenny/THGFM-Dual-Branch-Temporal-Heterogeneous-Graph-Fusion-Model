import argparse
import json
import os
import shutil
import sys
from collections import defaultdict

import numpy as np
from tqdm.auto import tqdm


_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_OAG_DIR = os.path.abspath(os.path.join(_CUR_DIR, ".."))
if _OAG_DIR not in sys.path:
    sys.path.insert(0, _OAG_DIR)

from THGFM.data import renamed_load  # noqa: E402


NONE_TIME_SENTINEL = np.int32(np.iinfo(np.int32).min)


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _resolve_graph_path(data_dir: str, domain: str, graph_file: str) -> str:
    if graph_file:
        if os.path.isabs(graph_file):
            candidates = [graph_file]
        else:
            candidates = [
                os.path.join(data_dir, graph_file),
                os.path.join(os.path.abspath(os.path.join(_OAG_DIR, "..")), "data", graph_file),
            ]
    else:
        name = f"graph{domain}.pk"
        candidates = [
            os.path.join(data_dir, name),
            os.path.join(os.path.abspath(os.path.join(_OAG_DIR, "..")), "data", name),
        ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "Graph file not found. Checked:\n- "
        + "\n- ".join(candidates)
    )


def _load_graph(graph_path: str):
    with open(graph_path, "rb") as f:
        return renamed_load(f)


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

        if "node_emb" in df:
            node_emb = np.asarray(list(df["node_emb"]), dtype=np.float32)
        else:
            node_emb = np.zeros((len(df), 400), dtype=np.float32)
        if "emb" not in df:
            raise KeyError(f"Missing required column 'emb' for node type '{node_type}'.")
        emb = np.asarray(list(df["emb"]), dtype=np.float32)
        citation = np.asarray(list(df["citation"]), dtype=np.float32).reshape(-1, 1)
        feat = np.concatenate((node_emb, emb, np.log10(citation + 0.01)), axis=1).astype(np.float32, copy=False)

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


def _split_by_time(t: int):
    if t < 2015:
        return "train"
    if t <= 2016:
        return "valid"
    return "test"


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


def _build_pf_labels_from_shared_meta(source_shared_dir: str, out_dir: str, source_meta, field_level: str):
    pf_rel = f"PF_in_{field_level}"
    rev_pf_rel = f"rev_{pf_rel}"
    rev_entry = _find_relation_entry(source_meta, "paper", "field", rev_pf_rel)
    cand_entry = _find_relation_entry(source_meta, "field", "paper", pf_rel)

    cand_ids = np.load(
        os.path.join(source_shared_dir, cand_entry["target_ids"]),
        mmap_mode="r",
    )
    target_ids = np.load(
        os.path.join(source_shared_dir, rev_entry["target_ids"]),
        mmap_mode="r",
    )
    indptr = np.load(
        os.path.join(source_shared_dir, rev_entry["indptr"]),
        mmap_mode="r",
    )
    source_ids = np.load(
        os.path.join(source_shared_dir, rev_entry["source_ids"]),
        mmap_mode="r",
    )
    edge_times = np.load(
        os.path.join(source_shared_dir, rev_entry["edge_times"]),
        mmap_mode="r",
    )
    none_sentinel = int(source_meta["none_time_sentinel"])

    label_dir = os.path.join(out_dir, "labels")
    _ensure_dir(label_dir)
    candidate_path = os.path.join("labels", "candidate_field_ids.npy")
    np.save(os.path.join(out_dir, candidate_path), cand_ids.astype(np.int64, copy=False), allow_pickle=False)

    split_maps = {
        "train": defaultdict(lambda: [set(), 0]),
        "valid": defaultdict(lambda: [set(), 0]),
        "test": defaultdict(lambda: [set(), 0]),
    }
    for i, paper_id in enumerate(target_ids):
        st = int(indptr[i])
        ed = int(indptr[i + 1])
        for field_id, t_raw in zip(source_ids[st:ed], edge_times[st:ed]):
            t = int(t_raw)
            if t == none_sentinel:
                continue
            split_name = _split_by_time(t)
            rec = split_maps[split_name][int(paper_id)]
            rec[0].add(int(field_id))
            if rec[1] == 0:
                rec[1] = t

    split_files = {}
    split_stats = {}
    for split_name in ("train", "valid", "test"):
        split_files[split_name], split_stats[split_name] = _save_pf_split_labels(
            out_dir=out_dir,
            split_name=split_name,
            split_map=split_maps[split_name],
        )

    return {
        "name": "paper_field",
        "field_level": field_level,
        "relation_type": pf_rel,
        "reverse_relation_type": rev_pf_rel,
        "candidate_ids": candidate_path,
        "splits": split_files,
        "stats": split_stats,
    }


def _save_pf_split_labels(out_dir: str, split_name: str, split_map):
    label_dir = os.path.join(out_dir, "labels")
    _ensure_dir(label_dir)

    paper_ids_sorted = np.array(sorted(split_map.keys()), dtype=np.int64)
    times = np.zeros(len(paper_ids_sorted), dtype=np.int32)
    offsets = np.zeros(len(paper_ids_sorted) + 1, dtype=np.int64)
    label_buf = []

    for i, paper_id in enumerate(paper_ids_sorted):
        labels, t = split_map[int(paper_id)]
        labels_sorted = np.array(sorted(labels), dtype=np.int64)
        if len(labels_sorted) == 0:
            continue
        label_buf.append(labels_sorted)
        offsets[i + 1] = offsets[i] + len(labels_sorted)
        times[i] = int(t)

    if len(label_buf) > 0:
        label_ids = np.concatenate(label_buf).astype(np.int64, copy=False)
    else:
        label_ids = np.empty((0,), dtype=np.int64)

    paper_ids_path = os.path.join("labels", f"{split_name}__paper_ids.npy")
    times_path = os.path.join("labels", f"{split_name}__times.npy")
    offsets_path = os.path.join("labels", f"{split_name}__label_offsets.npy")
    label_ids_path = os.path.join("labels", f"{split_name}__label_ids.npy")
    np.save(os.path.join(out_dir, paper_ids_path), paper_ids_sorted, allow_pickle=False)
    np.save(os.path.join(out_dir, times_path), times, allow_pickle=False)
    np.save(os.path.join(out_dir, offsets_path), offsets, allow_pickle=False)
    np.save(os.path.join(out_dir, label_ids_path), label_ids, allow_pickle=False)

    mean_labels = float(len(label_ids) / len(paper_ids_sorted)) if len(paper_ids_sorted) > 0 else 0.0
    return (
        {
            "paper_ids": paper_ids_path,
            "times": times_path,
            "label_offsets": offsets_path,
            "label_ids": label_ids_path,
        },
        {
            "paper_count": int(len(paper_ids_sorted)),
            "label_count": int(len(label_ids)),
            "avg_labels_per_paper": mean_labels,
        },
    )


def _build_pf_labels(graph, out_dir: str, field_level: str):
    pf_rel = f"PF_in_{field_level}"
    rev_pf_rel = f"rev_{pf_rel}"
    if (
        "field" not in graph.edge_list
        or "paper" not in graph.edge_list["field"]
        or pf_rel not in graph.edge_list["field"]["paper"]
    ):
        raise KeyError(
            f"Relation {pf_rel} not found in graph.edge_list['field']['paper']."
        )
    if (
        "paper" not in graph.edge_list
        or "field" not in graph.edge_list["paper"]
        or rev_pf_rel not in graph.edge_list["paper"]["field"]
    ):
        raise KeyError(
            f"Relation {rev_pf_rel} not found in graph.edge_list['paper']['field']."
        )

    candidate_ids = np.array(
        list(graph.edge_list["field"]["paper"][pf_rel].keys()),
        dtype=np.int64,
    )
    label_dir = os.path.join(out_dir, "labels")
    _ensure_dir(label_dir)
    candidate_path = os.path.join("labels", "candidate_field_ids.npy")
    np.save(os.path.join(out_dir, candidate_path), candidate_ids, allow_pickle=False)

    split_maps = {
        "train": defaultdict(lambda: [set(), 0]),
        "valid": defaultdict(lambda: [set(), 0]),
        "test": defaultdict(lambda: [set(), 0]),
    }
    rev_map = graph.edge_list["paper"]["field"][rev_pf_rel]
    for paper_id, field_time_map in tqdm(
        rev_map.items(),
        total=len(rev_map),
        desc=f"Build PF-{field_level} labels",
        dynamic_ncols=True,
    ):
        paper_id = int(paper_id)
        for field_id, t in field_time_map.items():
            if not isinstance(t, (int, np.integer)):
                continue
            split_name = _split_by_time(int(t))
            rec = split_maps[split_name][paper_id]
            rec[0].add(int(field_id))
            rec[1] = int(t)

    split_files = {}
    split_stats = {}
    for split_name in ("train", "valid", "test"):
        split_files[split_name], split_stats[split_name] = _save_pf_split_labels(
            out_dir=out_dir,
            split_name=split_name,
            split_map=split_maps[split_name],
        )

    return {
        "name": "paper_field",
        "field_level": field_level,
        "relation_type": pf_rel,
        "reverse_relation_type": rev_pf_rel,
        "candidate_ids": candidate_path,
        "splits": split_files,
        "stats": split_stats,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build shared read-only Paper-Field base from graph*.pk"
    )
    project_root = os.path.abspath(os.path.join(_OAG_DIR, ".."))
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(project_root, "data"),
        help="Directory containing graph*.pk",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="_CS",
        help="Legacy graph domain suffix for graph{domain}.pk",
    )
    parser.add_argument(
        "--graph_file",
        type=str,
        default="",
        help="Graph pickle filename. If set, overrides --domain.",
    )
    parser.add_argument(
        "--field_level",
        type=str,
        default="L2",
        choices=["L1", "L2"],
        help="Field level for labels.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(project_root, "data", "shared_pf_base_L2"),
        help="Output directory for shared base files.",
    )
    parser.add_argument(
        "--source_shared_base_dir",
        type=str,
        default="",
        help="Optional existing shared base directory to reuse arrays from (recommended).",
    )
    args = parser.parse_args()

    _ensure_dir(args.out_dir)
    if args.source_shared_base_dir:
        source_meta_path = os.path.join(args.source_shared_base_dir, "meta.json")
        with open(source_meta_path, "r", encoding="utf-8") as f:
            source_meta = json.load(f)
        _link_shared_subdir(args.source_shared_base_dir, args.out_dir, "features")
        _link_shared_subdir(args.source_shared_base_dir, args.out_dir, "relations")
        task_info = _build_pf_labels_from_shared_meta(
            source_shared_dir=args.source_shared_base_dir,
            out_dir=args.out_dir,
            source_meta=source_meta,
            field_level=args.field_level,
        )
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
            "split_max_time": source_meta.get(
                "split_max_time",
                {"train": 2014, "valid": 2016, "test": 2020},
            ),
            "task": task_info,
        }
    else:
        graph_path = _resolve_graph_path(args.data_dir, args.domain, args.graph_file)
        print(f"[INFO] Loading graph: {graph_path}", flush=True)
        graph = _load_graph(graph_path)

        feature_files, feature_dims, node_counts = _build_features(graph, args.out_dir)
        relation_entries, meta_graph = _build_relations(graph, args.out_dir)
        task_info = _build_pf_labels(graph, args.out_dir, field_level=args.field_level)

        all_times = [int(t) for t in graph.times.keys() if isinstance(t, (int, np.integer))]
        time_max = int(max(all_times)) if all_times else 2016

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
            "split_max_time": {
                "train": 2014,
                "valid": 2016,
                "test": time_max,
            },
            "task": task_info,
        }

    meta_path = os.path.join(args.out_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=True, indent=2)
    print(f"[OK] Shared PF base written to: {args.out_dir}", flush=True)
    print(f"[OK] Metadata: {meta_path}", flush=True)


if __name__ == "__main__":
    main()
