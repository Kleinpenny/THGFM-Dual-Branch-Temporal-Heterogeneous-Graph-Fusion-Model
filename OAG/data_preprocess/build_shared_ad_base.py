import argparse
import json
import os
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


def _flatten_group_arrays(group_arrays):
    offsets = np.zeros(len(group_arrays) + 1, dtype=np.int64)
    for i, arr in enumerate(group_arrays):
        offsets[i + 1] = offsets[i] + len(arr)
    if len(group_arrays) > 0 and offsets[-1] > 0:
        flat = np.concatenate(group_arrays)
    else:
        flat = np.empty((0,), dtype=np.int64)
    return offsets, flat


def _save_split_triplets(out_dir: str, split_name: str, per_group_triplets):
    group_offsets = np.zeros(len(per_group_triplets) + 1, dtype=np.int64)
    paper_ids_buf = []
    author_pos_buf = []
    times_buf = []
    nonempty = []

    for gid, tri in enumerate(per_group_triplets):
        if len(tri) > 0:
            arr = np.asarray(tri, dtype=np.int64)
            paper_ids_buf.append(arr[:, 0].astype(np.int64, copy=False))
            author_pos_buf.append(arr[:, 1].astype(np.int32, copy=False))
            times_buf.append(arr[:, 2].astype(np.int32, copy=False))
            nonempty.append(gid)
            group_offsets[gid + 1] = group_offsets[gid] + arr.shape[0]
        else:
            group_offsets[gid + 1] = group_offsets[gid]

    if len(paper_ids_buf) > 0:
        paper_ids = np.concatenate(paper_ids_buf)
        author_pos = np.concatenate(author_pos_buf)
        times = np.concatenate(times_buf)
    else:
        paper_ids = np.empty((0,), dtype=np.int64)
        author_pos = np.empty((0,), dtype=np.int32)
        times = np.empty((0,), dtype=np.int32)

    grp_dir = os.path.join(out_dir, "groups")
    _ensure_dir(grp_dir)
    np.save(os.path.join(grp_dir, f"{split_name}__offsets.npy"), group_offsets, allow_pickle=False)
    np.save(os.path.join(grp_dir, f"{split_name}__paper_ids.npy"), paper_ids, allow_pickle=False)
    np.save(os.path.join(grp_dir, f"{split_name}__author_pos.npy"), author_pos, allow_pickle=False)
    np.save(os.path.join(grp_dir, f"{split_name}__times.npy"), times, allow_pickle=False)
    np.save(
        os.path.join(grp_dir, f"{split_name}__group_ids.npy"),
        np.asarray(nonempty, dtype=np.int64),
        allow_pickle=False,
    )
    return int(len(nonempty)), int(len(paper_ids))


def _build_ad_groups(graph, out_dir: str):
    apd = graph.edge_list["author"]["paper"]["rev_AP_write_first"]
    first_author_ids = {aid for aid, papers in apd.items() if len(papers) >= 2}

    author_df = graph.node_feature["author"][["name"]].sort_index()
    name_to_author_ids = defaultdict(list)
    for aid, row in tqdm(author_df.iterrows(), total=len(author_df), desc="Group authors by name", dynamic_ncols=True):
        aid = int(aid)
        if aid in first_author_ids:
            name_to_author_ids[str(row["name"])].append(aid)

    # Keep deterministic order for reproducibility.
    group_author_lists = [
        np.asarray(sorted(name_to_author_ids[name]), dtype=np.int64)
        for name in sorted(name_to_author_ids.keys())
        if len(name_to_author_ids[name]) >= 4
    ]

    per_group_train = []
    per_group_valid = []
    per_group_test = []
    for author_ids in tqdm(group_author_lists, desc="Build split triplets", dynamic_ncols=True):
        train_tri = []
        valid_tri = []
        test_tri = []
        for local_idx, aid in enumerate(author_ids):
            paper_dict = apd[int(aid)]
            for p_id, t in paper_dict.items():
                if not isinstance(t, (int, np.integer)):
                    continue
                t = int(t)
                rec = (int(p_id), int(local_idx), t)
                if t < 2015:
                    train_tri.append(rec)
                elif t <= 2016:
                    valid_tri.append(rec)
                else:
                    test_tri.append(rec)
        per_group_train.append(train_tri)
        per_group_valid.append(valid_tri)
        per_group_test.append(test_tri)

    grp_dir = os.path.join(out_dir, "groups")
    _ensure_dir(grp_dir)
    author_offsets, author_ids_flat = _flatten_group_arrays(group_author_lists)
    np.save(os.path.join(grp_dir, "group_authors__offsets.npy"), author_offsets, allow_pickle=False)
    np.save(os.path.join(grp_dir, "group_authors__ids.npy"), author_ids_flat, allow_pickle=False)

    train_nonempty, train_pairs = _save_split_triplets(out_dir, "train", per_group_train)
    valid_nonempty, valid_pairs = _save_split_triplets(out_dir, "valid", per_group_valid)
    test_nonempty, test_pairs = _save_split_triplets(out_dir, "test", per_group_test)

    return {
        "group_count": int(len(group_author_lists)),
        "train_nonempty_groups": train_nonempty,
        "valid_nonempty_groups": valid_nonempty,
        "test_nonempty_groups": test_nonempty,
        "train_pair_count": train_pairs,
        "valid_pair_count": valid_pairs,
        "test_pair_count": test_pairs,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build shared read-only AD base from graph*.pk"
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
        "--out_dir",
        type=str,
        default=os.path.join(project_root, "data", "shared_ad_base"),
        help="Output directory for shared base files.",
    )
    args = parser.parse_args()

    _ensure_dir(args.out_dir)
    graph_path = _resolve_graph_path(args.data_dir, args.domain, args.graph_file)
    print(f"[INFO] Loading graph: {graph_path}", flush=True)
    graph = _load_graph(graph_path)

    feature_files, feature_dims, node_counts = _build_features(graph, args.out_dir)
    relation_entries, meta_graph = _build_relations(graph, args.out_dir)
    group_stats = _build_ad_groups(graph, args.out_dir)

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
        "group_stats": group_stats,
    }

    meta_path = os.path.join(args.out_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=True, indent=2)
    print(f"[OK] Shared AD base written to: {args.out_dir}", flush=True)
    print(f"[OK] Metadata: {meta_path}", flush=True)


if __name__ == "__main__":
    main()
