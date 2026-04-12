import json
import os
from collections import defaultdict

import numpy as np
import torch


class RelationCSR:
    def __init__(self, target_ids, indptr, source_ids, edge_times, none_time_sentinel):
        self.target_ids = target_ids
        self.indptr = indptr
        self.source_ids = source_ids
        self.edge_times = edge_times
        self.none_time_sentinel = none_time_sentinel

    def get_neighbors(self, target_id: int):
        pos = np.searchsorted(self.target_ids, target_id)
        if pos >= len(self.target_ids) or int(self.target_ids[pos]) != target_id:
            return (
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.int32),
            )
        st = int(self.indptr[pos])
        ed = int(self.indptr[pos + 1])
        return self.source_ids[st:ed], self.edge_times[st:ed]


class SharedGraphBase:
    def __init__(self, shared_dir: str):
        meta_path = os.path.join(shared_dir, "meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.shared_dir = shared_dir
        self.node_types = list(self.meta["node_types"])
        self.none_time_sentinel = int(self.meta["none_time_sentinel"])
        self.feature_dims = {k: int(v) for k, v in self.meta["feature_dims"].items()}
        self.meta_graph = [tuple(x) for x in self.meta["meta_graph"]]
        self.split_max_time = {k: int(v) for k, v in self.meta["split_max_time"].items()}

        self.features = {}
        for node_type, rel_path in self.meta["feature_files"].items():
            arr = np.load(os.path.join(shared_dir, rel_path), mmap_mode="r")
            self.features[node_type] = torch.from_numpy(arr)

        self.relations = defaultdict(lambda: defaultdict(dict))
        for rel in self.meta["relations"]:
            target_type = rel["target_type"]
            source_type = rel["source_type"]
            relation_type = rel["relation_type"]
            target_ids = np.load(os.path.join(shared_dir, rel["target_ids"]), mmap_mode="r")
            indptr = np.load(os.path.join(shared_dir, rel["indptr"]), mmap_mode="r")
            source_ids = np.load(os.path.join(shared_dir, rel["source_ids"]), mmap_mode="r")
            edge_times = np.load(os.path.join(shared_dir, rel["edge_times"]), mmap_mode="r")
            self.relations[target_type][source_type][relation_type] = RelationCSR(
                target_ids=target_ids,
                indptr=indptr,
                source_ids=source_ids,
                edge_times=edge_times,
                none_time_sentinel=self.none_time_sentinel,
            )

    def load_npy(self, rel_path: str):
        return np.load(os.path.join(self.shared_dir, rel_path), mmap_mode="r")


def sample_subgraph_shared(base, split_name: str, inp, sampled_depth, sampled_number, rng):
    max_time_in_range = int(base.split_max_time[split_name])

    layer_data = defaultdict(dict)
    budget = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))

    def add_budget(target_type, target_id, target_time):
        source_types = base.relations.get(target_type, {})
        for source_type, relation_map in source_types.items():
            for relation_type, relation in relation_map.items():
                if relation_type == "self":
                    continue
                source_ids, edge_times = relation.get_neighbors(target_id)
                if len(source_ids) == 0:
                    continue

                if len(source_ids) > sampled_number:
                    sampled_idx = rng.choice(len(source_ids), sampled_number, replace=False)
                    source_ids = source_ids[sampled_idx]
                    edge_times = edge_times[sampled_idx]

                neigh_cnt = len(source_ids)
                if neigh_cnt == 0:
                    continue
                inc = 1.0 / float(neigh_cnt)
                for source_id, source_time_raw in zip(source_ids, edge_times):
                    source_id = int(source_id)
                    if int(source_time_raw) == base.none_time_sentinel:
                        source_time = int(target_time)
                    else:
                        source_time = int(source_time_raw)
                    if source_time > max_time_in_range or source_id in layer_data[source_type]:
                        continue
                    budget[source_type][source_id][0] += inc
                    budget[source_type][source_id][1] = source_time

    for node_type, arr in inp.items():
        for node_id, node_time in arr:
            node_id = int(node_id)
            node_time = int(node_time)
            layer_data[node_type][node_id] = [len(layer_data[node_type]), node_time]
    for node_type, arr in inp.items():
        for node_id, node_time in arr:
            add_budget(node_type, int(node_id), int(node_time))

    for _ in range(sampled_depth):
        source_types = list(budget.keys())
        for source_type in source_types:
            keys = np.array(list(budget[source_type].keys()), dtype=np.int64)
            if len(keys) == 0:
                continue
            if sampled_number > len(keys):
                sampled_keys = keys
            else:
                score = np.array(list(budget[source_type].values()), dtype=np.float64)[:, 0] ** 2
                score = score / np.sum(score)
                sampled_pos = rng.choice(len(score), sampled_number, p=score, replace=False)
                sampled_keys = keys[sampled_pos]

            for k in sampled_keys:
                k = int(k)
                layer_data[source_type][k] = [len(layer_data[source_type]), budget[source_type][k][1]]
            for k in sampled_keys:
                k = int(k)
                add_budget(source_type, k, budget[source_type][k][1])
                budget[source_type].pop(k, None)

    feature = {}
    times = {}
    for node_type in base.node_types:
        node_map = layer_data.get(node_type, {})
        if len(node_map) == 0:
            feature[node_type] = torch.empty((0, base.feature_dims[node_type]), dtype=torch.float32)
            times[node_type] = np.empty((0,), dtype=np.int64)
            continue
        node_ids = np.array(list(node_map.keys()), dtype=np.int64)
        node_times = np.array([node_map[x][1] for x in node_ids], dtype=np.int64)
        feature[node_type] = base.features[node_type].index_select(0, torch.from_numpy(node_ids))
        times[node_type] = node_times

    edge_list = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for node_type, node_map in layer_data.items():
        for _, (ser, _) in node_map.items():
            edge_list[node_type][node_type]["self"].append([ser, ser])

    for target_type, source_type, relation_type in base.meta_graph:
        relation = base.relations[target_type][source_type][relation_type]
        target_nodes = layer_data.get(target_type, {})
        source_nodes = layer_data.get(source_type, {})
        if len(target_nodes) == 0 or len(source_nodes) == 0:
            continue
        for target_key, (target_ser, _) in target_nodes.items():
            source_ids, _ = relation.get_neighbors(int(target_key))
            for source_id in source_ids:
                src_info = source_nodes.get(int(source_id))
                if src_info is not None:
                    edge_list[target_type][source_type][relation_type].append(
                        [target_ser, src_info[0]]
                    )

    return feature, times, edge_list


def to_torch_shared(feature, times, edge_list, base):
    node_dict = {}
    node_feature_chunks = []
    node_type_arr = []
    node_time_arr = []

    node_num = 0
    for node_type in base.node_types:
        feat = feature[node_type]
        node_dict[node_type] = [node_num, len(node_dict)]
        node_num += int(feat.shape[0])
        node_feature_chunks.append(feat)
        node_type_arr.extend([node_dict[node_type][1]] * int(feat.shape[0]))
        node_time_arr.extend(list(times[node_type]))

    if len(node_feature_chunks) > 0:
        node_feature = torch.cat(node_feature_chunks, dim=0).float()
    else:
        node_feature = torch.empty((0, 0), dtype=torch.float32)
    node_type = torch.LongTensor(node_type_arr)

    edge_dict = {rel[2]: i for i, rel in enumerate(base.meta_graph)}
    edge_dict["self"] = len(edge_dict)

    edge_index = []
    edge_type = []
    edge_time = []
    for target_type in edge_list:
        for source_type in edge_list[target_type]:
            for relation_type in edge_list[target_type][source_type]:
                rel_edges = edge_list[target_type][source_type][relation_type]
                for ti, si in rel_edges:
                    tid = int(ti) + node_dict[target_type][0]
                    sid = int(si) + node_dict[source_type][0]
                    edge_index.append([sid, tid])
                    edge_type.append(edge_dict[relation_type])
                    edge_time.append(int(node_time_arr[tid] - node_time_arr[sid] + 120))

    if len(edge_index) == 0:
        edge_index_t = torch.empty((2, 0), dtype=torch.long)
        edge_type_t = torch.empty((0,), dtype=torch.long)
        edge_time_t = torch.empty((0,), dtype=torch.long)
    else:
        edge_index_t = torch.LongTensor(edge_index).t()
        edge_type_t = torch.LongTensor(edge_type)
        edge_time_t = torch.LongTensor(edge_time)

    return node_feature, node_type, edge_time_t, edge_index_t, edge_type_t, node_dict, edge_dict
