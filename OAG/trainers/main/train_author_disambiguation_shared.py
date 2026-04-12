import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm


_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_OAG_DIR = os.path.abspath(os.path.join(_CUR_DIR, "..", ".."))
if _OAG_DIR not in sys.path:
    # Ensure imports resolve with trainers/main directory layout.
    sys.path.insert(0, _OAG_DIR)

from tool_functions.wandb_utils import add_wandb_args, init_wandb_tracker  # noqa: E402
from THGFM.model import GNN, Matcher  # noqa: E402
from THGFM.utils import ndcg_at_k, mean_reciprocal_rank, randint  # noqa: E402


TQDM_COMMON_KW = {"dynamic_ncols": True}

_BASE = None
_SELECTED_GROUPS = {}
_RUN_ARGS = None


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


class SharedADBase:
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

        grp_dir = os.path.join(shared_dir, "groups")
        self.group_author_offsets = np.load(
            os.path.join(grp_dir, "group_authors__offsets.npy"), mmap_mode="r"
        )
        self.group_author_ids = np.load(
            os.path.join(grp_dir, "group_authors__ids.npy"), mmap_mode="r"
        )

        self.split_offsets = {}
        self.split_paper_ids = {}
        self.split_author_pos = {}
        self.split_times = {}
        self.split_group_ids = {}
        for split in ("train", "valid", "test"):
            self.split_offsets[split] = np.load(
                os.path.join(grp_dir, f"{split}__offsets.npy"),
                mmap_mode="r",
            )
            self.split_paper_ids[split] = np.load(
                os.path.join(grp_dir, f"{split}__paper_ids.npy"),
                mmap_mode="r",
            )
            self.split_author_pos[split] = np.load(
                os.path.join(grp_dir, f"{split}__author_pos.npy"),
                mmap_mode="r",
            )
            self.split_times[split] = np.load(
                os.path.join(grp_dir, f"{split}__times.npy"),
                mmap_mode="r",
            )
            self.split_group_ids[split] = np.load(
                os.path.join(grp_dir, f"{split}__group_ids.npy"),
                mmap_mode="r",
            )

    def get_group_authors(self, group_id: int) -> np.ndarray:
        st = int(self.group_author_offsets[group_id])
        ed = int(self.group_author_offsets[group_id + 1])
        return self.group_author_ids[st:ed]

    def get_group_triplets(self, split: str, group_id: int):
        st = int(self.split_offsets[split][group_id])
        ed = int(self.split_offsets[split][group_id + 1])
        return (
            self.split_paper_ids[split][st:ed],
            self.split_author_pos[split][st:ed],
            self.split_times[split][st:ed],
        )


def _set_global_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def mask_softmax(pred, size):
    loss = 0
    stx = 0
    for l in size:
        loss += torch.log_softmax(pred[stx: stx + l], dim=-1)[0] / np.log(l)
        stx += l
    return -loss


def _mask_output_edges(edge_list, batch_size: int):
    """
    Mirror legacy AD behavior: mask direct first-author edges for output papers.
    """
    if batch_size <= 0:
        return

    if (
        "paper" in edge_list
        and "author" in edge_list["paper"]
        and "AP_write_first" in edge_list["paper"]["author"]
    ):
        edge_list["paper"]["author"]["AP_write_first"] = [
            e
            for e in edge_list["paper"]["author"]["AP_write_first"]
            if int(e[0]) >= batch_size
        ]

    if (
        "author" in edge_list
        and "paper" in edge_list["author"]
        and "rev_AP_write_first" in edge_list["author"]["paper"]
    ):
        edge_list["author"]["paper"]["rev_AP_write_first"] = [
            e
            for e in edge_list["author"]["paper"]["rev_AP_write_first"]
            if int(e[1]) >= batch_size
        ]


def _sample_subgraph_shared(base, split_name: str, inp, sampled_depth, sampled_number, rng):
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


def _to_torch_shared(feature, times, edge_list, base):
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


def _author_disambiguation_sample(seed, split_name, batch_size, batch_id=-1, epoch=0):
    rng = np.random.default_rng(seed)
    group_ids = _SELECTED_GROUPS[split_name]
    if len(group_ids) == 0:
        raise RuntimeError(f"No available groups for split={split_name}.")
    need_names = max(1, batch_size // 4)
    sample_n = min(need_names, len(group_ids))
    selected_groups = rng.choice(group_ids, sample_n, replace=False)

    author_dict = {}
    author_info = []
    paper_info = []
    name_label = []
    max_time = int(_BASE.split_max_time[split_name])

    for gid in selected_groups:
        gid = int(gid)
        author_list = _BASE.get_group_authors(gid)
        for author_id in author_list:
            author_id = int(author_id)
            if author_id not in author_dict:
                author_dict[author_id] = len(author_dict)
                author_info.append([author_id, max_time])

        paper_ids, author_pos, paper_times = _BASE.get_group_triplets(split_name, gid)
        for p_id, a_pos, p_time in zip(paper_ids, author_pos, paper_times):
            p_id = int(p_id)
            a_pos = int(a_pos)
            p_time = int(p_time)
            paper_info.append([p_id, p_time])
            labels = [author_dict[int(author_list[a_pos])]]
            for x_id, author_id in enumerate(author_list):
                if x_id != a_pos:
                    labels.append(author_dict[int(author_id)])
            name_label.append(labels)

    if len(paper_info) == 0 or len(author_info) == 0:
        raise RuntimeError(
            f"Sampled empty batch for split={split_name}, epoch={epoch}, batch={batch_id}."
        )

    feature, times, edge_list = _sample_subgraph_shared(
        _BASE,
        split_name=split_name,
        inp={
            "paper": np.asarray(paper_info, dtype=np.int64),
            "author": np.asarray(author_info, dtype=np.int64),
        },
        sampled_depth=_RUN_ARGS.sample_depth,
        sampled_number=_RUN_ARGS.sample_width,
        rng=rng,
    )
    _mask_output_edges(edge_list=edge_list, batch_size=batch_size)
    node_feature, node_type, edge_time, edge_index, edge_type, node_dict, _ = _to_torch_shared(
        feature, times, edge_list, _BASE
    )

    ylabel = {}
    for x_id, author_ids in enumerate(name_label):
        ylabel[x_id + node_dict["paper"][0]] = np.array(author_ids, dtype=np.int64) + node_dict["author"][0]
    return node_feature, node_type, edge_time, edge_index, edge_type, ylabel


def _run_sampling_task(task):
    split_name, seed, batch_id, epoch = task
    return _author_disambiguation_sample(
        seed=int(seed),
        split_name=split_name,
        batch_size=_RUN_ARGS.batch_size,
        batch_id=int(batch_id),
        epoch=int(epoch),
    )


def _prepare_data(epoch):
    train_tasks = []
    for batch_id in np.arange(_RUN_ARGS.n_batch):
        train_tasks.append(("train", randint(), int(batch_id), int(epoch)))
    valid_task = ("valid", randint(), 0, int(epoch))
    return train_tasks, valid_task


def _prepare_data_sync(epoch):
    train_data = []
    for batch_id in np.arange(_RUN_ARGS.n_batch):
        train_data.append(_run_sampling_task(("train", randint(), int(batch_id), int(epoch))))
    valid_data = _run_sampling_task(("valid", randint(), 0, int(epoch)))
    return train_data, valid_data


def _select_groups(group_ids, data_percentage):
    group_ids = np.asarray(group_ids, dtype=np.int64)
    if len(group_ids) == 0:
        return group_ids
    keep = int(len(group_ids) * data_percentage)
    if keep <= 0:
        return np.empty((0,), dtype=np.int64)
    if keep >= len(group_ids):
        return group_ids
    return np.random.choice(group_ids, keep, replace=False)


def main():
    parser = argparse.ArgumentParser(description="Training GNN on AD task (shared read-only base)")

    project_root = _OAG_DIR
    parser.add_argument(
        "--shared_base_dir",
        type=str,
        default=os.path.join(project_root, "data", "shared_ad_base"),
        help="Directory produced by data_preprocess/build_shared_ad_base.py",
    )
    parser.add_argument("--model_dir", type=str, default="./model_save")
    parser.add_argument("--task_name", type=str, default="AD")
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument(
        "--conv_name",
        type=str,
        default="rtta",
        choices=[
            "rtta", "rtta_rte0", "rtta_rota",
            "ssta", "ssta_rte0", "ssta_rota",
            "drsf", "drsf_rota",
            "drsf_sf", "drsf_sf_rota",
            "drsf_cglf", "drsf_cglf_rota",
            "drsf_cgsf", "drsf_cgsf_rota",
            "drsf_nglf", "drsf_nglf_rota",
            "drsf_ngsf", "drsf_ngsf_rota",
            "drsf_tcnglf", "drsf_tc_nglf", "drsftc_nglf",
            "drsf_tcnglf_rota", "drsf_tc_nglf_rota", "drsftc_nglf_rota",
            "drsf_tcngsf", "drsf_tc_ngsf", "drsftc_ngsf",
            "drsf_tcngsf_rota", "drsf_tc_ngsf_rota", "drsftc_ngsf_rota",
            "drsf_tanglf", "drsf_tanglf_rota",
            "drsf_tangsf", "drsf_tangsf_rota",
            "drsf_gate", "drsf_gate_rota",
            "drsftc_ndg", "drsftc_ndg_rota",
            "drsftc_glf", "drsf_tcglf", "drsf_tc_glf",
            "drsftc_glf_rota", "drsf_tcglf_rota", "drsf_tc_glf_rota",
            "gcn", "gat",
        ],
    )
    parser.add_argument("--rota_mode", type=str, default="qk", choices=["none", "k", "qk"])
    parser.add_argument("--rota_keep_rte", type=int, default=1, choices=[0, 1])
    parser.add_argument("--rtta_rota_mode", type=str, default="", choices=["", "none", "k", "qk"])
    parser.add_argument("--ssta_rota_mode", type=str, default="", choices=["", "none", "k", "qk"])
    parser.add_argument("--rtta_rota_keep_rte", type=int, default=-1, choices=[-1, 0, 1])
    parser.add_argument("--ssta_rota_keep_rte", type=int, default=-1, choices=[-1, 0, 1])
    parser.add_argument("--n_hid", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--sample_depth", type=int, default=6)
    parser.add_argument("--sample_width", type=int, default=128)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam", "sgd", "adagrad"])
    parser.add_argument("--data_percentage", type=float, default=1.0)
    parser.add_argument("--n_epoch", type=int, default=200)
    parser.add_argument("--n_pool", type=int, default=4)
    parser.add_argument("--n_batch", type=int, default=32)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--clip", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--deterministic", action="store_true")
    add_wandb_args(parser, default_project="THGFM")
    args = parser.parse_args()

    _set_global_seed(args.seed, args.deterministic)

    global _BASE, _SELECTED_GROUPS, _RUN_ARGS
    _RUN_ARGS = args
    _BASE = SharedADBase(args.shared_base_dir)
    _SELECTED_GROUPS = {
        "train": _select_groups(_BASE.split_group_ids["train"], args.data_percentage),
        "valid": _select_groups(_BASE.split_group_ids["valid"], args.data_percentage),
        "test": np.asarray(_BASE.split_group_ids["test"], dtype=np.int64),
    }

    if len(_SELECTED_GROUPS["train"]) == 0:
        raise RuntimeError("No train groups selected. Increase --data_percentage.")
    if len(_SELECTED_GROUPS["valid"]) == 0:
        raise RuntimeError("No valid groups selected. Increase --data_percentage.")
    if len(_SELECTED_GROUPS["test"]) == 0:
        raise RuntimeError("No test groups available in shared base.")

    run_model_dir = os.path.join(args.model_dir, args.conv_name)
    os.makedirs(run_model_dir, exist_ok=True)
    best_model_path = os.path.join(run_model_dir, args.task_name)
    best_loss_model_path = os.path.join(run_model_dir, f"{args.task_name}_lowest_valid_loss")
    metrics_dir = os.path.join(run_model_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, f"{args.task_name}.jsonl")

    def wandb_metric_key(label: str) -> str:
        return (
            label.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
            .replace("-", "_")
        )

    def append_metrics(event, **kwargs):
        payload = {
            "event": event,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "task_name": args.task_name,
            "conv_name": args.conv_name,
            "seed": args.seed,
            "deterministic": args.deterministic,
        }
        payload.update(kwargs)
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    wandb_tracker = init_wandb_tracker(
        args,
        extra_config={
            "task_type": "author_disambiguation_shared",
            "metrics_path": metrics_path,
            "shared_base_dir": args.shared_base_dir,
        },
        default_group="author_disambiguation_shared",
    )
    append_metrics("run_start")

    pool = None
    if args.n_pool > 1:
        # IMPORTANT: mirror original AD training behavior.
        # Create pool before CUDA model initialization to avoid fork-after-CUDA deadlocks.
        # Also keep fork context so workers inherit read-only base without pickling large objects.
        pool = mp.get_context("fork").Pool(args.n_pool)

    device = torch.device(f"cuda:{args.cuda}") if args.cuda != -1 else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    in_dim = int(_BASE.feature_dims["paper"])
    gnn = GNN(
        conv_name=args.conv_name,
        in_dim=in_dim,
        n_hid=args.n_hid,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        num_types=len(_BASE.node_types),
        num_relations=len(_BASE.meta_graph) + 1,
        rota_mode=args.rota_mode,
        rota_keep_rte=bool(args.rota_keep_rte),
        rtta_rota_mode=(args.rtta_rota_mode if args.rtta_rota_mode else None),
        ssta_rota_mode=(args.ssta_rota_mode if args.ssta_rota_mode else None),
        rtta_rota_keep_rte=(None if args.rtta_rota_keep_rte == -1 else bool(args.rtta_rota_keep_rte)),
        ssta_rota_keep_rte=(None if args.ssta_rota_keep_rte == -1 else bool(args.ssta_rota_keep_rte)),
    ).to(device)
    matcher = Matcher(args.n_hid).to(device)
    model = nn.Sequential(gnn, matcher)

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters())
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters())
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    else:
        optimizer = torch.optim.Adagrad(model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000, eta_min=1e-6)

    best_val = float("-inf")
    best_loss = float("inf")
    st = time.time()

    def evaluate_model(curr_gnn, curr_matcher, split_name):
        with torch.no_grad():
            test_res = []
            for _ in range(10):
                sample = _author_disambiguation_sample(randint(), "test", args.batch_size)
                node_feature, node_type, edge_time, edge_index, edge_type, ylabel = sample
                node_rep = curr_gnn(
                    node_feature.to(device),
                    node_type.to(device),
                    edge_time.to(device),
                    edge_index.to(device),
                    edge_type.to(device),
                )

                author_key = []
                paper_key = []
                key_size = []
                for paper_id in ylabel:
                    author_ids = ylabel[paper_id]
                    paper_key.append(np.repeat(paper_id, len(author_ids)))
                    author_key.append(author_ids)
                    key_size.append(len(author_ids))
                paper_key = torch.LongTensor(np.concatenate(paper_key)).to(device)
                author_key = torch.LongTensor(np.concatenate(author_key)).to(device)

                test_paper_vecs = node_rep[paper_key]
                test_author_vecs = node_rep[author_key]
                res = curr_matcher(test_author_vecs, test_paper_vecs, pair=True)

                ser = 0
                for s in key_size:
                    p = res[ser: ser + s]
                    l = torch.zeros(s)
                    l[0] = 1
                    sorted_idx = p.argsort(descending=True).cpu()
                    r = l[sorted_idx]
                    test_res.append(r.cpu().detach().tolist())
                    ser += s

            test_ndcg = [ndcg_at_k(resi, len(resi)) for resi in test_res]
            avg_ndcg = float(np.average(test_ndcg))
            test_mrr = mean_reciprocal_rank(test_res)
            avg_mrr = float(np.average(test_mrr))
            print(f"{split_name} NDCG: {avg_ndcg:.4f}")
            print(f"{split_name} MRR:  {avg_mrr:.4f}")
            append_metrics(
                split_name.lower().replace(" ", "_"),
                ndcg=avg_ndcg,
                mrr=avg_mrr,
            )
            metric_key = wandb_metric_key(split_name)
            wandb_tracker.log(
                {
                    f"{metric_key}/ndcg": avg_ndcg,
                    f"{metric_key}/mrr": avg_mrr,
                }
            )
            wandb_tracker.summary(
                {
                    f"{metric_key}_ndcg": avg_ndcg,
                    f"{metric_key}_mrr": avg_mrr,
                }
            )

    run_exit_code = 0
    try:
        try:
            for epoch in tqdm(np.arange(args.n_epoch) + 1, desc="Epochs", **TQDM_COMMON_KW):
                if args.n_pool > 1:
                    train_tasks, valid_task = _prepare_data(epoch)
                    collect_fn = pool.imap if args.deterministic else pool.imap_unordered
                    train_data = list(
                        tqdm(
                            collect_fn(_run_sampling_task, train_tasks),
                            total=len(train_tasks),
                            desc=f"E{epoch} collect train",
                            leave=False,
                            **TQDM_COMMON_KW,
                        )
                    )
                    valid_data = pool.apply(_run_sampling_task, args=(valid_task,))
                else:
                    train_data, valid_data = _prepare_data_sync(epoch)

                et = time.time()
                print("Data Preparation: %.1fs" % (et - st))

                model.train()
                train_losses = []
                torch.cuda.empty_cache()
                for repeat_idx in tqdm(range(args.repeat), desc=f"E{epoch} repeats", leave=False, **TQDM_COMMON_KW):
                    for node_feature, node_type, edge_time, edge_index, edge_type, ylabel in tqdm(
                        train_data,
                        desc=f"E{epoch} R{repeat_idx + 1} train",
                        leave=False,
                        **TQDM_COMMON_KW,
                    ):
                        node_rep = gnn(
                            node_feature.to(device),
                            node_type.to(device),
                            edge_time.to(device),
                            edge_index.to(device),
                            edge_type.to(device),
                        )
                        author_key = []
                        paper_key = []
                        key_size = []
                        for paper_id in ylabel:
                            author_ids = ylabel[paper_id]
                            paper_key.append(np.repeat(paper_id, len(author_ids)))
                            author_key.append(author_ids)
                            key_size.append(len(author_ids))
                        paper_key = torch.LongTensor(np.concatenate(paper_key)).to(device)
                        author_key = torch.LongTensor(np.concatenate(author_key)).to(device)

                        train_paper_vecs = node_rep[paper_key]
                        train_author_vecs = node_rep[author_key]
                        res = matcher(train_author_vecs, train_paper_vecs, pair=True)
                        loss = mask_softmax(res, key_size)

                        optimizer.zero_grad()
                        torch.cuda.empty_cache()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                        optimizer.step()

                        train_losses.append(loss.cpu().detach().item())
                        scheduler.step()
                        del res, loss

                model.eval()
                with torch.no_grad():
                    node_feature, node_type, edge_time, edge_index, edge_type, ylabel = valid_data
                    node_rep = gnn(
                        node_feature.to(device),
                        node_type.to(device),
                        edge_time.to(device),
                        edge_index.to(device),
                        edge_type.to(device),
                    )
                    author_key = []
                    paper_key = []
                    key_size = []
                    for paper_id in ylabel:
                        author_ids = ylabel[paper_id]
                        paper_key.append(np.repeat(paper_id, len(author_ids)))
                        author_key.append(author_ids)
                        key_size.append(len(author_ids))
                    paper_key = torch.LongTensor(np.concatenate(paper_key)).to(device)
                    author_key = torch.LongTensor(np.concatenate(author_key)).to(device)

                    valid_paper_vecs = node_rep[paper_key]
                    valid_author_vecs = node_rep[author_key]
                    res = matcher(valid_author_vecs, valid_paper_vecs, pair=True)
                    loss = mask_softmax(res, key_size)

                    valid_res = []
                    ser = 0
                    for s in key_size:
                        p = res[ser: ser + s]
                        l = torch.zeros(s)
                        l[0] = 1
                        sorted_idx = p.argsort(descending=True).cpu()
                        r = l[sorted_idx]
                        valid_res.append(r.cpu().detach().tolist())
                        ser += s
                    valid_ndcg = np.average([ndcg_at_k(resi, len(resi)) for resi in valid_res])
                    valid_mrr = np.average(mean_reciprocal_rank(valid_res))
                    curr_valid_loss = float(loss.cpu().detach().item())
                    is_best = valid_ndcg > best_val
                    is_best_loss = curr_valid_loss < best_loss
                    if is_best:
                        best_val = valid_ndcg
                        torch.save(model, best_model_path)
                        print("UPDATE!!! Best model by highest Valid NDCG")
                    if is_best_loss:
                        best_loss = curr_valid_loss
                        torch.save(model, best_loss_model_path)
                        print("UPDATE!!! Best model by lowest Valid Loss")

                    st = time.time()
                    epoch_time = float(st - et)
                    print(
                        (
                            "Epoch: %d (%.1fs)  LR: %.5f Train Loss: %.2f  "
                            "Valid Loss: %.2f  Valid NDCG: %.4f  Valid MRR: %.4f"
                        )
                        % (
                            epoch,
                            epoch_time,
                            optimizer.param_groups[0]["lr"],
                            float(np.average(train_losses)),
                            curr_valid_loss,
                            float(valid_ndcg),
                            float(valid_mrr),
                        )
                    )
                    metrics_payload = {
                        "epoch": int(epoch),
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "train_loss": float(np.average(train_losses)),
                        "valid_loss": curr_valid_loss,
                        "valid_ndcg": float(valid_ndcg),
                        "valid_mrr": float(valid_mrr),
                        "best_valid_ndcg_updated": bool(is_best),
                        "best_valid_loss_updated": bool(is_best_loss),
                        "time_per_epoch": epoch_time,
                    }
                    append_metrics(
                        "epoch_validation",
                        **metrics_payload,
                    )
                    wandb_tracker.log(metrics_payload, step=int(epoch))
                    if is_best:
                        wandb_tracker.summary({"best_valid_ndcg": float(best_val)})
                    if is_best_loss:
                        wandb_tracker.summary({"best_valid_loss": float(best_loss)})
                    del res, loss
                del train_data, valid_data
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        evaluate_model(gnn, matcher, "Last Test")

        best_model = torch.load(best_model_path, map_location=device, weights_only=False)
        best_model.eval()
        gnn_best, matcher_best = best_model
        evaluate_model(gnn_best, matcher_best, "Best Test (highest Valid NDCG)")

        best_loss_model = torch.load(best_loss_model_path, map_location=device, weights_only=False)
        best_loss_model.eval()
        gnn_loss, matcher_loss = best_loss_model
        evaluate_model(gnn_loss, matcher_loss, "Best Test (lowest Valid Loss)")
    except Exception:
        run_exit_code = 1
        raise
    finally:
        wandb_tracker.finish(exit_code=run_exit_code)


if __name__ == "__main__":
    main()
