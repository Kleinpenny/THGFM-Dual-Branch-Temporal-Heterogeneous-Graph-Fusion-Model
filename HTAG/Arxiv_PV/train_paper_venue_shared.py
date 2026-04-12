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
_HTAG_DIR = os.path.abspath(os.path.join(_CUR_DIR, ".."))
if _HTAG_DIR not in sys.path:
    sys.path.insert(0, _HTAG_DIR)

from THGFM.model import GNN, Classifier  # noqa: E402
from THGFM.utils import randint  # noqa: E402
from THGFM.wandb_utils import add_wandb_args, init_wandb_tracker  # noqa: E402


TQDM_COMMON_KW = {"dynamic_ncols": True}
_BASE = None
_RUN_ARGS = None
_SELECTED_ROWS = {}


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
            self.features[node_type] = torch.from_numpy(np.copy(arr))

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


class SharedPVBase(SharedGraphBase):
    def __init__(self, shared_dir: str):
        super().__init__(shared_dir=shared_dir)
        task = self.meta.get("task", {})
        if task.get("name") != "paper_venue":
            raise ValueError(
                f"shared base at {shared_dir} is not paper_venue task "
                f"(got task={task.get('name')})."
            )
        self.relation_type = str(task["relation_type"])
        self.reverse_relation_type = str(task["reverse_relation_type"])
        self.candidate_venue_ids = self.load_npy(task["candidate_ids"])
        self.cand_index = {int(vid): idx for idx, vid in enumerate(self.candidate_venue_ids)}

        self.split_paper_ids = {}
        self.split_times = {}
        self.split_venue_ids = {}
        for split in ("train", "valid", "test"):
            split_files = task["splits"][split]
            self.split_paper_ids[split] = self.load_npy(split_files["paper_ids"])
            self.split_times[split] = self.load_npy(split_files["times"])
            self.split_venue_ids[split] = self.load_npy(split_files["venue_ids"])


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
                inc = 1.0 / float(neigh_cnt)
                for source_id, source_time_raw in zip(source_ids, edge_times):
                    source_id = int(source_id)
                    source_time = (
                        int(target_time)
                        if int(source_time_raw) == base.none_time_sentinel
                        else int(source_time_raw)
                    )
                    if source_time > max_time_in_range or source_id in layer_data[source_type]:
                        continue
                    budget[source_type][source_id][0] += inc
                    budget[source_type][source_id][1] = source_time

    for node_type, arr in inp.items():
        for node_id, node_time in arr:
            layer_data[node_type][int(node_id)] = [len(layer_data[node_type]), int(node_time)]
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
        feat_dim = int(base.feature_dims[node_type])
        if len(node_map) == 0:
            feature[node_type] = torch.empty((0, feat_dim), dtype=torch.float32)
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
                    edge_list[target_type][source_type][relation_type].append([target_ser, src_info[0]])
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

    node_feature = torch.cat(node_feature_chunks, dim=0).float() if len(node_feature_chunks) > 0 else torch.empty((0, 0), dtype=torch.float32)
    node_type = torch.LongTensor(node_type_arr)
    edge_dict = {rel[2]: i for i, rel in enumerate(base.meta_graph)}
    edge_dict["self"] = len(edge_dict)

    edge_index, edge_type, edge_time = [], [], []
    for target_type in edge_list:
        for source_type in edge_list[target_type]:
            for relation_type in edge_list[target_type][source_type]:
                for ti, si in edge_list[target_type][source_type][relation_type]:
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


def _set_global_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _mask_output_edges(edge_list, batch_size: int):
    if batch_size <= 0:
        return
    if "paper" in edge_list and "venue" in edge_list["paper"] and "rev_PV_Journal" in edge_list["paper"]["venue"]:
        edge_list["paper"]["venue"]["rev_PV_Journal"] = [
            e for e in edge_list["paper"]["venue"]["rev_PV_Journal"] if int(e[0]) >= batch_size
        ]
    if "venue" in edge_list and "paper" in edge_list["venue"] and "PV_Journal" in edge_list["venue"]["paper"]:
        edge_list["venue"]["paper"]["PV_Journal"] = [
            e for e in edge_list["venue"]["paper"]["PV_Journal"] if int(e[1]) >= batch_size
        ]


def _select_rows(total_rows: int, data_percentage: float):
    if total_rows <= 0:
        return np.empty((0,), dtype=np.int64)
    keep = int(total_rows * data_percentage)
    if keep <= 0:
        return np.empty((0,), dtype=np.int64)
    if keep >= total_rows:
        return np.arange(total_rows, dtype=np.int64)
    return np.random.choice(np.arange(total_rows, dtype=np.int64), keep, replace=False)


def _node_classification_sample(seed, split_name, batch_size):
    rng = np.random.default_rng(seed)
    row_ids = _SELECTED_ROWS[split_name]
    if len(row_ids) < batch_size:
        raise RuntimeError(
            f"Not enough rows for split={split_name}: have={len(row_ids)}, batch_size={batch_size}."
        )
    selected_rows = rng.choice(row_ids, batch_size, replace=False)

    paper_ids = _BASE.split_paper_ids[split_name][selected_rows].astype(np.int64, copy=False)
    paper_times = _BASE.split_times[split_name][selected_rows].astype(np.int64, copy=False)
    target_info = np.stack((paper_ids, paper_times), axis=1)

    feature, times, edge_list = sample_subgraph_shared(
        _BASE,
        split_name=split_name,
        inp={"paper": target_info},
        sampled_depth=_RUN_ARGS.sample_depth,
        sampled_number=_RUN_ARGS.sample_width,
        rng=rng,
    )
    _mask_output_edges(edge_list=edge_list, batch_size=batch_size)
    node_feature, node_type, edge_time, edge_index, edge_type, node_dict, _ = to_torch_shared(
        feature, times, edge_list, _BASE
    )

    ylabel = torch.zeros(batch_size, dtype=torch.long)
    split_venue_ids = _BASE.split_venue_ids[split_name]
    for x_id, row_id in enumerate(selected_rows):
        venue_id = int(split_venue_ids[int(row_id)])
        cls_idx = _BASE.cand_index.get(venue_id)
        if cls_idx is None:
            raise RuntimeError(f"Venue id {venue_id} from split={split_name} not found in candidate set.")
        ylabel[x_id] = int(cls_idx)
    x_ids = np.arange(batch_size, dtype=np.int64) + node_dict["paper"][0]
    return node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel


def _run_sampling_task(task):
    split_name, seed = task
    return _node_classification_sample(seed=int(seed), split_name=split_name, batch_size=_RUN_ARGS.batch_size)


def evaluate_model(gnn, classifier, device, sample_repeat=10):
    with torch.no_grad():
        test_acc = []
        for _ in range(sample_repeat):
            sample = _node_classification_sample(randint(), "test", _RUN_ARGS.batch_size)
            node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = sample
            node_rep = gnn(
                node_feature.to(device),
                node_type.to(device),
                edge_time.to(device),
                edge_index.to(device),
                edge_type.to(device),
            )
            res = classifier(node_rep[x_ids])
            pred = res.argmax(dim=1).cpu()
            acc = (pred == ylabel).float().mean().item()
            test_acc.append(float(acc))
        return float(np.average(test_acc))


def main():
    parser = argparse.ArgumentParser(description="Train HTAG ArXiv paper-venue task with shared base")
    parser.add_argument(
        "--shared_base_dir",
        type=str,
        default=os.path.join(_CUR_DIR, "data", "shared_pv_base_HTAG_arxiv"),
    )
    parser.add_argument("--model_dir", type=str, default=os.path.join(_CUR_DIR, "model_save"))
    parser.add_argument("--task_name", type=str, default="PV_HTAG_Arxiv")
    parser.add_argument("--cuda", type=int, default=1)
    parser.add_argument("--conv_name", type=str, default="rtta")
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
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--clip", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--deterministic", action="store_true")
    add_wandb_args(parser)
    args = parser.parse_args()

    _set_global_seed(args.seed, args.deterministic)

    global _BASE, _RUN_ARGS, _SELECTED_ROWS
    _RUN_ARGS = args
    _BASE = SharedPVBase(args.shared_base_dir)
    if "paper" not in _BASE.node_types:
        raise RuntimeError(
            f"Expected node type 'paper' in shared base (ArXiv). Got node_types={_BASE.node_types}."
        )
    _SELECTED_ROWS = {
        "train": _select_rows(len(_BASE.split_paper_ids["train"]), args.data_percentage),
        "valid": _select_rows(len(_BASE.split_paper_ids["valid"]), args.data_percentage),
        "test": np.arange(len(_BASE.split_paper_ids["test"]), dtype=np.int64),
    }
    for split in ("train", "valid", "test"):
        if len(_SELECTED_ROWS[split]) < args.batch_size:
            raise RuntimeError(
                f"{split} rows ({len(_SELECTED_ROWS[split])}) < batch_size ({args.batch_size})."
            )

    os.makedirs(args.model_dir, exist_ok=True)
    best_model_acc_path = os.path.join(args.model_dir, f"{args.task_name}_highest_valid_acc.pt")
    best_model_loss_path = os.path.join(args.model_dir, f"{args.task_name}_lowest_valid_loss.pt")
    metrics_path = os.path.join(args.model_dir, f"{args.task_name}.jsonl")
    best_valid_acc = float("-inf")
    best_valid_loss = float("inf")
    valid_acc_when_lowest_valid_loss_saved = None
    epoch_when_lowest_valid_loss_saved = 0

    pool = None
    if args.n_pool > 1:
        # Avoid "received 0 items of ancdata" when pickling tensor storages through
        # multiprocessing pipes (common under heavy load / systemd-run).
        try:
            torch.multiprocessing.set_sharing_strategy("file_system")
        except RuntimeError:
            pass
        # Fork worker pool before any CUDA context/model creation to avoid
        # fork-after-CUDA deadlocks during sampling collection.
        pool = mp.get_context("fork").Pool(args.n_pool)

    device = torch.device(f"cuda:{args.cuda}") if args.cuda >= 0 and torch.cuda.is_available() else torch.device("cpu")
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
    classifier = Classifier(args.n_hid, len(_BASE.candidate_venue_ids)).to(device)
    model = nn.Sequential(gnn, classifier)
    criterion = nn.NLLLoss()

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters())
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters())
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    else:
        optimizer = torch.optim.Adagrad(model.parameters())

    def append_metrics(event, **kwargs):
        payload = {"event": event, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        payload.update(kwargs)
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    wandb_tracker = init_wandb_tracker(
        args,
        extra_config={"dataset": "arxiv", "task": "paper_venue"},
        default_group=None,
    )
    try:
        for epoch in tqdm(np.arange(args.n_epoch) + 1, desc="Epochs", **TQDM_COMMON_KW):
            t0 = time.time()
            if pool is not None:
                tasks = [("train", randint()) for _ in range(args.n_batch)]
                train_data = list(
                    tqdm(
                        pool.imap_unordered(_run_sampling_task, tasks),
                        total=len(tasks),
                        desc=f"E{epoch} collect",
                        leave=False,
                        **TQDM_COMMON_KW,
                    )
                )
                valid_data = pool.apply(_run_sampling_task, args=(("valid", randint()),))
            else:
                train_data = [_run_sampling_task(("train", randint())) for _ in range(args.n_batch)]
                valid_data = _run_sampling_task(("valid", randint()))

            model.train()
            train_losses = []
            torch.cuda.empty_cache()
            for _ in range(args.repeat):
                for node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel in train_data:
                    node_rep = gnn(
                        node_feature.to(device),
                        node_type.to(device),
                        edge_time.to(device),
                        edge_index.to(device),
                        edge_type.to(device),
                    )
                    res = classifier(node_rep[x_ids])
                    loss = criterion(res, ylabel.to(device))
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                    optimizer.step()
                    train_losses.append(float(loss.detach().cpu().item()))

            model.eval()
            with torch.no_grad():
                node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = valid_data
                node_rep = gnn(
                    node_feature.to(device),
                    node_type.to(device),
                    edge_time.to(device),
                    edge_index.to(device),
                    edge_type.to(device),
                )
                res = classifier(node_rep[x_ids])
                valid_loss = float(criterion(res, ylabel.to(device)).detach().cpu().item())
                pred = res.argmax(dim=1).cpu()
                valid_acc = float((pred == ylabel).float().mean().item())

            epoch_sec = float(time.time() - t0)
            msg = (
                f"Epoch {int(epoch)} | train_loss={float(np.average(train_losses)):.4f} "
                f"valid_loss={valid_loss:.4f} valid_acc={valid_acc:.4f} time={epoch_sec:.1f}s"
            )
            print(msg, flush=True)
            is_best_acc = valid_acc > best_valid_acc
            is_best_loss = valid_loss < best_valid_loss
            if is_best_acc:
                best_valid_acc = valid_acc
                torch.save({"gnn": gnn.state_dict(), "classifier": classifier.state_dict()}, best_model_acc_path)
            if is_best_loss:
                best_valid_loss = valid_loss
                valid_acc_when_lowest_valid_loss_saved = valid_acc
                epoch_when_lowest_valid_loss_saved = int(epoch)
                torch.save({"gnn": gnn.state_dict(), "classifier": classifier.state_dict()}, best_model_loss_path)
            append_metrics(
                "epoch",
                epoch=int(epoch),
                train_loss=float(np.average(train_losses)),
                valid_loss=valid_loss,
                valid_acc=valid_acc,
                best_valid_acc=float(best_valid_acc),
                best_valid_loss=float(best_valid_loss),
                best_acc_updated=bool(is_best_acc),
                best_loss_updated=bool(is_best_loss),
                epoch_time_sec=epoch_sec,
                epoch_per_time=epoch_sec,
                valid_acc_when_lowest_valid_loss_saved=valid_acc_when_lowest_valid_loss_saved,
                epoch_when_lowest_valid_loss_saved=int(epoch_when_lowest_valid_loss_saved),
            )
            wb_epoch = {
                "epoch": int(epoch),
                "epoch_per_time": epoch_sec,
                "train_loss": float(np.average(train_losses)),
                "valid_loss": valid_loss,
                "valid_acc": valid_acc,
                "best_valid_acc": float(best_valid_acc),
                "best_valid_loss": float(best_valid_loss),
                "epoch_time_sec": epoch_sec,
            }
            if valid_acc_when_lowest_valid_loss_saved is not None:
                wb_epoch["valid_acc_when_lowest_valid_loss_saved"] = float(valid_acc_when_lowest_valid_loss_saved)
                wb_epoch["epoch_when_lowest_valid_loss_saved"] = int(epoch_when_lowest_valid_loss_saved)
            wandb_tracker.log(wb_epoch, step=int(epoch))

        if valid_acc_when_lowest_valid_loss_saved is None:
            raise RuntimeError("valid_acc_when_lowest_valid_loss_saved is unset; no valid-loss improvement recorded.")
        print(
            f"Lowest valid loss: {best_valid_loss:.5f} at epoch {epoch_when_lowest_valid_loss_saved} "
            f"(valid_acc_at_that_epoch={float(valid_acc_when_lowest_valid_loss_saved):.4f})",
            flush=True,
        )
        append_metrics(
            "valid_loss_best_epoch",
            epoch=int(epoch_when_lowest_valid_loss_saved),
            best_valid_loss=float(best_valid_loss),
            valid_acc_at_best_valid_loss=float(valid_acc_when_lowest_valid_loss_saved),
        )

        if not os.path.exists(best_model_acc_path):
            raise RuntimeError(f"Best-acc checkpoint not found: {best_model_acc_path}")
        if not os.path.exists(best_model_loss_path):
            raise RuntimeError(f"Best-loss checkpoint not found: {best_model_loss_path}")

        ckpt_acc = torch.load(best_model_acc_path, map_location=device)
        gnn.load_state_dict(ckpt_acc["gnn"])
        classifier.load_state_dict(ckpt_acc["classifier"])
        test_acc_from_best_valid_acc = evaluate_model(gnn, classifier, device, sample_repeat=10)
        print(f"Best Test (highest Valid Acc): {test_acc_from_best_valid_acc:.4f}", flush=True)
        append_metrics(
            "test_best_valid_acc",
            test_acc=test_acc_from_best_valid_acc,
            checkpoint=best_model_acc_path,
            test_acc_highest_valid_acc_checkpoint=test_acc_from_best_valid_acc,
        )
        wandb_tracker.log(
            {
                "final/test_acc_highest_valid_acc_ckpt": test_acc_from_best_valid_acc,
                "test_acc_best_valid_acc": test_acc_from_best_valid_acc,
            },
            step=int(args.n_epoch),
        )

        ckpt_loss = torch.load(best_model_loss_path, map_location=device)
        gnn.load_state_dict(ckpt_loss["gnn"])
        classifier.load_state_dict(ckpt_loss["classifier"])
        test_acc_from_best_valid_loss = evaluate_model(gnn, classifier, device, sample_repeat=10)
        print(f"Best Test (lowest Valid Loss): {test_acc_from_best_valid_loss:.4f}", flush=True)
        append_metrics(
            "test_best_valid_loss",
            test_acc=test_acc_from_best_valid_loss,
            checkpoint=best_model_loss_path,
            test_acc_lowest_valid_loss_checkpoint=test_acc_from_best_valid_loss,
            valid_acc_when_lowest_valid_loss_saved=valid_acc_when_lowest_valid_loss_saved,
            epoch_when_lowest_valid_loss_saved=int(epoch_when_lowest_valid_loss_saved),
        )
        wandb_tracker.log(
            {
                "final/test_acc_lowest_valid_loss_ckpt": test_acc_from_best_valid_loss,
                "test_acc_best_valid_loss": test_acc_from_best_valid_loss,
                "final/valid_acc_at_best_valid_loss": float(valid_acc_when_lowest_valid_loss_saved),
                "final/epoch_at_best_valid_loss": int(epoch_when_lowest_valid_loss_saved),
                "final/best_valid_loss": float(best_valid_loss),
            },
            step=int(args.n_epoch),
        )
        wandb_tracker.summary(
            {
                "test_acc/highest_valid_acc_checkpoint": test_acc_from_best_valid_acc,
                "test_acc/lowest_valid_loss_checkpoint": test_acc_from_best_valid_loss,
                "test_acc_best_valid_acc": test_acc_from_best_valid_acc,
                "test_acc_best_valid_loss": test_acc_from_best_valid_loss,
                "valid_acc_when_lowest_valid_loss_saved": valid_acc_when_lowest_valid_loss_saved,
                "valid_acc_at_best_valid_loss": float(valid_acc_when_lowest_valid_loss_saved),
                "epoch_when_lowest_valid_loss_saved": int(epoch_when_lowest_valid_loss_saved),
                "epoch_at_best_valid_loss": int(epoch_when_lowest_valid_loss_saved),
                "best_valid_loss": float(best_valid_loss),
            }
        )

        print(f"[OK] model saved: {best_model_acc_path}", flush=True)
        print(f"[OK] model saved: {best_model_loss_path}", flush=True)
        print(f"[OK] metrics saved: {metrics_path}", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        wandb_tracker.finish()


if __name__ == "__main__":
    main()
