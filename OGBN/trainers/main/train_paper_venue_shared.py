import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from ogb.nodeproppred import Evaluator


_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_OGBN_DIR = os.path.abspath(os.path.join(_CUR_DIR, "..", ".."))
if _OGBN_DIR not in sys.path:
    sys.path.insert(0, _OGBN_DIR)

from tool_functions.wandb_utils import add_wandb_args, init_wandb_tracker  # noqa: E402
from THGFM.model import GNN, Classifier  # noqa: E402
from THGFM.utils import randint  # noqa: E402
from tool_functions.shared_base_runtime import (  # noqa: E402
    SharedGraphBase,
    sample_subgraph_shared,
    to_torch_shared,
)


TQDM_COMMON_KW = {"dynamic_ncols": True}

_BASE = None
_RUN_ARGS = None
_SELECTED_ROWS = {}


class _TeeStream:
    """Duplicate writes to multiple file-like objects (e.g. console + log file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self._streams:
            s.flush()

    def isatty(self):
        return getattr(self._streams[0], "isatty", lambda: False)()


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

    if (
        "paper" in edge_list
        and "venue" in edge_list["paper"]
        and "rev_PV_Journal" in edge_list["paper"]["venue"]
    ):
        edge_list["paper"]["venue"]["rev_PV_Journal"] = [
            e for e in edge_list["paper"]["venue"]["rev_PV_Journal"] if int(e[0]) >= batch_size
        ]

    if (
        "venue" in edge_list
        and "paper" in edge_list["venue"]
        and "PV_Journal" in edge_list["venue"]["paper"]
    ):
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


def _node_classification_sample(seed, split_name, batch_size, batch_id=-1, epoch=0):
    rng = np.random.default_rng(seed)
    row_ids = _SELECTED_ROWS[split_name]
    if len(row_ids) == 0:
        raise RuntimeError(f"No rows available for split={split_name}.")
    if len(row_ids) < batch_size:
        raise RuntimeError(
            f"Not enough rows for split={split_name}: have={len(row_ids)}, "
            f"batch_size={batch_size}. Reduce --batch_size or increase --data_percentage."
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
            raise RuntimeError(
                f"Venue id {venue_id} from split={split_name} not found in candidate set."
            )
        ylabel[x_id] = int(cls_idx)
    x_ids = np.arange(batch_size, dtype=np.int64) + node_dict["paper"][0]
    return node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel


def _run_sampling_task(task):
    split_name, seed, batch_id, epoch = task
    return _node_classification_sample(
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


def main():
    parser = argparse.ArgumentParser(
        description="Train GNN on ogbn-mag paper→venue (PV) from shared base."
    )

    project_root = _OGBN_DIR
    parser.add_argument(
        "--shared_base_dir",
        type=str,
        default=os.path.join(project_root, "data", "shared_pv_base_ogbn_mag_graph"),
        help=(
            "Shared PV base directory (default: w2v graph export from "
            "data_preprocess/build_shared_pv_base_ogbn_mag.py). "
            "CSV-only base: data_preprocess/build_shared_ogbn_mag_pv_base.py → shared_pv_base_ogbn_mag."
        ),
    )
    parser.add_argument("--model_dir", type=str, default="./model_save")
    parser.add_argument("--task_name", type=str, default="PV_ogbn_mag")
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
    add_wandb_args(parser, default_project="THGFM-OGBN")
    args = parser.parse_args()
    _log_file = None

    _set_global_seed(args.seed, args.deterministic)

    global _BASE, _RUN_ARGS, _SELECTED_ROWS
    _RUN_ARGS = args
    _BASE = SharedPVBase(args.shared_base_dir)
    _SELECTED_ROWS = {
        "train": _select_rows(len(_BASE.split_paper_ids["train"]), args.data_percentage),
        "valid": _select_rows(len(_BASE.split_paper_ids["valid"]), args.data_percentage),
        "test": np.arange(len(_BASE.split_paper_ids["test"]), dtype=np.int64),
    }

    if len(_SELECTED_ROWS["train"]) == 0:
        raise RuntimeError("No train rows selected. Increase --data_percentage.")
    if len(_SELECTED_ROWS["valid"]) == 0:
        raise RuntimeError("No valid rows selected. Increase --data_percentage.")
    if len(_SELECTED_ROWS["test"]) == 0:
        raise RuntimeError("No test rows available in shared base.")
    if len(_SELECTED_ROWS["train"]) < args.batch_size:
        raise RuntimeError(
            f"Train rows ({len(_SELECTED_ROWS['train'])}) < batch_size ({args.batch_size})."
        )
    if len(_SELECTED_ROWS["valid"]) < args.batch_size:
        raise RuntimeError(
            f"Valid rows ({len(_SELECTED_ROWS['valid'])}) < batch_size ({args.batch_size})."
        )
    if len(_SELECTED_ROWS["test"]) < args.batch_size:
        raise RuntimeError(
            f"Test rows ({len(_SELECTED_ROWS['test'])}) < batch_size ({args.batch_size})."
        )

    # Distinct run folder + log filename: include key hparams for reproducibility
    run_hparam_tag = f"nhid{args.n_hid}_nlayers{args.n_layers}_nepoch{args.n_epoch}"
    run_model_dir = os.path.join(args.model_dir, args.conv_name, run_hparam_tag)
    os.makedirs(run_model_dir, exist_ok=True)
    train_log_path = os.path.join(run_model_dir, f"train_{args.task_name}_{run_hparam_tag}.log")
    _log_file = open(train_log_path, "w", encoding="utf-8")
    sys.stdout = _TeeStream(sys.__stdout__, _log_file)
    sys.stderr = _TeeStream(sys.__stderr__, _log_file)
    print(f"[run] model_dir={run_model_dir}")
    print(f"[run] train_log={train_log_path}")
    print(f"[run] hparams tag={run_hparam_tag}")

    best_model_path = os.path.join(run_model_dir, f"{args.task_name}_{run_hparam_tag}")
    best_loss_model_path = os.path.join(
        run_model_dir, f"{args.task_name}_lowest_valid_loss_{run_hparam_tag}"
    )
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
            "task_type": "paper_venue_ogbn_mag_shared",
            "metrics_path": metrics_path,
            "shared_base_dir": args.shared_base_dir,
        },
        default_group="paper_venue_ogbn_mag",
    )
    append_metrics("run_start")

    pool = None
    if args.n_pool > 1:
        pool = mp.get_context("fork").Pool(args.n_pool)

    device = torch.device(f"cuda:{args.cuda}") if args.cuda != -1 else torch.device("cpu")

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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000, eta_min=1e-6)

    best_val = float("-inf")
    best_loss = float("inf")
    valid_acc_at_best_valid_loss = None
    epoch_when_lowest_valid_loss_saved = 0
    st = time.time()
    evaluator = Evaluator(name="ogbn-mag")

    def evaluate_model(curr_gnn, curr_classifier, split_name):
        with torch.no_grad():
            test_acc_list = []
            for _ in range(10):
                sample = _node_classification_sample(randint(), "test", args.batch_size)
                node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = sample
                node_rep = curr_gnn(
                    node_feature.to(device),
                    node_type.to(device),
                    edge_time.to(device),
                    edge_index.to(device),
                    edge_type.to(device),
                )
                res = curr_classifier(node_rep[x_ids])
                test_acc = evaluator.eval(
                    {
                        "y_true": ylabel.unsqueeze(-1),
                        "y_pred": res.argmax(dim=1).cpu().unsqueeze(-1),
                    }
                )["acc"]
                test_acc_list.append(float(test_acc))

            avg_acc = float(np.average(test_acc_list))
            print(f"{split_name} Acc: {avg_acc:.4f}")
            append_metrics(
                split_name.lower().replace(" ", "_"),
                acc=avg_acc,
            )
            metric_key = wandb_metric_key(split_name)
            wandb_tracker.log(
                {
                    f"{metric_key}/acc": avg_acc,
                }
            )
            wandb_tracker.summary(
                {
                    f"{metric_key}_acc": avg_acc,
                }
            )
            return avg_acc

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
                    for node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel in tqdm(
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
                        res = classifier(node_rep[x_ids])
                        loss = criterion(res, ylabel.to(device))

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
                    node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = valid_data
                    node_rep = gnn(
                        node_feature.to(device),
                        node_type.to(device),
                        edge_time.to(device),
                        edge_index.to(device),
                        edge_type.to(device),
                    )
                    res = classifier(node_rep[x_ids])
                    loss = criterion(res, ylabel.to(device))

                    valid_acc = evaluator.eval(
                        {
                            "y_true": ylabel.unsqueeze(-1),
                            "y_pred": res.argmax(dim=1).cpu().unsqueeze(-1),
                        }
                    )["acc"]
                    curr_valid_loss = float(loss.cpu().detach().item())
                    is_best = float(valid_acc) > best_val
                    is_best_loss = curr_valid_loss < best_loss
                    if is_best:
                        best_val = float(valid_acc)
                        torch.save(model, best_model_path)
                        print("UPDATE!!! Best model by highest Valid Acc")
                    if is_best_loss:
                        best_loss = curr_valid_loss
                        valid_acc_at_best_valid_loss = float(valid_acc)
                        epoch_when_lowest_valid_loss_saved = int(epoch)
                        torch.save(model, best_loss_model_path)
                        print("UPDATE!!! Best model by lowest Valid Loss")

                    st = time.time()
                    epoch_time = float(st - et)
                    print(
                        (
                            "Epoch: %d (%.1fs)  LR: %.5f Train Loss: %.2f  "
                            "Valid Loss: %.2f  Valid Acc: %.4f"
                        )
                        % (
                            epoch,
                            epoch_time,
                            optimizer.param_groups[0]["lr"],
                            float(np.average(train_losses)),
                            curr_valid_loss,
                            float(valid_acc),
                        )
                    )
                    metrics_payload = {
                        "epoch": int(epoch),
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "train_loss": float(np.average(train_losses)),
                        "valid_loss": curr_valid_loss,
                        "valid_acc": float(valid_acc),
                        "best_valid_acc_updated": bool(is_best),
                        "best_valid_loss_updated": bool(is_best_loss),
                        "time_per_epoch": epoch_time,
                        "epoch_per_time": epoch_time,
                        "epoch_when_lowest_valid_loss_saved": int(epoch_when_lowest_valid_loss_saved),
                    }
                    if valid_acc_at_best_valid_loss is not None:
                        metrics_payload["valid_acc_at_best_valid_loss"] = float(
                            valid_acc_at_best_valid_loss
                        )
                    append_metrics("epoch_validation", **metrics_payload)
                    wandb_tracker.log(metrics_payload, step=int(epoch))
                    if is_best:
                        wandb_tracker.summary({"best_valid_acc": float(best_val)})
                    if is_best_loss:
                        wandb_tracker.summary(
                            {
                                "best_valid_loss": float(best_loss),
                                "valid_acc_at_best_valid_loss": float(valid_acc),
                                "epoch_when_lowest_valid_loss_saved": int(epoch),
                            }
                        )
                    del res, loss
                del train_data, valid_data
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        last_test_acc = evaluate_model(gnn, classifier, "Last Test")

        best_model = torch.load(best_model_path, map_location=device, weights_only=False)
        best_model.eval()
        gnn_best, classifier_best = best_model
        best_acc_ckpt_test_acc = evaluate_model(gnn_best, classifier_best, "Best Test (highest Valid Acc)")

        best_loss_model = torch.load(best_loss_model_path, map_location=device, weights_only=False)
        best_loss_model.eval()
        gnn_loss, classifier_loss = best_loss_model
        best_loss_ckpt_test_acc = evaluate_model(gnn_loss, classifier_loss, "Best Test (lowest Valid Loss)")

        wandb_tracker.log(
            {
                "final/test_acc_highest_valid_acc_ckpt": best_acc_ckpt_test_acc,
                "final/test_acc_lowest_valid_loss_ckpt": best_loss_ckpt_test_acc,
                "final/last_epoch_test_acc": last_test_acc,
            },
            step=int(args.n_epoch),
        )
        wandb_tracker.summary(
            {
                "test_acc/highest_valid_acc_checkpoint": best_acc_ckpt_test_acc,
                "test_acc/lowest_valid_loss_checkpoint": best_loss_ckpt_test_acc,
                "valid_acc_at_best_valid_loss": valid_acc_at_best_valid_loss,
                "epoch_when_lowest_valid_loss_saved": int(epoch_when_lowest_valid_loss_saved),
            }
        )
    except Exception:
        run_exit_code = 1
        raise
    finally:
        wandb_tracker.finish(exit_code=run_exit_code)
        if sys.stdout is not sys.__stdout__:
            sys.stdout.flush()
        if sys.stderr is not sys.__stderr__:
            sys.stderr.flush()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        if _log_file is not None:
            try:
                _log_file.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
