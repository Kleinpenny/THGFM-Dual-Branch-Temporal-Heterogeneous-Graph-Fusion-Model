"""
PV (paper→venue) accuracy evaluator for ogbn-mag.

Iterates every row in the chosen split(s) (train/valid/test) in index order.
There is no random subsampling of split rows; --seed only affects stochastic
neighbor sampling inside sample_subgraph_shared (same idea as training).

Use ``--splits valid,test`` to evaluate both splits in one run and emit
``valid_acc`` / ``test_acc`` (plus ``accuracy``, which matches test when test is included).
"""
import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
from ogb.nodeproppred import Evaluator
from tqdm.auto import tqdm


_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if _CUR_DIR not in sys.path:
    sys.path.insert(0, _CUR_DIR)
_OGBN_DIR = os.path.abspath(os.path.join(_CUR_DIR, ".."))
_DEFAULT_JSON_SAVE_DIR = os.path.join(_CUR_DIR, "results")
_MAIN_DIR = os.path.join(_OGBN_DIR, "trainers", "main")
if _OGBN_DIR not in sys.path:
    sys.path.insert(0, _OGBN_DIR)
if _MAIN_DIR not in sys.path:
    sys.path.insert(0, _MAIN_DIR)

from trainers.main.tool_functions.shared_base_runtime import (  # noqa: E402
    SharedGraphBase,
    sample_subgraph_shared,
    to_torch_shared,
)
from tool_functions.wandb_utils import add_wandb_args, init_wandb_tracker  # noqa: E402

from eval_result_layout import TASK_SUBDIR_OGBN_MAG, resolve_eval_output_dir  # noqa: E402

import THGFM  # noqa: E402

sys.modules.setdefault("HGFM", THGFM)  # legacy pickles saved as HGFM.*


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


def _build_batch(base: SharedPVBase, args, split_name: str, selected_rows: np.ndarray, rng):
    paper_ids = base.split_paper_ids[split_name][selected_rows].astype(np.int64, copy=False)
    paper_times = base.split_times[split_name][selected_rows].astype(np.int64, copy=False)
    target_info = np.stack((paper_ids, paper_times), axis=1)

    feature, times, edge_list = sample_subgraph_shared(
        base,
        split_name=split_name,
        inp={"paper": target_info},
        sampled_depth=args.sample_depth,
        sampled_number=args.sample_width,
        rng=rng,
    )
    _mask_output_edges(edge_list=edge_list, batch_size=len(selected_rows))
    node_feature, node_type, edge_time, edge_index, edge_type, node_dict, _ = to_torch_shared(
        feature, times, edge_list, base
    )

    ylabel = torch.zeros(len(selected_rows), dtype=torch.long)
    split_venue_ids = base.split_venue_ids[split_name]
    for x_id, row_id in enumerate(selected_rows):
        venue_id = int(split_venue_ids[int(row_id)])
        cls_idx = base.cand_index.get(venue_id)
        if cls_idx is None:
            raise RuntimeError(
                f"Venue id {venue_id} from split={split_name} not found in candidate set."
            )
        ylabel[x_id] = int(cls_idx)

    x_ids = np.arange(len(selected_rows), dtype=np.int64) + node_dict["paper"][0]
    return node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel


def _extract_model_hparams(gnn):
    n_hid = int(getattr(gnn, "n_hid", -1))
    n_layers = int(len(getattr(gnn, "gcs", [])))
    n_heads = -1
    gcs = getattr(gnn, "gcs", [])
    if len(gcs) > 0 and hasattr(gcs[0].base_conv, "n_heads"):
        n_heads = int(gcs[0].base_conv.n_heads)
    return n_hid, n_heads, n_layers


def _parse_splits_arg(split_single: str, splits_csv: str) -> list[str]:
    allowed = {"train", "valid", "test"}
    if splits_csv.strip():
        out = []
        for p in splits_csv.split(","):
            s = p.strip().lower()
            if not s:
                continue
            if s not in allowed:
                raise ValueError(f"Unknown split {s!r} in --splits (use train, valid, test).")
            out.append(s)
        if not out:
            raise ValueError("--splits is empty after parsing.")
        return out
    s = split_single.strip().lower()
    if s not in allowed:
        raise ValueError(f"Unknown --split {split_single!r}.")
    return [s]


def _eval_one_split(
    base: SharedPVBase,
    gnn,
    classifier,
    device: torch.device,
    args,
    split_name: str,
    ogb_evaluator: Evaluator,
    tracker,
) -> tuple[float, int, int]:
    """Returns (accuracy, n_evaluated_rows, full_split_size)."""
    n_split_full = len(base.split_paper_ids[split_name])
    if n_split_full == 0:
        raise RuntimeError(f"No rows found for split={split_name}.")
    all_rows = np.arange(n_split_full, dtype=np.int64)
    if args.max_test_rows is not None:
        cap = int(args.max_test_rows)
        if cap <= 0:
            raise ValueError("--max_test_rows must be positive")
        all_rows = all_rows[:cap]
    n_split = int(len(all_rows))

    y_true_list = []
    y_pred_list = []
    rng = np.random.default_rng(args.seed)

    with torch.no_grad():
        for st in tqdm(range(0, len(all_rows), args.batch_size), desc=f"Eval {split_name}"):
            ed = min(st + args.batch_size, len(all_rows))
            batch_rows = all_rows[st:ed]
            batch_idx = st // args.batch_size + 1
            total_batches = (len(all_rows) + args.batch_size - 1) // args.batch_size
            sample = _build_batch(base, args, split_name, batch_rows, rng)
            node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = sample

            node_rep = gnn(
                node_feature.to(device),
                node_type.to(device),
                edge_time.to(device),
                edge_index.to(device),
                edge_type.to(device),
            )
            logits = classifier(node_rep[x_ids])
            pred = torch.argmax(logits, dim=-1).cpu().numpy().reshape(-1, 1)
            y_pred_list.append(pred)
            y_true_list.append(ylabel.cpu().numpy().reshape(-1, 1))
            if tracker.enabled and (batch_idx == 1 or batch_idx % 50 == 0 or batch_idx == total_batches):
                tracker.log(
                    {
                        "eval/progress_batch": int(batch_idx),
                        "eval/progress_total_batches": int(total_batches),
                        "eval/progress_ratio": float(batch_idx / total_batches),
                        "eval/seed": int(args.seed),
                        "eval/split": split_name,
                    }
                )

    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)
    if int(y_true.shape[0]) != int(n_split):
        raise RuntimeError(
            f"Internal error: evaluated {y_true.shape[0]} rows but expected {n_split}."
        )
    result = ogb_evaluator.eval({"y_true": y_true, "y_pred": y_pred})
    return float(result["acc"]), n_split, n_split_full


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate saved PV_ogbn_mag checkpoint on OGBN-MAG splits (OGB accuracy)."
    )
    parser.add_argument(
        "--shared_base_dir",
        type=str,
        default=os.path.join(_OGBN_DIR, "data", "shared_pv_base_ogbn_mag_graph"),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(
            _OGBN_DIR, "trainers", "main", "model_save", "rtta", "PV_ogbn_mag"
        ),
        help="Path to saved model checkpoint.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "valid", "train"],
        help="Single split when --splits is not set.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="",
        help="Comma-separated splits to evaluate in one run (e.g. valid,test). "
        "When set, overrides --split. When empty (default), only --split is evaluated.",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--sample_depth", type=int, default=6)
    parser.add_argument("--sample_width", type=int, default=128)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument(
        "--seed",
        type=int,
        default=43,
        help="RNG seed for subgraph neighbor sampling only (all split rows are still evaluated).",
    )
    parser.add_argument(
        "--max_test_rows",
        type=int,
        default=None,
        help="If set, evaluate only the first N rows of the chosen split (smoke / quick debug).",
    )
    parser.add_argument(
        "--clip",
        type=float,
        default=None,
        help="Training clip value to be recorded in output json.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=_DEFAULT_JSON_SAVE_DIR,
        help=f"Root directory for evaluation JSON (default: {_DEFAULT_JSON_SAVE_DIR}). "
        "Unless --flat_save, files go under <save_dir>/OGBN-MAG/<MODEL>/.",
    )
    parser.add_argument(
        "--model_subdir",
        type=str,
        default="",
        help="Subfolder under OGBN-MAG (default: infer from checkpoint parent directory).",
    )
    parser.add_argument(
        "--flat_save",
        action="store_true",
        help="Write JSON directly under --save_dir (no OGBN-MAG/MODEL nesting).",
    )
    add_wandb_args(parser, default_project="OGBN-EVL")
    args = parser.parse_args()
    split_list = _parse_splits_arg(args.split, args.splits)
    splits_tag = "_".join(split_list)

    tracker = init_wandb_tracker(
        args,
        extra_config={
            "checkpoint": os.path.abspath(args.checkpoint),
            "splits": split_list,
            "task": TASK_SUBDIR_OGBN_MAG,
        },
        default_group=f"ogbn-eval-{splits_tag}",
        default_job_type="eval",
    )

    device = torch.device(f"cuda:{args.cuda}") if args.cuda != -1 else torch.device("cpu")
    base = SharedPVBase(args.shared_base_dir)

    model = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.eval()
    gnn, classifier = model
    gnn.eval()
    classifier.eval()
    n_hid, n_heads, n_layers = _extract_model_hparams(gnn)

    ogb_evaluator = Evaluator(name="ogbn-mag")
    per_split: dict[str, dict[str, float | int]] = {}
    valid_acc = None
    test_acc = None
    primary_acc = None

    for sp in split_list:
        acc, n_ev, n_full = _eval_one_split(
            base, gnn, classifier, device, args, sp, ogb_evaluator, tracker
        )
        per_split[sp] = {
            "accuracy": acc,
            "total_samples": n_ev,
            "full_split_size": n_full,
        }
        if sp == "valid":
            valid_acc = acc
        if sp == "test":
            test_acc = acc

    if test_acc is not None:
        primary_acc = test_acc
    elif valid_acc is not None:
        primary_acc = valid_acc
    else:
        primary_acc = float(per_split[split_list[-1]]["accuracy"])

    out_dir = resolve_eval_output_dir(
        args.save_dir,
        model_subdir=args.model_subdir or None,
        checkpoint_path=args.checkpoint,
        flat_save=args.flat_save,
    )
    os.makedirs(out_dir, exist_ok=True)
    ckpt_tag = os.path.basename(os.path.normpath(args.checkpoint))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(
        out_dir, f"eval_{splits_tag}_{ckpt_tag}_seed{int(args.seed)}_{timestamp}.json"
    )

    summary_payload = {
        "task": TASK_SUBDIR_OGBN_MAG,
        "splits": split_list,
        "per_split": per_split,
        "max_test_rows": args.max_test_rows,
        "evaluates_full_split": args.max_test_rows is None,
        "accuracy": primary_acc,
        "valid_acc": valid_acc,
        "test_acc": test_acc,
        "checkpoint": os.path.abspath(args.checkpoint),
        "shared_base_dir": os.path.abspath(args.shared_base_dir),
        "batch_size": int(args.batch_size),
        "sample_depth": int(args.sample_depth),
        "sample_width": int(args.sample_width),
        "seed": int(args.seed),
        "n_hid": n_hid,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "clip": (None if args.clip is None else float(args.clip)),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "results_dir": os.path.abspath(out_dir),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=True, indent=2)

    if tracker.enabled:
        log_payload = {
            "eval/accuracy": primary_acc,
            "eval/seed": int(args.seed),
            "eval/splits": ",".join(split_list),
        }
        if valid_acc is not None:
            log_payload["eval/valid_acc"] = valid_acc
        if test_acc is not None:
            log_payload["eval/test_acc"] = test_acc
        tracker.log(log_payload)
        sum_payload = {
            "accuracy": primary_acc,
            "seed": int(args.seed),
            "splits": split_list,
            "checkpoint": os.path.abspath(args.checkpoint),
            "saved_summary": os.path.abspath(summary_path),
        }
        if valid_acc is not None:
            sum_payload["valid_acc"] = valid_acc
        if test_acc is not None:
            sum_payload["test_acc"] = test_acc
        tracker.summary(sum_payload)
    tracker.finish(exit_code=0)

    for sp in split_list:
        ps = per_split[sp]
        print(f"split={sp} total_samples={ps['total_samples']}")
        print(f"accuracy={float(ps['accuracy']):.6f}")
    if valid_acc is not None:
        print(f"valid_acc={valid_acc:.6f}")
    if test_acc is not None:
        print(f"test_acc={test_acc:.6f}")
    print(f"accuracy={primary_acc:.6f}")
    print(f"saved_summary={summary_path}")


if __name__ == "__main__":
    main()
