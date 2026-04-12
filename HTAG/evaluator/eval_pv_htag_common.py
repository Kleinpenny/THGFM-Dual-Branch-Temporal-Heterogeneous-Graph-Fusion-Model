from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import torch

_CUR = os.path.dirname(os.path.abspath(__file__))
_HTAG = os.path.abspath(os.path.join(_CUR, ".."))
if _HTAG not in sys.path:
    sys.path.insert(0, _HTAG)

from THGFM.model import Classifier, GNN  # noqa: E402
from THGFM.wandb_utils import add_wandb_args, init_wandb_tracker  # noqa: E402

from eval_result_layout import infer_task_subdir, resolve_eval_output_dir  # noqa: E402


def _node_classification_sample_fixed_rows(trainer, seed: int, split_name: str, row_indices: np.ndarray):
    """
    Same subgraph pipeline as trainer._node_classification_sample, but evaluates an explicit
    contiguous (or arbitrary) set of row indices into split_paper_ids[split_name] instead of
    drawing a random subset.
    """
    rng = np.random.default_rng(int(seed))
    batch_size = int(len(row_indices))
    if batch_size <= 0:
        raise ValueError("row_indices must be non-empty")
    base = trainer._BASE
    run_args = trainer._RUN_ARGS
    paper_ids = base.split_paper_ids[split_name][row_indices].astype(np.int64, copy=False)
    paper_times = base.split_times[split_name][row_indices].astype(np.int64, copy=False)
    target_info = np.stack((paper_ids, paper_times), axis=1)

    feature, times, edge_list = trainer.sample_subgraph_shared(
        base,
        split_name=split_name,
        inp={"paper": target_info},
        sampled_depth=run_args.sample_depth,
        sampled_number=run_args.sample_width,
        rng=rng,
    )
    trainer._mask_output_edges(edge_list=edge_list, batch_size=batch_size)
    node_feature, node_type, edge_time, edge_index, edge_type, node_dict, _ = trainer.to_torch_shared(
        feature, times, edge_list, base
    )

    ylabel = torch.zeros(batch_size, dtype=torch.long)
    split_venue_ids = base.split_venue_ids[split_name]
    for x_id, row_id in enumerate(row_indices):
        venue_id = int(split_venue_ids[int(row_id)])
        cls_idx = base.cand_index.get(venue_id)
        if cls_idx is None:
            raise RuntimeError(
                f"Venue id {venue_id} from split={split_name} not found in candidate set."
            )
        ylabel[x_id] = int(cls_idx)
    x_ids = np.arange(batch_size, dtype=np.int64) + node_dict["paper"][0]
    return node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel


def _infer_conv_and_rota_from_ckpt(path: str):
    name = os.path.basename(path)
    stem = name[:-3] if name.endswith(".pt") else name
    parts = stem.split("_")
    conv_name = "rtta"
    rota_mode = "qk"
    rota_keep_rte = 1
    rtta_rota_mode = ""
    ssta_rota_mode = ""
    rtta_rota_keep_rte = -1
    ssta_rota_keep_rte = -1

    # Filenames from HTAG trainers: ``HTAG_{Arxiv|DBLP}_{conv}_rNNN_...`` (conv at index 2).
    # Pool / PV-prefixed names: ``PV_HTAG_{dataset}_{conv}_...`` (conv at index 3).
    if len(parts) >= 3 and parts[0] == "HTAG":
        conv_name = parts[2]
    elif len(parts) >= 4:
        conv_name = parts[3]
    if conv_name == "hgt":
        raise ValueError(
            "Checkpoint name indicates conv_name=hgt, which is no longer supported. Retrain with rtta or another variant."
        )
    for i, p in enumerate(parts):
        if p.startswith("r") and len(p) == 4 and i + 1 < len(parts) and parts[i + 1].startswith("s"):
            r = p
            if r == "r110":
                rota_keep_rte = 0
                rtta_rota_mode = "qk"
                ssta_rota_mode = "qk"
                rtta_rota_keep_rte = 0
                ssta_rota_keep_rte = 0
            elif r == "r111":
                rota_keep_rte = 1
                rtta_rota_mode = "qk"
                ssta_rota_mode = "qk"
                rtta_rota_keep_rte = 1
                ssta_rota_keep_rte = 1
            break
    return {
        "conv_name": conv_name,
        "rota_mode": rota_mode,
        "rota_keep_rte": rota_keep_rte,
        "rtta_rota_mode": rtta_rota_mode,
        "ssta_rota_mode": ssta_rota_mode,
        "rtta_rota_keep_rte": rtta_rota_keep_rte,
        "ssta_rota_keep_rte": ssta_rota_keep_rte,
    }


def build_parser(default_shared: str, default_task_subdir: str):
    p = argparse.ArgumentParser(description=f"Evaluate HTAG {default_task_subdir} PV checkpoint.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--shared_base_dir", type=str, default=default_shared)
    p.add_argument("--task_subdir", type=str, default=default_task_subdir)
    p.add_argument("--save_dir", type=str, default=os.path.join(_CUR, "results"))
    p.add_argument("--output_json", type=str, default="")
    p.add_argument("--model_subdir", type=str, default="")
    p.add_argument("--flat_save", action="store_true")
    p.add_argument("--no_save", action="store_true")
    p.add_argument("--cuda", type=int, default=0)
    p.add_argument("--n_hid", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--sample_depth", type=int, default=6)
    p.add_argument("--sample_width", type=int, default=128)
    p.add_argument("--eval_rounds", type=int, default=10)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument(
        "--max_test_rows",
        type=int,
        default=None,
        help="If set, evaluate only the first N test rows (smoke / quick debug).",
    )
    p.add_argument("--deterministic", action="store_true")
    add_wandb_args(p, default_project="HTAG-EVL")
    return p


def run_eval(dataset: str):
    if dataset == "Arxiv":
        from Arxiv_PV import train_paper_venue_shared as trainer  # noqa: E402

        default_shared = os.path.join(_HTAG, "Arxiv_PV", "data", "shared_pv_base_HTAG_arxiv")
    elif dataset == "DBLP":
        from DBLP_PV import train_paper_venue_shared as trainer  # noqa: E402

        default_shared = os.path.join(_HTAG, "DBLP_PV", "data", "shared_pv_base_HTAG_dblp")
    else:
        raise ValueError(dataset)

    parser = build_parser(default_shared=default_shared, default_task_subdir=dataset)
    args = parser.parse_args()
    ckpt = args.checkpoint
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    trainer._set_global_seed(args.seed, args.deterministic)
    base = trainer.SharedPVBase(args.shared_base_dir)

    trainer._BASE = base
    trainer._RUN_ARGS = SimpleNamespace(
        sample_depth=args.sample_depth, sample_width=args.sample_width, batch_size=args.batch_size
    )
    trainer._SELECTED_ROWS = {
        "train": np.arange(len(base.split_paper_ids["train"]), dtype=np.int64),
        "valid": np.arange(len(base.split_paper_ids["valid"]), dtype=np.int64),
        "test": np.arange(len(base.split_paper_ids["test"]), dtype=np.int64),
    }

    for split in ("valid", "test"):
        if len(trainer._SELECTED_ROWS[split]) == 0:
            raise RuntimeError(f"{split} split is empty.")
    if len(trainer._SELECTED_ROWS["valid"]) < args.batch_size:
        raise RuntimeError(
            f"valid rows ({len(trainer._SELECTED_ROWS['valid'])}) < batch_size ({args.batch_size}) "
            "(needed for stochastic valid batches)."
        )

    device = torch.device(f"cuda:{args.cuda}") if args.cuda >= 0 and torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    cfg = _infer_conv_and_rota_from_ckpt(ckpt)
    in_dim = int(base.feature_dims["paper"])
    gnn = GNN(
        conv_name=cfg["conv_name"],
        in_dim=in_dim,
        n_hid=args.n_hid,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        num_types=len(base.node_types),
        num_relations=len(base.meta_graph) + 1,
        rota_mode=cfg["rota_mode"],
        rota_keep_rte=bool(cfg["rota_keep_rte"]),
        rtta_rota_mode=(cfg["rtta_rota_mode"] if cfg["rtta_rota_mode"] else None),
        ssta_rota_mode=(cfg["ssta_rota_mode"] if cfg["ssta_rota_mode"] else None),
        rtta_rota_keep_rte=(None if cfg["rtta_rota_keep_rte"] == -1 else bool(cfg["rtta_rota_keep_rte"])),
        ssta_rota_keep_rte=(None if cfg["ssta_rota_keep_rte"] == -1 else bool(cfg["ssta_rota_keep_rte"])),
    ).to(device)
    classifier = Classifier(args.n_hid, len(base.candidate_venue_ids)).to(device)

    state = torch.load(ckpt, map_location=device)
    if "gnn" in state and "classifier" in state:
        gnn.load_state_dict(state["gnn"])
        classifier.load_state_dict(state["classifier"])
    elif "model" in state:
        raise RuntimeError("Unsupported checkpoint format (expected 'gnn' and 'classifier' state dicts).")
    else:
        raise KeyError(f"Unexpected checkpoint format: {list(state.keys())}")

    t0 = time.time()
    with torch.no_grad():
        valid_acc = []
        for _ in range(args.eval_rounds):
            s = trainer._node_classification_sample(trainer.randint(), "valid", args.batch_size)
            node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = s
            node_rep = gnn(
                node_feature.to(device),
                node_type.to(device),
                edge_time.to(device),
                edge_index.to(device),
                edge_type.to(device),
            )
            pred = classifier(node_rep[x_ids]).argmax(dim=1).cpu()
            valid_acc.append(float((pred == ylabel).float().mean().item()))

        test_row_ids = trainer._SELECTED_ROWS["test"]
        if args.max_test_rows is not None:
            cap = int(args.max_test_rows)
            if cap <= 0:
                raise ValueError("--max_test_rows must be positive")
            test_row_ids = test_row_ids[:cap]
        test_n = int(len(test_row_ids))
        test_correct = 0
        for start in range(0, test_n, args.batch_size):
            chunk = test_row_ids[start : start + args.batch_size]
            s = _node_classification_sample_fixed_rows(
                trainer, int(args.seed) + int(start), "test", chunk
            )
            node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = s
            node_rep = gnn(
                node_feature.to(device),
                node_type.to(device),
                edge_time.to(device),
                edge_index.to(device),
                edge_type.to(device),
            )
            pred = classifier(node_rep[x_ids]).argmax(dim=1).cpu()
            test_correct += int((pred == ylabel).sum().item())
    elapsed = time.time() - t0

    valid_acc_m = float(np.mean(valid_acc))
    test_acc_m = float(test_correct / test_n) if test_n else 0.0
    args.task_name = f"eval_{os.path.basename(os.path.abspath(ckpt))}_s{args.seed}"
    wb = init_wandb_tracker(
        args,
        extra_config={"task_type": "eval_pv_shared", "checkpoint": os.path.abspath(ckpt), "dataset": dataset},
        default_group=f"eval-{dataset.lower()}",
        default_job_type="eval",
    )
    wb.log(
        {
            "valid_acc": valid_acc_m,
            "test_acc": test_acc_m,
            "test_n": test_n,
            "elapsed_sec": round(elapsed, 3),
            "seed": args.seed,
        }
    )
    wb.summary({"best_valid_acc": valid_acc_m, "best_test_acc": test_acc_m})
    wb.finish(exit_code=0)

    full_test_n = int(len(trainer._SELECTED_ROWS["test"]))
    out = {
        "task": (args.task_subdir or "").strip() or infer_task_subdir(args.shared_base_dir),
        "checkpoint": os.path.abspath(ckpt),
        "shared_base_dir": os.path.abspath(args.shared_base_dir),
        "conv_name": cfg["conv_name"],
        "batch_size": args.batch_size,
        "sample_depth": args.sample_depth,
        "sample_width": args.sample_width,
        "eval_rounds": args.eval_rounds,
        "seed": args.seed,
        "valid_acc": valid_acc_m,
        "test_acc": test_acc_m,
        "test_n": test_n,
        "test_eval": "full_split" if args.max_test_rows is None else "partial_first_n_rows",
        "full_test_split_size": full_test_n,
        "max_test_rows": args.max_test_rows,
        "elapsed_sec": round(elapsed, 3),
    }
    print(json.dumps(out, indent=2, ensure_ascii=True))

    if args.no_save:
        return
    if args.output_json.strip():
        out_path = args.output_json.strip()
        parent = os.path.dirname(os.path.abspath(out_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
    else:
        out_dir = resolve_eval_output_dir(
            args.save_dir,
            task_subdir=out["task"],
            model_subdir=args.model_subdir or None,
            checkpoint_path=ckpt,
            flat_save=args.flat_save,
        )
        os.makedirs(out_dir, exist_ok=True)
        ckpt_tag = os.path.basename(os.path.normpath(ckpt))
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(out_dir, f"eval_{ckpt_tag}_{ts}.json")
        out["results_dir"] = os.path.abspath(out_dir)
    if args.output_json.strip():
        out.setdefault("results_dir", os.path.dirname(os.path.abspath(out_path)))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=True)
    print(f"[OK] wrote {out_path}", flush=True)
