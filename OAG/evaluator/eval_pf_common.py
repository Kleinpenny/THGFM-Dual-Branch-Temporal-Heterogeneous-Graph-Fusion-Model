"""
Shared logic for paper-field (PF) test evaluation at L1 or L2 (THGFM checkpoints only).

Every paper in the test split is evaluated exactly once, in row-index order (batched). There is
no random subsampling of test rows; `--seed` only affects stochastic neighbor sampling inside
`sample_subgraph_shared` (same convention as training).

Multi-label targets are normalized to a distribution; metrics follow train_paper_field_shared.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
import time

import numpy as np
import torch
from tqdm.auto import tqdm

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_RESULTS_DIR = os.path.join(_CUR_DIR, "results")
_OAG_DIR = os.path.abspath(os.path.join(_CUR_DIR, ".."))
_TRAINER_MAIN = os.path.join(_OAG_DIR, "trainers", "main")
for _p in (_OAG_DIR, _TRAINER_MAIN):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import THGFM  # noqa: E402
from THGFM.utils import mean_reciprocal_rank, ndcg_at_k  # noqa: E402

sys.modules.setdefault("HGFM", THGFM)  # legacy pickles saved as HGFM.*
from tool_functions.shared_base_runtime import (  # noqa: E402
    SharedGraphBase,
    sample_subgraph_shared,
    to_torch_shared,
)

from eval_result_layout import resolve_eval_output_dir  # noqa: E402

try:
    import wandb
except Exception:
    wandb = None


class SharedPFBase(SharedGraphBase):
    def __init__(self, shared_dir: str, field_level: str):
        super().__init__(shared_dir=shared_dir)

        task = self.meta.get("task", {})
        if task.get("name") != "paper_field":
            raise ValueError(
                f"shared base at {shared_dir} is not paper_field task "
                f"(got task={task.get('name')})."
            )
        meta_level = str(task.get("field_level", ""))
        if meta_level and meta_level != field_level:
            raise ValueError(f"field_level mismatch: base={meta_level}, requested={field_level}")

        self.field_level = field_level
        self.relation_type = str(task["relation_type"])
        self.reverse_relation_type = str(task["reverse_relation_type"])
        self.candidate_field_ids = self.load_npy(task["candidate_ids"])
        self.cand_index = {int(fid): idx for idx, fid in enumerate(self.candidate_field_ids)}

        self.split_paper_ids = {}
        self.split_times = {}
        self.split_label_offsets = {}
        self.split_label_ids = {}
        for split in ("train", "valid", "test"):
            split_files = task["splits"][split]
            self.split_paper_ids[split] = self.load_npy(split_files["paper_ids"])
            self.split_times[split] = self.load_npy(split_files["times"])
            self.split_label_offsets[split] = self.load_npy(split_files["label_offsets"])
            self.split_label_ids[split] = self.load_npy(split_files["label_ids"])


def _mask_output_edges_pf(edge_list, batch_size: int, pf_rel: str, rev_pf_rel: str):
    if batch_size <= 0:
        return
    if (
        "paper" in edge_list
        and "field" in edge_list["paper"]
        and rev_pf_rel in edge_list["paper"]["field"]
    ):
        edge_list["paper"]["field"][rev_pf_rel] = [
            e for e in edge_list["paper"]["field"][rev_pf_rel] if int(e[0]) >= batch_size
        ]
    if "field" in edge_list and "paper" in edge_list["field"] and pf_rel in edge_list["field"]["paper"]:
        edge_list["field"]["paper"][pf_rel] = [
            e for e in edge_list["field"]["paper"][pf_rel] if int(e[1]) >= batch_size
        ]


def _labels_for_rows_pf(base: SharedPFBase, split_name: str, selected_rows: np.ndarray) -> torch.Tensor:
    num_classes = len(base.candidate_field_ids)
    y = np.zeros((len(selected_rows), num_classes), dtype=np.float32)
    offsets = base.split_label_offsets[split_name]
    label_ids = base.split_label_ids[split_name]
    for x_id, row_id in enumerate(selected_rows):
        st = int(offsets[int(row_id)])
        ed = int(offsets[int(row_id) + 1])
        for field_id in label_ids[st:ed]:
            cls_idx = base.cand_index.get(int(field_id))
            if cls_idx is not None:
                y[x_id, cls_idx] = 1.0
    row_sum = y.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0):
        raise RuntimeError(f"Unlabeled papers in split={split_name} for selected rows.")
    y = y / row_sum
    return torch.from_numpy(y)


def _build_batch_thgfm_pf(
    base: SharedPFBase,
    split_name: str,
    selected_rows: np.ndarray,
    sample_depth: int,
    sample_width: int,
    rng: np.random.Generator,
):
    paper_ids = base.split_paper_ids[split_name][selected_rows].astype(np.int64, copy=False)
    paper_times = base.split_times[split_name][selected_rows].astype(np.int64, copy=False)
    target_info = np.stack((paper_ids, paper_times), axis=1)

    feature, times, edge_list = sample_subgraph_shared(
        base,
        split_name=split_name,
        inp={"paper": target_info},
        sampled_depth=sample_depth,
        sampled_number=sample_width,
        rng=rng,
    )
    _mask_output_edges_pf(
        edge_list,
        len(selected_rows),
        base.relation_type,
        base.reverse_relation_type,
    )
    node_feature, node_type, edge_time, edge_index, edge_type, node_dict, _ = to_torch_shared(
        feature, times, edge_list, base
    )
    ylabel = _labels_for_rows_pf(base, split_name, selected_rows)
    x_ids = np.arange(len(selected_rows), dtype=np.int64) + node_dict["paper"][0]
    return node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel


def _append_pf_ranking(logits: torch.Tensor, ylabel: torch.Tensor, test_res: list):
    sorted_idx = logits.argsort(descending=True).cpu()
    y_np = ylabel.cpu().numpy()
    for ai, bi in zip(y_np, sorted_idx):
        test_res.append(ai[bi.cpu().numpy()])


def _resolve_checkpoint_path_pf(raw: str, field_level: str) -> str:
    if os.path.isfile(raw):
        return raw
    if os.path.isdir(raw):
        candidates = []
        for fn in os.listdir(raw):
            if "lowest_valid_loss" not in fn:
                continue
            path = os.path.join(raw, fn)
            if not os.path.isfile(path):
                continue
            if not fn.startswith("PF_"):
                continue
            if f"_{field_level}_" in fn or fn.endswith(f"_{field_level}_lowest_valid_loss"):
                candidates.append(fn)
        if not candidates:
            raise FileNotFoundError(
                f"No PF-{field_level} *_lowest_valid_loss checkpoint under directory: {raw}"
            )
        candidates.sort()
        chosen = os.path.join(raw, candidates[0])
        if len(candidates) > 1:
            print(f"[resolve] multiple PF-{field_level} lowest_valid_loss files; using: {candidates[0]}")
        return chosen
    raise FileNotFoundError(f"Checkpoint path not found: {raw}")


def _test_paper_count_pf(base: SharedPFBase, split_name: str) -> int:
    n = int(len(base.split_paper_ids[split_name]))
    if int(len(base.split_times[split_name])) != n:
        raise ValueError(
            f"{split_name} split length mismatch: paper_ids={n}, times={len(base.split_times[split_name])}"
        )
    n_off = int(len(base.split_label_offsets[split_name]))
    if n_off != n + 1:
        raise ValueError(
            f"{split_name} label_offsets length mismatch: expected paper_count+1={n + 1}, got {n_off}"
        )
    return n


def default_shared_base(field_level: str) -> str:
    return os.path.abspath(
        os.path.join(_OAG_DIR, "..", "data", f"shared_pf_base_CS_XLNet_{field_level}")
    )


def run_main(field_level: str, argv: list[str] | None = None) -> int:
    if field_level not in ("L1", "L2"):
        raise ValueError(f"field_level must be L1 or L2, got {field_level!r}")

    parser = argparse.ArgumentParser(
        description=f"PF-{field_level} test NDCG / MRR on OAG shared base (multi-label, THGFM)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint file or directory with PF-*_lowest_valid_loss for this level.",
    )
    parser.add_argument(
        "--shared_base_dir",
        type=str,
        default=default_shared_base(field_level),
        help=f"Directory with meta.json for paper_field {field_level}.",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--sample_depth", type=int, default=6)
    parser.add_argument("--sample_width", type=int, default=128)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument(
        "--max_test_rows",
        type=int,
        default=None,
        help="If set, evaluate only the first N test rows (smoke / quick debug).",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=_RESULTS_DIR,
        help="Root directory for eval outputs (default: evaluator/results); "
        "summaries go under <save_dir>/PF_<L>/MODEL/.",
    )
    parser.add_argument(
        "--model_subdir",
        type=str,
        default="",
        help="Subfolder name under the task directory (default: infer from checkpoint parent dir).",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="OAG-EVL",
        help="wandb project name when --wandb is enabled.",
    )
    args = parser.parse_args(argv)
    progress_every = int(os.environ.get("EVAL_PROGRESS_EVERY", "10"))

    ckpt_path = _resolve_checkpoint_path_pf(args.checkpoint, field_level)
    device = torch.device(f"cuda:{args.cuda}") if args.cuda >= 0 else torch.device("cpu")
    tag = os.path.basename(os.path.normpath(ckpt_path))

    wb_run = None
    if args.wandb:
        if wandb is None:
            print("[warn] wandb is not installed; skip wandb logging.", flush=True)
        else:
            wb_run = wandb.init(
                project=args.wandb_project,
                name=f"PF-{field_level}-seed{args.seed}-{tag}",
                config={
                    "task": f"PF-{field_level}",
                    "checkpoint": os.path.abspath(ckpt_path),
                    "seed": int(args.seed),
                    "cuda": int(args.cuda),
                },
                reinit=True,
            )
            wandb.log({"status": 0.0})

    rng = np.random.default_rng(args.seed)
    base = SharedPFBase(args.shared_base_dir, field_level=field_level)
    split_name = "test"
    n_test_full = _test_paper_count_pf(base, split_name)
    all_rows = np.arange(n_test_full, dtype=np.int64)
    if args.max_test_rows is not None:
        cap = int(args.max_test_rows)
        if cap <= 0:
            raise ValueError("--max_test_rows must be positive")
        all_rows = all_rows[:cap]
    n_test = int(len(all_rows))
    if args.max_test_rows is None:
        print(
            f"[eval][PF-{field_level}] {split_name}: {n_test} papers — evaluating ALL row indices "
            f"0..{n_test - 1} in order (no random subsampling of the test split).",
            flush=True,
        )
    else:
        print(
            f"[eval][PF-{field_level}] {split_name}: partial eval first {n_test} of {n_test_full} "
            f"test rows (--max_test_rows).",
            flush=True,
        )

    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(raw, dict) and "model" in raw:
        raise ValueError(
            "Checkpoint looks like a state_dict bundle; use a full nn.Sequential from train_paper_field_shared.py."
        )

    backend = "thgfm"
    model = raw
    model.eval()
    gnn, classifier = model[0], model[1]
    test_res: list = []

    total_steps = (len(all_rows) + args.batch_size - 1) // args.batch_size
    start_ts = time.time()
    with torch.no_grad():
        for step_idx, st in enumerate(
            tqdm(
                range(0, len(all_rows), args.batch_size),
                desc=f"THGFM PF-{field_level} {split_name}",
            ),
            start=1,
        ):
            ed = min(st + args.batch_size, len(all_rows))
            batch_rows = all_rows[st:ed]
            sample = _build_batch_thgfm_pf(
                base,
                split_name,
                batch_rows,
                args.sample_depth,
                args.sample_width,
                rng,
            )
            node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = sample
            node_rep = gnn(
                node_feature.to(device),
                node_type.to(device),
                edge_time.to(device),
                edge_index.to(device),
                edge_type.to(device),
            )
            logits = classifier(node_rep[x_ids])
            _append_pf_ranking(logits, ylabel, test_res)
            if progress_every > 0 and (step_idx % progress_every == 0 or step_idx == total_steps):
                pct = 100.0 * step_idx / max(total_steps, 1)
                elapsed = time.time() - start_ts
                print(
                    f"[progress][PF-{field_level}][thgfm] step={step_idx}/{total_steps} "
                    f"({pct:.1f}%) elapsed={elapsed:.1f}s",
                    flush=True,
                )

    if len(test_res) != n_test:
        raise RuntimeError(
            f"Expected one metric list per test paper ({n_test}), got {len(test_res)} — "
            "evaluation loop may be out of sync with the test split."
        )

    ndcg_vals = [ndcg_at_k(resi, len(resi)) for resi in test_res]
    mrr_vals = mean_reciprocal_rank(test_res)
    avg_ndcg = float(np.average(ndcg_vals))
    avg_mrr = float(np.average(mrr_vals))

    task_subdir = f"PF_{field_level}"
    out_dir = resolve_eval_output_dir(
        args.save_dir,
        task_subdir=task_subdir,
        model_subdir=args.model_subdir or None,
        checkpoint_path=ckpt_path,
    )
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(out_dir, f"eval_pf_{field_level}_test_seed{args.seed}_{tag}_{ts}.json")

    summary = {
        "task": f"PF-{field_level}",
        "split": split_name,
        "evaluates_all_test_papers": args.max_test_rows is None,
        "test_paper_count": n_test,
        "full_test_split_size": n_test_full,
        "max_test_rows": args.max_test_rows,
        "total_samples": int(len(test_res)),
        "ndcg": avg_ndcg,
        "mrr": avg_mrr,
        "checkpoint": os.path.abspath(ckpt_path),
        "backend": backend,
        "shared_base_dir": os.path.abspath(args.shared_base_dir),
        "results_dir": os.path.abspath(out_dir),
        "field_level": field_level,
        "batch_size": args.batch_size,
        "sample_depth": args.sample_depth,
        "sample_width": args.sample_width,
        "seed": args.seed,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)

    if wb_run is not None:
        wandb.log({"status": 1.0, "ndcg": avg_ndcg, "mrr": avg_mrr, "total_samples": int(len(test_res))})
        wb_run.finish()

    print(f"PF-{field_level} split={split_name} total_samples={len(test_res)}")
    print(f"NDCG={avg_ndcg:.6f}  MRR={avg_mrr:.6f}")
    print(f"saved_summary={summary_path}")
    return 0
