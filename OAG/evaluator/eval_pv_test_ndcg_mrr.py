"""
Evaluate paper-venue (PV) checkpoints on the OAG shared-base *test* split with NDCG and MRR.

Every paper in the test split is evaluated exactly once, in row-index order (batched). There is
no random subsampling of test rows; `--seed` only affects stochastic neighbor sampling inside
`sample_subgraph_shared` (same convention as training).

Expects a full ``nn.Sequential`` checkpoint from ``train_paper_venue_shared.py`` (GNN + classifier).
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


class SharedPVBase(SharedGraphBase):
    def __init__(self, shared_dir: str):
        super().__init__(shared_dir=shared_dir)

        task = self.meta.get("task", {})
        if task.get("name") != "paper_venue":
            raise ValueError(
                f"shared base at {shared_dir} is not paper_venue task "
                f"(got task={task.get('name')})."
            )

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


def _build_batch_thgfm(
    base: SharedPVBase,
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


def _append_ranking_from_logits(logits: torch.Tensor, ylabel: torch.Tensor, test_res: list):
    sorted_idx = logits.argsort(descending=True).cpu()
    for ai, bi in zip(ylabel, sorted_idx):
        test_res.append((bi == int(ai.item())).int().tolist())


def _test_paper_count_pv(base: SharedPVBase, split_name: str) -> int:
    n = int(len(base.split_paper_ids[split_name]))
    if int(len(base.split_times[split_name])) != n:
        raise ValueError(
            f"{split_name} split length mismatch: paper_ids={n}, times={len(base.split_times[split_name])}"
        )
    if int(len(base.split_venue_ids[split_name])) != n:
        raise ValueError(
            f"{split_name} split length mismatch: paper_ids={n}, venue_ids={len(base.split_venue_ids[split_name])}"
        )
    return n


def _resolve_checkpoint_path(raw: str) -> str:
    if os.path.isfile(raw):
        return raw
    if os.path.isdir(raw):
        candidates = []
        for fn in os.listdir(raw):
            if not fn.startswith("PV_"):
                continue
            if "lowest_valid_loss" not in fn:
                continue
            if os.path.isfile(os.path.join(raw, fn)):
                candidates.append(fn)
        if not candidates:
            raise FileNotFoundError(
                f"No PV *_lowest_valid_loss checkpoint under directory: {raw}"
            )
        candidates.sort()
        chosen = os.path.join(raw, candidates[0])
        if len(candidates) > 1:
            print(f"[resolve] multiple PV lowest_valid_loss files; using: {candidates[0]}")
        return chosen
    raise FileNotFoundError(f"Checkpoint path not found: {raw}")


def main():
    parser = argparse.ArgumentParser(description="PV test NDCG / MRR on OAG shared base (THGFM checkpoints)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file, or directory containing PV_*_lowest_valid_loss",
    )
    parser.add_argument(
        "--shared_base_dir",
        type=str,
        default=os.path.abspath(os.path.join(_OAG_DIR, "..", "data", "shared_pv_base_CS_XLNet")),
        help="Directory with meta.json for paper_venue (build_shared_pv_base output).",
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
        "summaries go under <save_dir>/PV/MODEL/.",
    )
    parser.add_argument(
        "--model_subdir",
        type=str,
        default="",
        help="Subfolder name under PV (default: infer from checkpoint parent dir).",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="OAG-EVL",
        help="wandb project name when --wandb is enabled.",
    )
    args = parser.parse_args()
    progress_every = int(os.environ.get("EVAL_PROGRESS_EVERY", "10"))

    ckpt_path = _resolve_checkpoint_path(args.checkpoint)
    device = torch.device(f"cuda:{args.cuda}") if args.cuda >= 0 else torch.device("cpu")
    tag = os.path.basename(os.path.normpath(ckpt_path))

    wb_run = None
    if args.wandb:
        if wandb is None:
            print("[warn] wandb is not installed; skip wandb logging.", flush=True)
        else:
            wb_run = wandb.init(
                project=args.wandb_project,
                name=f"PV-seed{args.seed}-{tag}",
                config={
                    "task": "PV",
                    "checkpoint": os.path.abspath(ckpt_path),
                    "seed": int(args.seed),
                    "cuda": int(args.cuda),
                },
                reinit=True,
            )
            wandb.log({"status": 0.0})

    rng = np.random.default_rng(args.seed)
    base = SharedPVBase(args.shared_base_dir)
    split_name = "test"
    n_test_full = _test_paper_count_pv(base, split_name)
    all_rows = np.arange(n_test_full, dtype=np.int64)
    if args.max_test_rows is not None:
        cap = int(args.max_test_rows)
        if cap <= 0:
            raise ValueError("--max_test_rows must be positive")
        all_rows = all_rows[:cap]
    n_test = int(len(all_rows))
    if args.max_test_rows is None:
        print(
            f"[eval][PV] {split_name}: {n_test} papers — evaluating ALL row indices 0..{n_test - 1} "
            f"in order (no random subsampling of the test split).",
            flush=True,
        )
    else:
        print(
            f"[eval][PV] {split_name}: partial eval first {n_test} of {n_test_full} test rows (--max_test_rows).",
            flush=True,
        )

    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(raw, dict) and "model" in raw:
        raise ValueError(
            "This checkpoint looks like a raw state_dict bundle. "
            "Use a full nn.Sequential (GNN + classifier) saved by train_paper_venue_shared.py."
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
                desc=f"THGFM {split_name}",
            ),
            start=1,
        ):
            ed = min(st + args.batch_size, len(all_rows))
            batch_rows = all_rows[st:ed]
            sample = _build_batch_thgfm(
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
            _append_ranking_from_logits(logits, ylabel, test_res)
            if progress_every > 0 and (step_idx % progress_every == 0 or step_idx == total_steps):
                pct = 100.0 * step_idx / max(total_steps, 1)
                elapsed = time.time() - start_ts
                print(
                    f"[progress][PV][thgfm] step={step_idx}/{total_steps} "
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

    out_dir = resolve_eval_output_dir(
        args.save_dir,
        task_subdir="PV",
        model_subdir=args.model_subdir or None,
        checkpoint_path=ckpt_path,
    )
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(out_dir, f"eval_test_seed{args.seed}_{tag}_{ts}.json")

    summary = {
        "task": "PV",
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

    print(f"split={split_name} total_samples={len(test_res)}")
    print(f"NDCG={avg_ndcg:.6f}  MRR={avg_mrr:.6f}")
    print(f"saved_summary={summary_path}")


if __name__ == "__main__":
    main()
