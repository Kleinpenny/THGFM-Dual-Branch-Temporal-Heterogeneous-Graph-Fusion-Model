#!/usr/bin/env python3
"""
Create and run a task pool for OAG evaluation:
- discover all *_lowest_valid_loss checkpoints under model roots
- run PV / PF-L1 / PF-L2 evaluations as applicable
- run each task for seeds 35 and 36
- keep at most 6 concurrent jobs on GPU 0
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

EVALUATOR_DIR = Path(__file__).resolve().parent
OAG_DIR = EVALUATOR_DIR.parent

MODEL_ROOTS = [
    OAG_DIR / "model_save_shared",
]

TASK_SCRIPT_BY_PREFIX = {
    "PV_": "eval_pv_test_ndcg_mrr.py",
    "PF_XLNet_L1_": "eval_pf_l1_test_ndcg_mrr.py",
    "PF_XLNet_L2_": "eval_pf_l2_test_ndcg_mrr.py",
    "PF_L1_": "eval_pf_l1_test_ndcg_mrr.py",
    "PF_L2_": "eval_pf_l2_test_ndcg_mrr.py",
}


def infer_script(name: str) -> str | None:
    for prefix, script in TASK_SCRIPT_BY_PREFIX.items():
        if name.startswith(prefix):
            return script
    return None


def discover_checkpoints() -> list[Path]:
    items: set[Path] = set()
    for root in MODEL_ROOTS:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.name.endswith("_lowest_valid_loss"):
                items.add(p)
    return sorted(items)


def build_tasks(seeds: list[int]) -> list[tuple[str, Path, int, str]]:
    tasks = []
    for ckpt in discover_checkpoints():
        script = infer_script(ckpt.name)
        if script is None:
            continue
        rel = ckpt.relative_to(OAG_DIR)
        for seed in seeds:
            label = f"{rel}::seed{seed}"
            tasks.append((script, ckpt, seed, label))
    return tasks


def run_task(script: str, checkpoint: Path, seed: int, label: str, gpu_id: int, wandb_project: str):
    log_dir = EVALUATOR_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_label = label.replace("/", "__").replace("::", "__")
    log_file = log_dir / f"{safe_label}.log"

    cmd = [
        sys.executable,
        str(EVALUATOR_DIR / script),
        "--checkpoint",
        str(checkpoint),
        "--cuda",
        str(gpu_id),
        "--seed",
        str(seed),
        "--wandb",
        "--wandb_project",
        wandb_project,
    ]

    print(f"[START ] {label}", flush=True)
    with open(log_file, "w", encoding="utf-8") as f:
        rc = subprocess.run(cmd, cwd=str(EVALUATOR_DIR), stdout=f, stderr=subprocess.STDOUT).returncode

    if rc == 0:
        print(f"[  OK  ] {label}", flush=True)
    else:
        print(f"[FAILED] {label} (rc={rc}) -> {log_file}", flush=True)
    return label, rc


def main():
    parser = argparse.ArgumentParser(description="Run full OAG evaluator task pool.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[35, 36])
    parser.add_argument("--max_parallel", type=int, default=6)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--wandb_project", type=str, default="OAG-EVL")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    tasks = build_tasks(args.seeds)
    print(
        f"Task pool created: {len(tasks)} tasks; max_parallel={args.max_parallel}; gpu={args.gpu}; "
        f"seeds={args.seeds}; wandb_project={args.wandb_project}",
        flush=True,
    )
    if tasks:
        for i, (_, _, _, label) in enumerate(tasks, start=1):
            print(f"  [{i:03d}] {label}")
    else:
        print("No tasks discovered. Exiting.")
        return 1

    if args.dry_run:
        return 0

    results: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
        futures = {
            pool.submit(run_task, script, ckpt, seed, label, args.gpu, args.wandb_project): label
            for script, ckpt, seed, label in tasks
        }
        for fut in as_completed(futures):
            label, rc = fut.result()
            results[label] = rc

    failed = [k for k, v in results.items() if v != 0]
    print(f"\nSummary: total={len(results)} ok={len(results) - len(failed)} failed={len(failed)}")
    if failed:
        for x in failed:
            print(f"  FAILED: {x}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
