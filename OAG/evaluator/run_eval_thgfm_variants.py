#!/usr/bin/env python3
"""
Parallel evaluation runner for THGFM variant checkpoints.
Auto-discovers PV / PF-L1 / PF-L2 *_lowest_valid_loss checkpoints under per-conv subdirectories.
"""

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

EVALUATOR_DIR = Path(__file__).parent
OAG_DIR = EVALUATOR_DIR.parent
MODEL_BASE = OAG_DIR / "model_save_shared"

VARIANT_DIRS = [
    "ssta",
    "ssta_rota",
    "ssta_rte0",
    "rtta",
    "rtta_rota",
    "rtta_rte0",
    "drsf_sf",
    "drsf_sf_rota",
    "drsf_glf",
    "drsf_ngsf",
    "drsf_ngsf_rota",
    "drsf_nglf",
    "drsf_nglf_rota",
    "drsf_tcngsf",
    "drsf_tcngsf_rota",
    "drsf_tcnglf",
    "drsf_tcnglf_rota",
]

TASK_PREFIX_TO_SCRIPT = {
    "PV_": "eval_pv_test_ndcg_mrr.py",
    "PF_XLNet_L1_": "eval_pf_l1_test_ndcg_mrr.py",
    "PF_XLNet_L2_": "eval_pf_l2_test_ndcg_mrr.py",
}

MAX_PARALLEL = 4
GPU_ID = 0


def discover_tasks():
    tasks = []
    for variant_dir in VARIANT_DIRS:
        dir_path = MODEL_BASE / variant_dir
        if not dir_path.is_dir():
            print(f"[SKIP] Directory not found: {dir_path}", flush=True)
            continue
        for entry in sorted(dir_path.iterdir()):
            if not entry.name.endswith("_lowest_valid_loss"):
                continue
            for prefix, script in TASK_PREFIX_TO_SCRIPT.items():
                if entry.name.startswith(prefix):
                    label = f"{variant_dir}/{entry.name}"
                    tasks.append((script, entry, label))
                    break
    return tasks


def run_task(script: str, checkpoint: Path, label: str):
    log_dir = EVALUATOR_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_label = label.replace("/", "__")
    log_file = log_dir / f"{safe_label}.log"

    cmd = [
        sys.executable,
        str(EVALUATOR_DIR / script),
        "--checkpoint",
        str(checkpoint),
        "--cuda",
        str(GPU_ID),
    ]

    print(f"[START ] {label}", flush=True)
    with open(log_file, "w") as f:
        result = subprocess.run(cmd, cwd=str(EVALUATOR_DIR), stdout=f, stderr=subprocess.STDOUT)

    if result.returncode == 0:
        print(f"[  OK  ] {label}", flush=True)
    else:
        print(f"[FAILED] {label} (rc={result.returncode})  ->  {log_file}", flush=True)

    return label, result.returncode


def main():
    tasks = discover_tasks()
    print(f"Discovered {len(tasks)} evaluation tasks across {len(VARIANT_DIRS)} variant directories.")
    print(f"Running max {MAX_PARALLEL} concurrent jobs on GPU {GPU_ID}\n", flush=True)

    results = {}
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
        futures = {
            executor.submit(run_task, script, ckpt, label): label
            for script, ckpt, label in tasks
        }
        for future in as_completed(futures):
            label, rc = future.result()
            results[label] = rc

    print("\n=== Summary ===")
    ok = [l for l, rc in results.items() if rc == 0]
    failed = [l for l, rc in results.items() if rc != 0]
    print(f"  OK:     {len(ok)}")
    print(f"  FAILED: {len(failed)}")
    if failed:
        print("\nFailed tasks:")
        for l in sorted(failed):
            print(f"  {l}")
        sys.exit(1)
    print("\nAll tasks completed successfully.")


if __name__ == "__main__":
    main()
