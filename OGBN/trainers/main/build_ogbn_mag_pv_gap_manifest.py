#!/usr/bin/env python3
"""
JSONL manifest for OGBN-MAG(PV) cells marked '--' in THGFM-Paper Table (appendix, Other Datasets).

As of the paper draft, this covers:
  - r001_s001 / NGSF: validation column missing (re-run records full metrics jsonl).
  - r110_s110 / LF, NGLF, TC-NGSF: incomplete prior runs (no best_test H/L in metrics).

Reuses the same CLI layout as build_ogbn_drsf_ablation_manifest.py (wandb on by default in commands).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_ogbn_drsf_ablation_manifest import (
    DATA_ROOT,
    OgbnAblationSpec,
    PROJECT_DIR,
    _sanitize_unit_token,
    command_for_spec,
)


def gap_specs() -> tuple[OgbnAblationSpec, ...]:
    qk0 = ("qk", "qk", "0", "0")
    return (
        OgbnAblationSpec("ngsf_r001_s001", "drsf_ngsf", "qk", "1", None, None, None, None),
        OgbnAblationSpec("lf_r110_s110", "drsf_rota", "qk", "0", *qk0),
        OgbnAblationSpec("nglf_r110_s110", "drsf_nglf_rota", "qk", "0", *qk0),
        OgbnAblationSpec("tcngsf_r110_s110", "drsf_tcngsf_rota", "qk", "0", *qk0),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build OGBN-MAG(PV) gap-fill task manifest (4 tasks, seed 43 by default).")
    p.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "manifests" / "task_pool_ogbn_mag_pv_gaps_s43.jsonl"),
        help="Output JSONL path.",
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[43])
    p.add_argument(
        "--shared-base-dir",
        default=str(DATA_ROOT / "shared_pv_base_ogbn_mag_graph"),
        help="OGBN shared PV graph base.",
    )
    p.add_argument("--model-dir", default="./model_save_shared")
    p.add_argument("--n-pool", type=int, default=8)
    p.add_argument("--cpu-threads", type=int, default=1)
    p.add_argument("--wandb-project", default="THGFM-OGBN")
    p.add_argument("--wandb-entity", default="")
    p.add_argument("--wandb-mode", default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    specs = gap_specs()
    tasks: list[dict] = []
    for seed in args.seeds:
        for spec in specs:
            unit_name = f"pv-ogbn-{_sanitize_unit_token(spec.suffix)}-s{seed}"
            log_name = f"./logs/train_paper_venue_ogbn_gap_{spec.suffix}_s{seed}.log"
            tasks.append(
                {
                    "unit_name": unit_name,
                    "workdir": str(PROJECT_DIR),
                    "log_file": log_name,
                    "setenv": {
                        "OMP_NUM_THREADS": str(args.cpu_threads),
                        "MKL_NUM_THREADS": str(args.cpu_threads),
                        "OPENBLAS_NUM_THREADS": str(args.cpu_threads),
                        "NUMEXPR_NUM_THREADS": str(args.cpu_threads),
                        "BLIS_NUM_THREADS": str(args.cpu_threads),
                        "VECLIB_MAXIMUM_THREADS": str(args.cpu_threads),
                        "TORCH_NUM_THREADS": str(args.cpu_threads),
                        "OMP_WAIT_POLICY": "PASSIVE",
                        "KMP_BLOCKTIME": "0",
                    },
                    "command": command_for_spec(
                        spec,
                        seed,
                        shared_base_dir=args.shared_base_dir,
                        model_dir=args.model_dir,
                        n_pool=args.n_pool,
                        wandb_project=args.wandb_project,
                        wandb_entity=args.wandb_entity,
                        wandb_mode=args.wandb_mode,
                    ),
                }
            )

    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=True) + "\n")
    print(f"wrote {len(tasks)} tasks to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
