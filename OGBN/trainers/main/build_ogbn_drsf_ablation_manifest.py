#!/usr/bin/env python3
"""
JSONL manifest for OGBN-MAG PV DRSF table (THGFM-Paper appendix Other Datasets).

Covers:
  - r001_s001: missing LF / NGLF / TC-NGLF (plain convs, default RoTA/RTE branch flags).
  - r110_s110: full 8 fusion rows (RoTA QK, RTE off on branches) — aligned with OAG DRSF_ABLATIONS_R110 + SSTA/RTTA.
  - r111_s111: full 8 fusion rows (RoTA QK, RTE on) — aligned with OAG DRSF_ABLATIONS + SSTA/RTTA.

Task names: PV_ogbn_mag_{suffix}_s{seed} (distinct metrics files per cell).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_ROOT = PROJECT_DIR / "data"


@dataclass(frozen=True)
class OgbnAblationSpec:
    suffix: str
    conv: str
    rota_mode: str
    rota_keep_rte: str
    # None = omit CLI flag (use trainer defaults).
    rtta_mode: str | None
    ssta_mode: str | None
    rtta_keep: str | None
    ssta_keep: str | None


def _r001_fill() -> tuple[OgbnAblationSpec, ...]:
    """Plain convs; match already-logged r001_s001 SF/NGSF/TC-NGSF runs (default branch flags)."""
    return (
        OgbnAblationSpec("lf_r001_s001", "drsf_cglf", "qk", "1", None, None, None, None),
        OgbnAblationSpec("nglf_r001_s001", "drsf_nglf", "qk", "1", None, None, None, None),
        OgbnAblationSpec("tcnglf_r001_s001", "drsf_tc_nglf", "qk", "1", None, None, None, None),
    )


def _r110_block() -> tuple[OgbnAblationSpec, ...]:
    qk0 = ("qk", "qk", "0", "0")
    return (
        OgbnAblationSpec("ssta_r110_s110", "ssta_rota", "qk", "0", *qk0),
        OgbnAblationSpec("rtta_r110_s110", "rtta_rota", "qk", "0", *qk0),
        OgbnAblationSpec("sf_r110_s110", "drsf_sf_rota", "qk", "0", *qk0),
        OgbnAblationSpec("lf_r110_s110", "drsf_rota", "qk", "0", *qk0),
        OgbnAblationSpec("ngsf_r110_s110", "drsf_ngsf_rota", "qk", "0", *qk0),
        OgbnAblationSpec("nglf_r110_s110", "drsf_nglf_rota", "qk", "0", *qk0),
        OgbnAblationSpec("tcngsf_r110_s110", "drsf_tcngsf_rota", "qk", "0", *qk0),
        OgbnAblationSpec("tcnglf_r110_s110", "drsf_tcnglf_rota", "qk", "0", *qk0),
    )


def _r111_block() -> tuple[OgbnAblationSpec, ...]:
    qk1 = ("qk", "qk", "1", "1")
    return (
        OgbnAblationSpec("ssta_r111_s111", "ssta_rota", "qk", "1", *qk1),
        OgbnAblationSpec("rtta_r111_s111", "rtta_rota", "qk", "1", *qk1),
        OgbnAblationSpec("sf_r111_s111", "drsf_sf_rota", "qk", "1", *qk1),
        OgbnAblationSpec("lf_r111_s111", "drsf_rota", "qk", "1", *qk1),
        OgbnAblationSpec("ngsf_r111_s111", "drsf_ngsf_rota", "qk", "1", *qk1),
        OgbnAblationSpec("nglf_r111_s111", "drsf_nglf_rota", "qk", "1", *qk1),
        OgbnAblationSpec("tcngsf_r111_s111", "drsf_tcngsf_rota", "qk", "1", *qk1),
        OgbnAblationSpec("tcnglf_r111_s111", "drsf_tcnglf_rota", "qk", "1", *qk1),
    )


def all_specs() -> tuple[OgbnAblationSpec, ...]:
    return _r001_fill() + _r110_block() + _r111_block()


def _sanitize_unit_token(s: str) -> str:
    return s.replace("_", "-")


def command_for_spec(
    spec: OgbnAblationSpec,
    seed: int,
    *,
    shared_base_dir: str,
    model_dir: str,
    n_pool: int,
    wandb_project: str,
    wandb_entity: str,
    wandb_mode: str,
) -> list[str]:
    task_name = f"PV_ogbn_mag_{spec.suffix}_s{seed}"
    cmd: list[str] = [
        "python",
        "./trainers/main/train_paper_venue_shared_task_pool.py",
        "--shared_base_dir",
        shared_base_dir,
        "--model_dir",
        model_dir,
        "--task_name",
        task_name,
        "--conv_name",
        spec.conv,
        "--cuda",
        "0",
        "--n_pool",
        str(n_pool),
        "--seed",
        str(seed),
        "--deterministic",
        "--use_wandb",
        "--wandb_project",
        wandb_project,
        "--wandb_group",
        "pv-ogbn-drsf-ablation",
        "--wandb_run_name",
        f"{task_name}_shared",
        "--rota_mode",
        spec.rota_mode,
        "--rota_keep_rte",
        spec.rota_keep_rte,
    ]
    if spec.rtta_mode is not None:
        cmd.extend(["--rtta_rota_mode", spec.rtta_mode])
    if spec.ssta_mode is not None:
        cmd.extend(["--ssta_rota_mode", spec.ssta_mode])
    if spec.rtta_keep is not None:
        cmd.extend(["--rtta_rota_keep_rte", spec.rtta_keep])
    if spec.ssta_keep is not None:
        cmd.extend(["--ssta_rota_keep_rte", spec.ssta_keep])
    if wandb_entity:
        cmd.extend(["--wandb_entity", wandb_entity])
    if wandb_mode:
        cmd.extend(["--wandb_mode", wandb_mode])
    return cmd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build OGBN DRSF ablation task-pool manifest (19 tasks, seed 43).")
    p.add_argument(
        "--output",
        default=str(SCRIPT_DIR / "manifests" / "task_pool_ogbn_drsf_all_stages_s43.jsonl"),
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
    specs = all_specs()
    tasks: list[dict] = []
    for seed in args.seeds:
        for spec in specs:
            unit_name = f"pv-ogbn-{_sanitize_unit_token(spec.suffix)}-s{seed}"
            log_name = f"./logs/train_paper_venue_ogbn_drsf_{spec.suffix}_s{seed}.log"
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
