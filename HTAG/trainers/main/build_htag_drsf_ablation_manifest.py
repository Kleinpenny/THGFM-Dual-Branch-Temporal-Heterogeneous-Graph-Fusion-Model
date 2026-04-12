#!/usr/bin/env python3
"""
JSONL manifest for HTAG Arxiv + DBLP PV DRSF appendix table (THGFM-Paper.tex).

Aligned with OGBN `build_ogbn_drsf_ablation_manifest.py`:
  - r001_s001: SSTA/RTTA + DRSF fusion grid with default per-branch flags (omit overrides).
  - r110_s110: RoTA QK on branches, RTE off on both branches.
  - r111_s111: RoTA QK on branches, RTE on on both branches.

Task names: PV_HTAG_{Arxiv|DBLP}_{suffix}_s{seed} — one metrics jsonl per table cell.

Requires `Arxiv_PV/train_paper_venue_shared.py` and `DBLP_PV/train_paper_venue_shared.py`
to accept --rota_mode / --rtta_rota_* / --ssta_rota_* (same semantics as OGBN PV trainer).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
HTAG_DIR = SCRIPT_DIR.parent.parent


@dataclass(frozen=True)
class HtagAblationSpec:
    suffix: str
    conv: str
    rota_mode: str
    rota_keep_rte: str
    rtta_mode: str | None
    ssta_mode: str | None
    rtta_keep: str | None
    ssta_keep: str | None


@dataclass(frozen=True)
class DatasetRow:
    dataset_id: str
    trainer_relpath: str
    log_prefix: str
    unit_prefix: str
    task_prefix: str


def _datasets() -> dict[str, DatasetRow]:
    return {
        "arxiv": DatasetRow(
            dataset_id="arxiv",
            trainer_relpath="./Arxiv_PV/train_paper_venue_shared_task_pool.py",
            log_prefix="train_thgfm_pv_arxiv",
            unit_prefix="thgfm-pv-arxiv",
            task_prefix="PV_HTAG_Arxiv",
        ),
        "dblp": DatasetRow(
            dataset_id="dblp",
            trainer_relpath="./DBLP_PV/train_paper_venue_shared_task_pool.py",
            log_prefix="train_thgfm_pv_dblp",
            unit_prefix="thgfm-pv-dblp",
            task_prefix="PV_HTAG_DBLP",
        ),
    }


def _r001_s001_block() -> tuple[HtagAblationSpec, ...]:
    """Match OGBN-MAG r001_s001 table: plain convs + default branch overrides (None)."""
    z = (None, None, None, None)
    return (
        HtagAblationSpec("ssta_r001_s001", "ssta", "qk", "1", *z),
        HtagAblationSpec("rtta_r001_s001", "rtta", "qk", "1", *z),
        HtagAblationSpec("sf_r001_s001", "drsf_sf", "qk", "1", *z),
        HtagAblationSpec("lf_r001_s001", "drsf", "qk", "1", *z),
        HtagAblationSpec("ngsf_r001_s001", "drsf_ngsf", "qk", "1", *z),
        HtagAblationSpec("nglf_r001_s001", "drsf_nglf", "qk", "1", *z),
        HtagAblationSpec("tcngsf_r001_s001", "drsf_tcngsf", "qk", "1", *z),
        HtagAblationSpec("tcnglf_r001_s001", "drsf_tc_nglf", "qk", "1", *z),
    )


def _r110_s110_block() -> tuple[HtagAblationSpec, ...]:
    qk0 = ("qk", "qk", "0", "0")
    return (
        HtagAblationSpec("ssta_r110_s110", "ssta_rota", "qk", "0", *qk0),
        HtagAblationSpec("rtta_r110_s110", "rtta_rota", "qk", "0", *qk0),
        HtagAblationSpec("sf_r110_s110", "drsf_sf_rota", "qk", "0", *qk0),
        HtagAblationSpec("lf_r110_s110", "drsf_rota", "qk", "0", *qk0),
        HtagAblationSpec("ngsf_r110_s110", "drsf_ngsf_rota", "qk", "0", *qk0),
        HtagAblationSpec("nglf_r110_s110", "drsf_nglf_rota", "qk", "0", *qk0),
        HtagAblationSpec("tcngsf_r110_s110", "drsf_tcngsf_rota", "qk", "0", *qk0),
        HtagAblationSpec("tcnglf_r110_s110", "drsf_tcnglf_rota", "qk", "0", *qk0),
    )


def _r111_s111_block() -> tuple[HtagAblationSpec, ...]:
    qk1 = ("qk", "qk", "1", "1")
    return (
        HtagAblationSpec("ssta_r111_s111", "ssta_rota", "qk", "1", *qk1),
        HtagAblationSpec("rtta_r111_s111", "rtta_rota", "qk", "1", *qk1),
        HtagAblationSpec("sf_r111_s111", "drsf_sf_rota", "qk", "1", *qk1),
        HtagAblationSpec("lf_r111_s111", "drsf_rota", "qk", "1", *qk1),
        HtagAblationSpec("ngsf_r111_s111", "drsf_ngsf_rota", "qk", "1", *qk1),
        HtagAblationSpec("nglf_r111_s111", "drsf_nglf_rota", "qk", "1", *qk1),
        HtagAblationSpec("tcngsf_r111_s111", "drsf_tcngsf_rota", "qk", "1", *qk1),
        HtagAblationSpec("tcnglf_r111_s111", "drsf_tcnglf_rota", "qk", "1", *qk1),
    )


def all_specs() -> tuple[HtagAblationSpec, ...]:
    return _r001_s001_block() + _r110_s110_block() + _r111_s111_block()


def _sanitize_unit_token(s: str) -> str:
    return s.replace("_", "-")


def _wandb_argv(
    *,
    use: bool,
    project: str,
    entity: str,
    group: str,
    mode: str,
    run_name: str,
) -> list[str]:
    if not use:
        return []
    argv = [
        "--use_wandb",
        "--wandb_project",
        project,
        "--wandb_mode",
        mode,
        "--wandb_run_name",
        run_name,
    ]
    if entity:
        argv.extend(["--wandb_entity", entity])
    if group:
        argv.extend(["--wandb_group", group])
    return argv


def command_for_spec(
    d: DatasetRow,
    spec: HtagAblationSpec,
    seed: int,
    *,
    model_dir: str,
    n_pool: int,
    cuda: str,
    wandb_use: bool,
    wandb_project: str,
    wandb_entity: str,
    wandb_group: str,
    wandb_mode: str,
) -> list[str]:
    task_name = f"{d.task_prefix}_{spec.suffix}_s{seed}"
    cmd: list[str] = [
        "python",
        d.trainer_relpath,
        "--shared_base_dir",
        str(
            HTAG_DIR / ("Arxiv_PV/data/shared_pv_base_HTAG_arxiv" if d.dataset_id == "arxiv" else "DBLP_PV/data/shared_pv_base_HTAG_dblp")
        ),
        "--model_dir",
        model_dir,
        "--task_name",
        task_name,
        "--conv_name",
        spec.conv,
        "--cuda",
        str(cuda),
        "--n_pool",
        str(n_pool),
        "--seed",
        str(seed),
        "--deterministic",
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
    cmd.extend(
        _wandb_argv(
            use=wandb_use,
            project=wandb_project,
            entity=wandb_entity,
            group=wandb_group or f"htag-drsf-s{seed}",
            mode=wandb_mode,
            run_name=task_name,
        )
    )
    return cmd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build HTAG DRSF appendix task-pool manifest (Arxiv + DBLP).")
    p.add_argument(
        "--output",
        default=str(SCRIPT_DIR / "manifests" / "task_pool_htag_drsf_appendix_s43.jsonl"),
        help="Output JSONL path.",
    )
    p.add_argument("--datasets", nargs="+", choices=["arxiv", "dblp"], default=["arxiv", "dblp"])
    p.add_argument("--seeds", nargs="+", type=int, default=[43])
    p.add_argument("--model-dir", default="./model_save_shared")
    p.add_argument("--n-pool", type=int, default=8)
    p.add_argument("--cpu-threads", type=int, default=1)
    p.add_argument("--cuda", default="0", help="Placeholder CUDA id; scheduler overwrites when using --gpu-ids.")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="THGFM-HTAG-PV")
    p.add_argument("--wandb-entity", default="")
    p.add_argument("--wandb-group", default="")
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    p.add_argument(
        "--unit-name-suffix",
        default="",
        help="Appended to systemd unit_name to avoid collisions with older units.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ds_map = _datasets()
    specs = all_specs()
    tasks: list[dict] = []
    suf = str(args.unit_name_suffix).strip()

    for seed in args.seeds:
        for did in args.datasets:
            d = ds_map[did]
            for spec in specs:
                base_unit = f"{d.unit_prefix}-{_sanitize_unit_token(spec.suffix)}-s{seed}"
                unit_name = f"{base_unit}-{suf}" if suf else base_unit
                log_name = f"./logs/task_pool/{d.log_prefix}_{spec.suffix}_s{seed}.log"
                tasks.append(
                    {
                        "unit_name": unit_name,
                        "workdir": str(HTAG_DIR),
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
                            d,
                            spec,
                            seed,
                            model_dir=args.model_dir,
                            n_pool=args.n_pool,
                            cuda=args.cuda,
                            wandb_use=bool(args.wandb),
                            wandb_project=args.wandb_project,
                            wandb_entity=args.wandb_entity,
                            wandb_group=args.wandb_group,
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
