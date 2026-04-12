#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_ROOT = PROJECT_DIR.parent / "data"

DEFAULT_CONVS = [
    "rtta",
    "rtta_rte0",
    "rtta_rota",
    "ssta",
    "ssta_rte0",
    "ssta_rota",
    "drsf",
    "drsf_rota",
    "drsf_sf",
    "drsf_sf_rota",
    "drsf_cglf",
    "drsf_cglf_rota",
    "drsf_cgsf",
    "drsf_cgsf_rota",
    "drsf_nglf",
    "drsf_nglf_rota",
    "drsf_ngsf",
    "drsf_ngsf_rota",
    "drsf_tcnglf",
    "drsf_tc_nglf",
    "drsftc_nglf",
    "drsf_tcnglf_rota",
    "drsf_tc_nglf_rota",
    "drsftc_nglf_rota",
    "drsf_tcngsf",
    "drsf_tc_ngsf",
    "drsftc_ngsf",
    "drsf_tcngsf_rota",
    "drsf_tc_ngsf_rota",
    "drsftc_ngsf_rota",
    "drsf_tanglf",
    "drsf_tanglf_rota",
    "drsf_tangsf",
    "drsf_tangsf_rota",
    "drsf_gate",
    "drsf_gate_rota",
    "drsftc_ndg",
    "drsftc_ndg_rota",
    "drsftc_glf",
    "drsf_tcglf",
    "drsf_tc_glf",
    "drsftc_glf_rota",
    "drsf_tcglf_rota",
    "drsf_tc_glf_rota",
    "gcn",
    "gat",
]


def _branch_code(conv_name: str) -> str:
    """
    Branch code format: q k rte (e.g. 001 means RTE only, 110 means RoTA-QK only).
    """
    name = str(conv_name).lower()
    has_rota = ("rota" in name) or ("-ng" in name) or ("_ng" in name) or ("tcng" in name) or ("tang" in name)
    has_rte = "rte0" not in name
    q = "1" if has_rota else "0"
    k = "1" if has_rota else "0"
    rte = "1" if has_rte else "0"
    return f"{q}{k}{rte}"


def _conv_variant_tag(conv: str) -> str:
    name = str(conv).lower()
    if name.startswith("rtta"):
        return f"r{_branch_code(name)}"
    if name.startswith("ssta"):
        return f"s{_branch_code(name)}"
    return f"r{_branch_code(name)}_s{_branch_code(name)}"


def _sanitize_for_unit(token: str) -> str:
    return token.replace("_", "-")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate OGBN PV task-pool manifest (JSONL).")
    p.add_argument("--output", default="-", help="Output JSONL path, or '-' for stdout.")
    p.add_argument("--seeds", nargs="+", type=int, default=[43], help="Seeds for each conv variant.")
    p.add_argument(
        "--conv-names",
        nargs="+",
        default=DEFAULT_CONVS,
        help="Conv variants to include (default: all convs declared in trainer choices).",
    )
    p.add_argument(
        "--shared-base-dir",
        default=str(DATA_ROOT / "shared_pv_base_ogbn_mag_graph"),
        help="Shared base path for OGBN PV.",
    )
    p.add_argument("--model-dir", default="./model_save_shared", help="Model save root.")
    p.add_argument("--n-pool", type=int, default=8, help="Per-task sampling worker count.")
    p.add_argument("--cpu-threads", type=int, default=1, help="Per-task BLAS/OpenMP thread cap.")
    p.add_argument("--cuda", default="0", help="Default CUDA id in manifest.")
    p.add_argument("--wandb-project", default="THGFM-OGBN", help="--wandb_project for tasks.")
    p.add_argument("--wandb-entity", default="", help="Optional --wandb_entity.")
    p.add_argument("--wandb-mode", default="", help="Optional --wandb_mode.")
    return p.parse_args()


def command_for_task(args: argparse.Namespace, conv: str, seed: int) -> list[str]:
    variant_tag = _conv_variant_tag(conv)
    task_name = f"PV_ogbn_mag_{conv}_{variant_tag}_s{seed}"
    cmd = [
        "python",
        "./trainers/main/train_paper_venue_shared_task_pool.py",
        "--shared_base_dir",
        args.shared_base_dir,
        "--model_dir",
        args.model_dir,
        "--task_name",
        task_name,
        "--conv_name",
        conv,
        "--cuda",
        str(args.cuda),
        "--n_pool",
        str(args.n_pool),
        "--seed",
        str(seed),
        "--deterministic",
        "--use_wandb",
        "--wandb_project",
        args.wandb_project,
        "--wandb_group",
        "pv-ogbn-task-pool",
        "--wandb_run_name",
        f"{task_name}_shared",
    ]
    if args.wandb_entity:
        cmd.extend(["--wandb_entity", args.wandb_entity])
    if args.wandb_mode:
        cmd.extend(["--wandb_mode", args.wandb_mode])
    return cmd


def build_tasks(args: argparse.Namespace) -> list[dict]:
    tasks: list[dict] = []
    for seed in args.seeds:
        for conv in args.conv_names:
            variant_tag = _conv_variant_tag(conv)
            unit_name = f"pv-ogbn-{_sanitize_for_unit(conv)}-{_sanitize_for_unit(variant_tag)}-s{seed}"
            tasks.append(
                {
                    "unit_name": unit_name,
                    "workdir": str(PROJECT_DIR),
                    "log_file": f"./logs/train_paper_venue_ogbn_shared_{conv}_{variant_tag}_s{seed}.log",
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
                    "command": command_for_task(args, conv, seed),
                }
            )
    return tasks


def write_manifest(path: Path, tasks: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=True))
            f.write("\n")


def main() -> int:
    args = parse_args()
    tasks = build_tasks(args)

    if args.output == "-":
        for t in tasks:
            print(json.dumps(t, ensure_ascii=True))
        return 0

    out = Path(args.output).expanduser().resolve()
    write_manifest(out, tasks)
    print(f"wrote {len(tasks)} tasks to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
