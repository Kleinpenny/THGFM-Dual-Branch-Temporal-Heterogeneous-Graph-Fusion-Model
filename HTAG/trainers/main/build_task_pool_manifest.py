#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
HTAG_DIR = SCRIPT_DIR.parent.parent


@dataclass(frozen=True)
class DatasetConfig:
    dataset_id: str
    trainer: str
    shared_base_dir: str
    task_prefix: str
    log_prefix: str
    unit_prefix: str


def default_datasets() -> dict[str, DatasetConfig]:
    return {
        "dblp": DatasetConfig(
            dataset_id="dblp",
            trainer="./DBLP_PV/train_paper_venue_shared_task_pool.py",
            shared_base_dir=str(HTAG_DIR / "DBLP_PV" / "data" / "shared_pv_base_HTAG_dblp"),
            task_prefix="PV_HTAG_DBLP",
            log_prefix="train_thgfm_pv_dblp",
            unit_prefix="thgfm-pv-dblp",
        ),
        "arxiv": DatasetConfig(
            dataset_id="arxiv",
            trainer="./Arxiv_PV/train_paper_venue_shared_task_pool.py",
            shared_base_dir=str(HTAG_DIR / "Arxiv_PV" / "data" / "shared_pv_base_HTAG_arxiv"),
            task_prefix="PV_HTAG_Arxiv",
            log_prefix="train_thgfm_pv_arxiv",
            unit_prefix="thgfm-pv-arxiv",
        ),
    }


# Keep this list aligned with conv names used in HTAG experiments.
DEFAULT_CONVS = (
    "rtta",
    "ssta",
    "drsf",
    "drsf_tcngsf",
    "drsf_tcnglf",
)


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
    # DRSF family and table aliases default to dual-branch tag.
    return f"r{_branch_code(name)}_s{_branch_code(name)}"


def _sanitize_for_unit(token: str) -> str:
    return token.replace("_", "-")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate JSONL manifest for HTAG conv-ablation task pool.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["dblp", "arxiv"],
        default=["dblp", "arxiv"],
        help="Datasets included in manifest.",
    )
    parser.add_argument(
        "--convs",
        nargs="+",
        default=list(DEFAULT_CONVS),
        help="Conv variants to expand.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[43, 44, 45],
        help="Seeds to expand for every conv/dataset combination.",
    )
    parser.add_argument(
        "--output",
        default="-",
        help="Output JSONL path, or '-' for stdout.",
    )
    parser.add_argument(
        "--model-dir",
        default="./model_save_shared",
        help="Value passed to --model_dir in trainer commands.",
    )
    parser.add_argument(
        "--cuda",
        default="0",
        help="Placeholder --cuda in manifest; run_systemd_task_pool.py overwrites it when using --gpu-ids.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Append --use_wandb and related flags to trainer commands.",
    )
    parser.add_argument("--wandb-project", default="THGFM-HTAG-PV", help="Passed as --wandb_project.")
    parser.add_argument("--wandb-entity", default="", help="Passed as --wandb_entity when non-empty.")
    parser.add_argument("--wandb-group", default="", help="Passed as --wandb_group when non-empty.")
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument(
        "--wandb-tags",
        nargs="*",
        default=[],
        help="Optional tags passed through to trainers.",
    )
    parser.add_argument(
        "--unit-name-suffix",
        default="",
        help="Appended to systemd unit_name only (avoids collisions with past systemd-run units).",
    )
    parser.add_argument(
        "--n-pool",
        type=int,
        default=8,
        help="Per-task sampling worker count passed to trainer --n_pool.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=1,
        help="Per-task CPU thread cap for OMP/MKL/OPENBLAS/NUMEXPR/Torch.",
    )
    parser.add_argument(
        "--write-defaults",
        action="store_true",
        help="Write default manifest into trainers/main/manifests/task_pool_htag_conv_ablation.jsonl.",
    )
    return parser.parse_args()


def command_for_task(
    dataset: DatasetConfig,
    *,
    conv: str,
    seed: int,
    n_pool: int,
    model_dir: str,
    cuda: str,
    wandb_argv: list[str],
) -> list[str]:
    variant_tag = _conv_variant_tag(conv)
    task_name = f"{dataset.task_prefix}_{conv}_{variant_tag}_s{seed}"
    cmd: list[str] = [
        "python",
        dataset.trainer,
        "--shared_base_dir",
        dataset.shared_base_dir,
        "--model_dir",
        model_dir,
        "--task_name",
        task_name,
        "--conv_name",
        conv,
        "--cuda",
        str(cuda),
        "--n_pool",
        str(n_pool),
        "--seed",
        str(seed),
        "--deterministic",
    ]
    cmd.extend(wandb_argv)
    return cmd


def _wandb_argv_for_task(
    *,
    use_wandb: bool,
    wandb_project: str,
    wandb_entity: str,
    wandb_group: str,
    wandb_mode: str,
    wandb_tags: list[str],
    task_name: str,
) -> list[str]:
    if not use_wandb:
        return []
    argv = [
        "--use_wandb",
        "--wandb_project",
        wandb_project,
        "--wandb_mode",
        wandb_mode,
        "--wandb_run_name",
        task_name,
    ]
    if wandb_entity:
        argv.extend(["--wandb_entity", wandb_entity])
    if wandb_group:
        argv.extend(["--wandb_group", wandb_group])
    if wandb_tags:
        argv.append("--wandb_tags")
        argv.extend(wandb_tags)
    return argv


def build_tasks(
    *,
    dataset_ids: list[str],
    convs: list[str],
    seeds: list[int],
    n_pool: int,
    cpu_threads: int,
    model_dir: str,
    cuda_placeholder: str,
    wandb_use: bool,
    wandb_project: str,
    wandb_entity: str,
    wandb_group: str,
    wandb_mode: str,
    wandb_tags: list[str],
    unit_name_suffix: str,
) -> list[dict[str, object]]:
    dset_map = default_datasets()
    tasks: list[dict[str, object]] = []
    for dataset_id in dataset_ids:
        dataset = dset_map[dataset_id]
        for conv in convs:
            for seed in seeds:
                variant_tag = _conv_variant_tag(conv)
                task_name = f"{dataset.task_prefix}_{conv}_{variant_tag}_s{seed}"
                unit_base = f"{dataset.unit_prefix}-{_sanitize_for_unit(conv)}-{_sanitize_for_unit(variant_tag)}-s{seed}"
                suf = str(unit_name_suffix).strip()
                unit_name = f"{unit_base}-{suf}" if suf else unit_base
                log_name = f"{dataset.log_prefix}_{conv}_{variant_tag}_s{seed}.log"
                w_argv = _wandb_argv_for_task(
                    use_wandb=wandb_use,
                    wandb_project=wandb_project,
                    wandb_entity=wandb_entity,
                    wandb_group=wandb_group,
                    wandb_mode=wandb_mode,
                    wandb_tags=wandb_tags,
                    task_name=task_name,
                )
                tasks.append(
                    {
                        "unit_name": unit_name,
                        "workdir": str(HTAG_DIR),
                        "log_file": f"./logs/task_pool/{log_name}",
                        "setenv": {
                            "OMP_NUM_THREADS": str(cpu_threads),
                            "MKL_NUM_THREADS": str(cpu_threads),
                            "OPENBLAS_NUM_THREADS": str(cpu_threads),
                            "NUMEXPR_NUM_THREADS": str(cpu_threads),
                            "BLIS_NUM_THREADS": str(cpu_threads),
                            "VECLIB_MAXIMUM_THREADS": str(cpu_threads),
                            "TORCH_NUM_THREADS": str(cpu_threads),
                            "OMP_WAIT_POLICY": "PASSIVE",
                            "KMP_BLOCKTIME": "0",
                        },
                        "command": command_for_task(
                            dataset,
                            conv=conv,
                            seed=seed,
                            n_pool=n_pool,
                            model_dir=model_dir,
                            cuda=cuda_placeholder,
                            wandb_argv=w_argv,
                        ),
                    }
                )
    return tasks


def write_manifest(path: Path, tasks: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=True))
            f.write("\n")


def main() -> int:
    args = parse_args()

    tasks = build_tasks(
        dataset_ids=args.datasets,
        convs=args.convs,
        seeds=args.seeds,
        n_pool=args.n_pool,
        cpu_threads=args.cpu_threads,
        model_dir=args.model_dir,
        cuda_placeholder=str(args.cuda),
        wandb_use=bool(args.wandb),
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        wandb_mode=args.wandb_mode,
        wandb_tags=list(args.wandb_tags),
        unit_name_suffix=args.unit_name_suffix,
    )

    if args.write_defaults:
        default_path = SCRIPT_DIR / "manifests" / "task_pool_htag_conv_ablation.jsonl"
        write_manifest(default_path, tasks)
        print(f"wrote {len(tasks)} tasks to {default_path}")
        return 0

    if args.output == "-":
        for task in tasks:
            print(json.dumps(task, ensure_ascii=True))
        return 0

    output_path = Path(args.output).expanduser().resolve()
    write_manifest(output_path, tasks)
    print(f"wrote {len(tasks)} tasks to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
