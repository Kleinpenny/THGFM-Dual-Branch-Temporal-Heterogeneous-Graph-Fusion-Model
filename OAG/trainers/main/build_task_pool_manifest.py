#!/usr/bin/env python3
import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_ROOT = PROJECT_DIR.parent / "data"


@dataclass(frozen=True)
class AblationSpec:
    suffix: str
    conv: str
    rota_mode: str
    rota_keep_rte: str
    rtta_mode: str
    ssta_mode: str
    rtta_keep: str
    ssta_keep: str


@dataclass(frozen=True)
class FamilyConfig:
    family_id: str
    trainer: str
    shared_base_dir: str
    model_dir: str
    unit_prefix: str
    log_prefix: str
    task_prefix: str
    wandb_group: str
    cuda: str
    extra_args: tuple[str, ...]
    ablations: tuple[AblationSpec, ...]


DRSF_ABLATIONS = (
    # Target DRSF fusion variants for ablation:
    # SF, LF, NGSF, NGLF, TC-NGSF, TC-NGLF.
    #
    # Config A (RoTA + RTE): r111_s111
    AblationSpec("lf_r111_s111", "drsf_rota", "qk", "1", "qk", "qk", "1", "1"),
    AblationSpec("sf_r111_s111", "drsf_sf_rota", "qk", "1", "qk", "qk", "1", "1"),
    AblationSpec("ngsf_r111_s111", "drsf_ngsf_rota", "qk", "1", "qk", "qk", "1", "1"),
    AblationSpec("nglf_r111_s111", "drsf_nglf_rota", "qk", "1", "qk", "qk", "1", "1"),
    AblationSpec("tcngsf_r111_s111", "drsf_tcngsf_rota", "qk", "1", "qk", "qk", "1", "1"),
    AblationSpec("tcnglf_r111_s111", "drsf_tcnglf_rota", "qk", "1", "qk", "qk", "1", "1"),
)

DRSF_ABLATIONS_R001 = (
    # Config B (RoTA + RTE): r001_s111
    AblationSpec("lf_r001_s111", "drsf_rota", "none", "1", "qk", "qk", "1", "1"),
    AblationSpec("sf_r001_s111", "drsf_sf_rota", "none", "1", "qk", "qk", "1", "1"),
    AblationSpec("ngsf_r001_s111", "drsf_ngsf_rota", "none", "1", "qk", "qk", "1", "1"),
    AblationSpec("nglf_r001_s111", "drsf_nglf_rota", "none", "1", "qk", "qk", "1", "1"),
    AblationSpec("tcngsf_r001_s111", "drsf_tcngsf_rota", "none", "1", "qk", "qk", "1", "1"),
    AblationSpec("tcnglf_r001_s111", "drsf_tcnglf_rota", "none", "1", "qk", "qk", "1", "1"),
)

DRSF_ABLATIONS_R110 = (
    # Config C (RoTA only, no RTE): r110_s110
    AblationSpec("lf_r110_s110", "drsf_rota", "qk", "0", "qk", "qk", "0", "0"),
    AblationSpec("sf_r110_s110", "drsf_sf_rota", "qk", "0", "qk", "qk", "0", "0"),
    AblationSpec("ngsf_r110_s110", "drsf_ngsf_rota", "qk", "0", "qk", "qk", "0", "0"),
    AblationSpec("nglf_r110_s110", "drsf_nglf_rota", "qk", "0", "qk", "qk", "0", "0"),
    AblationSpec("tcngsf_r110_s110", "drsf_tcngsf_rota", "qk", "0", "qk", "qk", "0", "0"),
    AblationSpec("tcnglf_r110_s110", "drsf_tcnglf_rota", "qk", "0", "qk", "qk", "0", "0"),
)

DRSF_ABLATIONS_FULL = DRSF_ABLATIONS + DRSF_ABLATIONS_R001 + DRSF_ABLATIONS_R110

DRSF_ABLATIONS_TA = (
    # TA-NGLF and TA-NGSF only, both RoTA configs.
    AblationSpec("tanglf_r111_s111", "drsf_tanglf_rota", "qk", "1", "qk", "qk", "1", "1"),
    AblationSpec("tangsf_r111_s111", "drsf_tangsf_rota", "qk", "1", "qk", "qk", "1", "1"),
    AblationSpec("tanglf_r001_s111", "drsf_tanglf_rota", "none", "1", "qk", "qk", "1", "1"),
    AblationSpec("tangsf_r001_s111", "drsf_tangsf_rota", "none", "1", "qk", "qk", "1", "1"),
)

# Missing cells in docs/THGFM-Paper.tex Table (appendix-detailed-ablation-drsf-r111-s111):
# - PF-L1: NGSF/NGLF/TC-NGSF/TC-NGLF
# - PF-L2: SF/LF/NGSF/NGLF/TC-NGSF/TC-NGLF
# - PV:    SF/LF/NGSF/NGLF/TC-NGSF/TC-NGLF
DRSF_R111_TABLE_MISSING_SUFFIXES_BY_FAMILY: dict[str, tuple[str, ...]] = {
    "pf-l1": ("ngsf_r111_s111", "nglf_r111_s111", "tcngsf_r111_s111", "tcnglf_r111_s111"),
    "pf-l2": (
        "sf_r111_s111",
        "lf_r111_s111",
        "ngsf_r111_s111",
        "nglf_r111_s111",
        "tcngsf_r111_s111",
        "tcnglf_r111_s111",
    ),
    "pv": (
        "sf_r111_s111",
        "lf_r111_s111",
        "ngsf_r111_s111",
        "nglf_r111_s111",
        "tcngsf_r111_s111",
        "tcnglf_r111_s111",
    ),
}


def default_families() -> dict[str, FamilyConfig]:
    return {
        "ad": FamilyConfig(
            family_id="ad",
            trainer="./trainers/main/train_author_disambiguation_shared.py",
            shared_base_dir=str(DATA_ROOT / "shared_ad_base_CS_XLNet"),
            model_dir="./model_save_shared",
            unit_prefix="ad",
            log_prefix="train_author_disambiguation_xlnet_shared",
            task_prefix="AD_XLNet",
            wandb_group="ad-shared-ablation",
            cuda="0",
            extra_args=(),
            ablations=DRSF_ABLATIONS_FULL,
        ),
        "pf-l1": FamilyConfig(
            family_id="pf-l1",
            trainer="./trainers/main/train_paper_field_shared.py",
            shared_base_dir=str(DATA_ROOT / "shared_pf_base_CS_XLNet_L1"),
            model_dir="./model_save_shared",
            unit_prefix="pf-l1",
            log_prefix="train_paper_field_l1_shared",
            task_prefix="PF_XLNet_L1",
            wandb_group="pf-l1-shared-ablation",
            cuda="1",
            extra_args=("--field_level", "L1"),
            ablations=DRSF_ABLATIONS_FULL,
        ),
        "pf-l2": FamilyConfig(
            family_id="pf-l2",
            trainer="./trainers/main/train_paper_field_shared.py",
            shared_base_dir=str(DATA_ROOT / "shared_pf_base_CS_XLNet_L2"),
            model_dir="./model_save_shared",
            unit_prefix="pf-l2",
            log_prefix="train_paper_field_l2_shared",
            task_prefix="PF_XLNet_L2",
            wandb_group="pf-l2-shared-ablation",
            cuda="2",
            extra_args=("--field_level", "L2"),
            ablations=DRSF_ABLATIONS_FULL,
        ),
        "pv": FamilyConfig(
            family_id="pv",
            trainer="./trainers/main/train_paper_venue_shared.py",
            shared_base_dir=str(DATA_ROOT / "shared_pv_base_CS_XLNet"),
            model_dir="./model_save_shared",
            unit_prefix="pv",
            log_prefix="train_paper_venue_shared",
            task_prefix="PV_XLNet",
            wandb_group="pv-shared-ablation",
            cuda="3",
            extra_args=(),
            ablations=DRSF_ABLATIONS_FULL,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate JSONL manifests for the systemd task pool scheduler.",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=["ad", "pf-l1", "pf-l2", "pv"],
        default=["ad"],
        help="Task families to include in the output manifest.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[43],
        help="Seeds to expand for every included family.",
    )
    parser.add_argument(
        "--output",
        default="-",
        help="Output JSONL path, or '-' for stdout.",
    )
    parser.add_argument(
        "--write-defaults",
        action="store_true",
        help="Write the default AD-only, combined, and TA-only manifests into trainers/main.",
    )
    parser.add_argument(
        "--ablation-set",
        choices=list(_ABLATION_SETS.keys()),
        default=None,
        help=(
            "Override ablation set for all families. "
            "full=R111+R001+R110 all variants, r111=R111 only, r001=R001 only, r110=R110 only, "
            "ta=TA-NGLF/NGSF only. "
            "Default: use the per-family default (full)."
        ),
    )
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", "THGFM"),
        help="Value passed as --wandb_project.",
    )
    parser.add_argument(
        "--wandb-entity",
        default=os.environ.get("WANDB_ENTITY", ""),
        help="Optional value passed as --wandb_entity.",
    )
    parser.add_argument(
        "--wandb-mode",
        default=os.environ.get("WANDB_MODE", ""),
        help="Optional value passed as --wandb_mode.",
    )
    parser.add_argument("--ad-cuda", default="0", help="CUDA index for AD tasks.")
    parser.add_argument("--pf-l1-cuda", default="1", help="CUDA index for PF-L1 tasks.")
    parser.add_argument("--pf-l2-cuda", default="2", help="CUDA index for PF-L2 tasks.")
    parser.add_argument("--pv-cuda", default="3", help="CUDA index for PV tasks.")
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
        "--manifest-profile",
        choices=["default", "drsf-r111-table-missing"],
        default="default",
        help=(
            "Manifest generation profile. "
            "default=regular family x ablation expansion; "
            "drsf-r111-table-missing=only missing cells for Table appendix-detailed-ablation-drsf-r111-s111."
        ),
    )
    return parser.parse_args()


def command_for_task(
    family: FamilyConfig,
    spec: AblationSpec,
    seed: int,
    *,
    n_pool: int,
    wandb_project: str,
    wandb_entity: str,
    wandb_mode: str,
) -> list[str]:
    task_name = f"{family.task_prefix}_{spec.suffix}_s{seed}"
    command = [
        "python",
        family.trainer,
        "--shared_base_dir",
        family.shared_base_dir,
        *family.extra_args,
        "--model_dir",
        family.model_dir,
        "--task_name",
        task_name,
        "--conv_name",
        spec.conv,
        "--cuda",
        family.cuda,
        "--n_pool",
        str(n_pool),
        "--seed",
        str(seed),
        "--deterministic",
        "--use_wandb",
        "--wandb_project",
        wandb_project,
        "--wandb_run_name",
        f"{task_name}_shared",
        "--wandb_group",
        family.wandb_group,
    ]
    if wandb_entity:
        command.extend(["--wandb_entity", wandb_entity])
    if wandb_mode:
        command.extend(["--wandb_mode", wandb_mode])
    # Explicitly pass branch switches (including mode='none') for reproducible ablations.
    command.extend(["--rota_mode", spec.rota_mode, "--rota_keep_rte", spec.rota_keep_rte])
    command.extend(["--rtta_rota_mode", spec.rtta_mode])
    command.extend(["--ssta_rota_mode", spec.ssta_mode])
    if spec.rtta_keep != "-1":
        command.extend(["--rtta_rota_keep_rte", spec.rtta_keep])
    if spec.ssta_keep != "-1":
        command.extend(["--ssta_rota_keep_rte", spec.ssta_keep])
    return command


_ABLATION_SETS: dict[str, tuple[AblationSpec, ...]] = {
    "full": DRSF_ABLATIONS_FULL,
    "r111": DRSF_ABLATIONS,
    "r001": DRSF_ABLATIONS_R001,
    "r110": DRSF_ABLATIONS_R110,
    "ta": DRSF_ABLATIONS_TA,
}


def tasks_for_families(
    families: list[str],
    seeds: list[int],
    *,
    n_pool: int,
    cpu_threads: int,
    wandb_project: str,
    wandb_entity: str,
    wandb_mode: str,
    cuda_overrides: dict[str, str],
    ablations_override: tuple[AblationSpec, ...] | None = None,
) -> list[dict[str, object]]:
    family_map = default_families()
    for family_id, cuda in cuda_overrides.items():
        fam = family_map[family_id]
        family_map[family_id] = FamilyConfig(
            family_id=fam.family_id,
            trainer=fam.trainer,
            shared_base_dir=fam.shared_base_dir,
            model_dir=fam.model_dir,
            unit_prefix=fam.unit_prefix,
            log_prefix=fam.log_prefix,
            task_prefix=fam.task_prefix,
            wandb_group=fam.wandb_group,
            cuda=str(cuda),
            extra_args=fam.extra_args,
            ablations=ablations_override if ablations_override is not None else fam.ablations,
        )

    tasks: list[dict[str, object]] = []
    for family_id in families:
        family = family_map[family_id]
        for seed in seeds:
            for spec in family.ablations:
                tasks.append(
                    {
                        "unit_name": f"{family.unit_prefix}-{spec.suffix}-s{seed}",
                        "workdir": str(PROJECT_DIR),
                        "log_file": f"./logs/{family.log_prefix}_{spec.suffix}_s{seed}.log",
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
                            family,
                            spec,
                            seed,
                            n_pool=n_pool,
                            wandb_project=wandb_project,
                            wandb_entity=wandb_entity,
                            wandb_mode=wandb_mode,
                        ),
                    }
                )
    return tasks


def tasks_for_missing_drsf_table_r111(
    seeds: list[int],
    *,
    n_pool: int,
    cpu_threads: int,
    wandb_project: str,
    wandb_entity: str,
    wandb_mode: str,
    cuda_overrides: dict[str, str],
) -> list[dict[str, object]]:
    family_map = default_families()
    for family_id, cuda in cuda_overrides.items():
        fam = family_map[family_id]
        family_map[family_id] = FamilyConfig(
            family_id=fam.family_id,
            trainer=fam.trainer,
            shared_base_dir=fam.shared_base_dir,
            model_dir=fam.model_dir,
            unit_prefix=fam.unit_prefix,
            log_prefix=fam.log_prefix,
            task_prefix=fam.task_prefix,
            wandb_group=fam.wandb_group,
            cuda=str(cuda),
            extra_args=fam.extra_args,
            ablations=fam.ablations,
        )

    spec_by_suffix = {spec.suffix: spec for spec in DRSF_ABLATIONS}
    tasks: list[dict[str, object]] = []
    for family_id, suffixes in DRSF_R111_TABLE_MISSING_SUFFIXES_BY_FAMILY.items():
        family = family_map[family_id]
        for seed in seeds:
            for suffix in suffixes:
                spec = spec_by_suffix[suffix]
                tasks.append(
                    {
                        "unit_name": f"{family.unit_prefix}-{spec.suffix}-s{seed}",
                        "workdir": str(PROJECT_DIR),
                        "log_file": f"./logs/{family.log_prefix}_{spec.suffix}_s{seed}.log",
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
                            family,
                            spec,
                            seed,
                            n_pool=n_pool,
                            wandb_project=wandb_project,
                            wandb_entity=wandb_entity,
                            wandb_mode=wandb_mode,
                        ),
                    }
                )
    return tasks


def write_manifest(path: Path, tasks: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task, ensure_ascii=True))
            handle.write("\n")


def main() -> int:
    args = parse_args()
    cuda_overrides = {
        "ad": args.ad_cuda,
        "pf-l1": args.pf_l1_cuda,
        "pf-l2": args.pf_l2_cuda,
        "pv": args.pv_cuda,
    }
    ablations_override = _ABLATION_SETS[args.ablation_set] if args.ablation_set else None

    shared_kwargs = dict(
        n_pool=args.n_pool,
        cpu_threads=args.cpu_threads,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
        cuda_overrides=cuda_overrides,
    )

    if args.write_defaults:
        ad_tasks = tasks_for_families(["ad"], args.seeds, **shared_kwargs)
        all_tasks = tasks_for_families(["ad", "pf-l1", "pf-l2", "pv"], args.seeds, **shared_kwargs)
        ta_tasks = tasks_for_families(
            ["ad", "pf-l1", "pf-l2", "pv"], args.seeds,
            ablations_override=DRSF_ABLATIONS_TA, **shared_kwargs,
        )
        manifest_dir = SCRIPT_DIR / "manifests"
        ad_path = manifest_dir / "task_pool.jsonl"
        all_path = manifest_dir / "task_pool_all.jsonl"
        ta_path = manifest_dir / "task_pool_ta.jsonl"
        write_manifest(ad_path, ad_tasks)
        write_manifest(all_path, all_tasks)
        write_manifest(ta_path, ta_tasks)
        print(f"wrote {len(ad_tasks)} tasks to {ad_path}")
        print(f"wrote {len(all_tasks)} tasks to {all_path}")
        print(f"wrote {len(ta_tasks)} tasks to {ta_path}")
        return 0

    if args.manifest_profile == "drsf-r111-table-missing":
        tasks = tasks_for_missing_drsf_table_r111(args.seeds, **shared_kwargs)
    else:
        tasks = tasks_for_families(
            args.families,
            args.seeds,
            ablations_override=ablations_override,
            **shared_kwargs,
        )

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
