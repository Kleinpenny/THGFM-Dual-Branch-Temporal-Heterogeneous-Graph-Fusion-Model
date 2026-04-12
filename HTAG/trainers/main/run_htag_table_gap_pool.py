#!/usr/bin/env python3
"""
Build a manifest for HTAG paper-table gaps (DBLP columns + RTTA row), optionally clean
stale checkpoints/metrics, then run trainers/main/run_manifest_subprocess_pool.py.

Table gaps: HTAG-DBLP missing for listed DRSF specs; RTTA on Arxiv+DBLP
when metrics jsonl lacks epoch 200 + test_best_* events.

Defaults: GPU 0 only, up to 12 concurrent jobs, seed 43, Weights & Biases enabled.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
HTAG_DIR = SCRIPT_DIR.parent.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_htag_drsf_ablation_manifest as drsf  # noqa: E402
import build_task_pool_manifest as conv  # noqa: E402


# Suffixes matching build_htag_drsf_ablation_manifest.HtagAblationSpec.suffix for DBLP gaps
# in the current paper table (HTAG-DBLP all "--" for these rows).
_GAP_DBLP_SUFFIXES: frozenset[str] = frozenset(
    {
        "lf_r001_s001",
        "ngsf_r001_s001",
        "nglf_r001_s001",
        "tcngsf_r001_s001",
        "tcnglf_r001_s001",
        "ssta_r110_s110",
        "rtta_r110_s110",
        "sf_r110_s110",
        "lf_r110_s110",
        "ngsf_r110_s110",
        "nglf_r110_s110",
        "tcngsf_r110_s110",
        "tcnglf_r110_s110",
        "ssta_r111_s111",
        "rtta_r111_s111",
        "sf_r111_s111",
        "lf_r111_s111",
        "ngsf_r111_s111",
        "nglf_r111_s111",
        "tcngsf_r111_s111",
        "tcnglf_r111_s111",
    }
)


def _is_run_complete(metrics_path: Path) -> bool:
    if not metrics_path.is_file():
        return False
    has_200 = False
    has_acc = False
    has_loss = False
    try:
        with metrics_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ev = str(o.get("event", ""))
                if ev == "epoch" and int(o.get("epoch", 0)) == 200:
                    has_200 = True
                if ev == "test_best_valid_acc":
                    has_acc = True
                if ev == "test_best_valid_loss":
                    has_loss = True
    except OSError:
        return False
    return has_200 and has_acc and has_loss


def _clean_task_artifacts(model_dir: Path, task_name: str) -> None:
    for name in (
        f"{task_name}.jsonl",
        f"{task_name}_highest_valid_acc.pt",
        f"{task_name}_lowest_valid_loss.pt",
    ):
        p = model_dir / name
        if p.is_file():
            p.unlink()
            print(f"[clean] removed {p}")


def _thread_env(cpu_threads: int) -> dict[str, str]:
    t = str(int(cpu_threads))
    return {
        "OMP_NUM_THREADS": t,
        "MKL_NUM_THREADS": t,
        "OPENBLAS_NUM_THREADS": t,
        "NUMEXPR_NUM_THREADS": t,
        "BLIS_NUM_THREADS": t,
        "VECLIB_MAXIMUM_THREADS": t,
        "TORCH_NUM_THREADS": t,
        "OMP_WAIT_POLICY": "PASSIVE",
        "KMP_BLOCKTIME": "0",
    }


def _wandb_tags_argv(use: bool, tags: list[str]) -> list[str]:
    if not use or not tags:
        return []
    return ["--wandb_tags", *tags]


def _parse_task_name_from_command(cmd: list[str]) -> str | None:
    for i, tok in enumerate(cmd):
        if tok == "--task_name" and i + 1 < len(cmd):
            return str(cmd[i + 1])
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HTAG table-gap task pool (DBLP + RTTA).")
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--model-dir", default=str(HTAG_DIR / "model_save_shared"))
    p.add_argument("--n-pool", type=int, default=8)
    p.add_argument("--cpu-threads", type=int, default=1)
    p.add_argument("--cuda", default="0", help="Placeholder in manifest; pool overwrites to --gpu-ids.")
    p.add_argument(
        "--manifest",
        default="",
        help="Output JSONL path (default: trainers/main/manifests/task_pool_htag_table_gap_s{seed}.jsonl).",
    )
    p.add_argument(
        "--skip-complete",
        action="store_true",
        help="Omit tasks whose metrics jsonl already has epoch 200 + test_best_* events.",
    )
    p.add_argument(
        "--clean",
        action="store_true",
        help="Before running, delete metrics jsonl + best checkpoints for each queued task_name.",
    )
    p.add_argument("--wandb", dest="wandb", action="store_true", help="Enable Weights & Biases (default: on).")
    p.add_argument("--no-wandb", dest="wandb", action="store_false", help="Disable Weights & Biases.")
    p.set_defaults(wandb=True)
    p.add_argument("--wandb-project", default="THGFM-HTAG-PV")
    p.add_argument("--wandb-entity", default="")
    p.add_argument("--wandb-group", default="htag-table-gap")
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--wandb-tags", nargs="*", default=["htag", "table-gap", "pv"])
    p.add_argument("--unit-suffix", default="gap20260331", help="Appended to unit_name for uniqueness.")
    p.add_argument("--dry-run", action="store_true", help="Print tasks and exit (no manifest write, no train).")
    p.add_argument(
        "--no-run",
        action="store_true",
        help="Only write manifest (and optional clean); do not launch subprocess pool.",
    )
    p.add_argument("--max-running", type=int, default=12)
    p.add_argument("--max-per-gpu", type=int, default=12)
    p.add_argument("--gpu-ids", nargs="+", default=["0"])
    p.add_argument("--poll-seconds", type=int, default=20)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    seed = int(args.seed)
    model_dir = Path(args.model_dir).expanduser().resolve()
    ds_map = drsf._datasets()
    suf = str(args.unit_suffix).strip()
    wandb_tags = list(args.wandb_tags)

    tasks: list[dict] = []

    # --- DBLP DRSF gap specs ---
    for spec in drsf.all_specs():
        if spec.suffix not in _GAP_DBLP_SUFFIXES:
            continue
        d = ds_map["dblp"]
        task_name = f"{d.task_prefix}_{spec.suffix}_s{seed}"
        metrics_path = model_dir / f"{task_name}.jsonl"
        if args.skip_complete and _is_run_complete(metrics_path):
            print(f"[skip complete] {metrics_path.name}")
            continue

        base_unit = f"{d.unit_prefix}-table-gap-{_sanitize(spec.suffix)}-s{seed}"
        unit_name = f"{base_unit}-{suf}" if suf else base_unit
        log_name = f"./logs/task_pool/{d.log_prefix}_table_gap_{spec.suffix}_s{seed}.log"
        cmd = drsf.command_for_spec(
            d,
            spec,
            seed,
            model_dir="./model_save_shared",
            n_pool=args.n_pool,
            cuda=args.cuda,
            wandb_use=args.wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_group=args.wandb_group or f"htag-table-gap-s{seed}",
            wandb_mode=args.wandb_mode,
        )
        cmd.extend(_wandb_tags_argv(args.wandb, wandb_tags))
        tasks.append(
            {
                "unit_name": unit_name,
                "workdir": str(HTAG_DIR),
                "log_file": log_name,
                "setenv": _thread_env(args.cpu_threads),
                "command": cmd,
            }
        )

    # --- RTTA (Arxiv + DBLP), same naming as build_task_pool_manifest ---
    dsets = conv.default_datasets()
    for did in ("arxiv", "dblp"):
        dataset = dsets[did]
        task_name = f"{dataset.task_prefix}_rtta_{conv._conv_variant_tag('rtta')}_s{seed}"
        metrics_path = model_dir / f"{task_name}.jsonl"
        if args.skip_complete and _is_run_complete(metrics_path):
            print(f"[skip complete] {metrics_path.name}")
            continue

        base_unit = f"{dataset.unit_prefix}-table-gap-rtta-{conv._conv_variant_tag('rtta')}-s{seed}"
        unit_name = f"{base_unit}-{suf}" if suf else base_unit
        log_name = f"./logs/task_pool/{dataset.log_prefix}_table_gap_rtta_s{seed}.log"
        cmd = conv.command_for_task(
            dataset,
            conv="rtta",
            seed=seed,
            n_pool=args.n_pool,
            model_dir="./model_save_shared",
            cuda=args.cuda,
            wandb_argv=conv._wandb_argv_for_task(
                use_wandb=args.wandb,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                wandb_group=args.wandb_group or f"htag-table-gap-s{seed}",
                wandb_mode=args.wandb_mode,
                wandb_tags=wandb_tags,
                task_name=task_name,
            ),
        )
        tasks.append(
            {
                "unit_name": unit_name,
                "workdir": str(HTAG_DIR),
                "log_file": log_name,
                "setenv": _thread_env(args.cpu_threads),
                "command": cmd,
            }
        )

    if args.manifest:
        out = Path(args.manifest).expanduser().resolve()
    else:
        out = SCRIPT_DIR / "manifests" / f"task_pool_htag_table_gap_s{seed}.jsonl"

    print(f"[pool plan] queued_tasks={len(tasks)} (DBLP gap + RTTA x2)")
    if args.dry_run:
        for t in tasks:
            print(f"  {t['unit_name']}: {' '.join(t['command'])}")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=True) + "\n")
    print(f"[write] {out}")

    if args.clean:
        for t in tasks:
            tn = _parse_task_name_from_command(t["command"])
            if tn:
                _clean_task_artifacts(model_dir, tn)

    if args.no_run:
        return 0

    pool_py = SCRIPT_DIR / "run_manifest_subprocess_pool.py"
    cmd = [
        sys.executable,
        str(pool_py),
        "--manifest",
        str(out),
        "--project-dir",
        str(HTAG_DIR),
        "--gpu-ids",
        *[str(x) for x in args.gpu_ids],
        "--max-running",
        str(int(args.max_running)),
        "--max-per-gpu",
        str(int(args.max_per_gpu)),
        "--poll-seconds",
        str(int(args.poll_seconds)),
    ]
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", ",".join(str(x) for x in args.gpu_ids))
    print(f"[exec] {' '.join(cmd)}")
    return int(subprocess.call(cmd, cwd=str(HTAG_DIR), env=env))


def _sanitize(s: str) -> str:
    return s.replace("_", "-")


if __name__ == "__main__":
    raise SystemExit(main())
