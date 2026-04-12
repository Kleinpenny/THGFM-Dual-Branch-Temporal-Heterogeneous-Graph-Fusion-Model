#!/usr/bin/env python3
"""
Build a JSONL manifest for HTAG Arxiv+DBLP runs that are missing final test metrics.

Scans model_save_shared for `test_best_*` events (same completion rule as collect_htag_drsf_metrics.py).
Includes incomplete `rtta` conv runs from build_task_pool_manifest naming.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
HTAG_DIR = SCRIPT_DIR.parent.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_htag_drsf_ablation_manifest as drsf  # noqa: E402
import build_task_pool_manifest as conv  # noqa: E402
from collect_htag_drsf_metrics import _candidate_paths  # noqa: E402


def _has_test_metrics(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            ev = str(o.get("event", ""))
            if ev.startswith("test_best"):
                return True
    return False


def _metrics_complete_for_spec(*, model_dir: Path, did: str, suf: str, seed: int) -> bool:
    prefix = "PV_HTAG_Arxiv" if did == "arxiv" else "PV_HTAG_DBLP"
    cands = _candidate_paths(model_dir, prefix=prefix, did=did, suf=suf, seed=seed, legacy=True)
    return any(_has_test_metrics(p) for p in cands)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Manifest for missing HTAG PV metrics runs.")
    p.add_argument("--output", required=True, help="Output JSONL path.")
    p.add_argument("--model-dir", default=str(HTAG_DIR / "model_save_shared"))
    p.add_argument("--seeds", nargs="+", type=int, default=[43])
    p.add_argument("--n-pool", type=int, default=8)
    p.add_argument("--cpu-threads", type=int, default=1)
    p.add_argument("--cuda", default="0")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="THGFM-HTAG-PV")
    p.add_argument("--wandb-entity", default="")
    p.add_argument("--wandb-group", default="htag-gap-fill")
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--wandb-tags", nargs="*", default=["htag", "gap-fill", "pv"])
    p.add_argument("--unit-suffix", default="", help="Appended to each unit_name for uniqueness.")
    p.add_argument(
        "--force-rtta",
        action="store_true",
        help="Queue rtta even when a metrics jsonl exists but has no test_best (risk: concurrent writers).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()
    ds_map = drsf._datasets()
    suf_tag = str(args.unit_suffix).strip()
    wandb_tags = list(args.wandb_tags)

    tasks: list[dict] = []

    for seed in args.seeds:
        # DRSF appendix grid
        for did in ("arxiv", "dblp"):
            d = ds_map[did]
            for spec in drsf.all_specs():
                if _metrics_complete_for_spec(model_dir=model_dir, did=did, suf=spec.suffix, seed=seed):
                    continue
                base_unit = f"{d.unit_prefix}-gap-{_sanitize(spec.suffix)}-s{seed}"
                unit_name = f"{base_unit}-{suf_tag}" if suf_tag else base_unit
                log_name = f"./logs/task_pool/{d.log_prefix}_{spec.suffix}_s{seed}.log"
                tasks.append(
                    {
                        "unit_name": unit_name,
                        "workdir": str(HTAG_DIR),
                        "log_file": log_name,
                        "setenv": _thread_env(args.cpu_threads),
                        "command": drsf.command_for_spec(
                            d,
                            spec,
                            seed,
                            model_dir="./model_save_shared",
                            n_pool=args.n_pool,
                            cuda=args.cuda,
                            wandb_use=bool(args.wandb),
                            wandb_project=args.wandb_project,
                            wandb_entity=args.wandb_entity,
                            wandb_group=args.wandb_group or f"htag-gap-s{seed}",
                            wandb_mode=args.wandb_mode,
                        )
                        + _wandb_tags_argv(bool(args.wandb), wandb_tags),
                    }
                )

        # RTTA (single conv tag r001_s001)
        dsets = conv.default_datasets()
        for did in ("arxiv", "dblp"):
            dataset = dsets[did]
            p = model_dir / f"{dataset.task_prefix}_rtta_r001_s001_s{seed}.jsonl"
            if _has_test_metrics(p):
                continue
            if p.is_file() and p.stat().st_size > 0 and not args.force_rtta:
                print(
                    f"[skip rtta] {p.name}: metrics file present but no test_best yet "
                    f"(omit --force-rtta to avoid a second writer)",
                    file=sys.stderr,
                )
                continue
            variant_tag = conv._conv_variant_tag("rtta")
            task_name = f"{dataset.task_prefix}_rtta_{variant_tag}_s{seed}"
            base_unit = f"{dataset.unit_prefix}-gap-rtta-{variant_tag}-s{seed}"
            unit_name = f"{base_unit}-{suf_tag}" if suf_tag else base_unit
            log_name = f"./logs/task_pool/{dataset.log_prefix}_rtta_{variant_tag}_s{seed}.log"
            cmd = conv.command_for_task(
                dataset,
                conv="rtta",
                seed=seed,
                n_pool=args.n_pool,
                model_dir="./model_save_shared",
                cuda=args.cuda,
                wandb_argv=conv._wandb_argv_for_task(
                    use_wandb=bool(args.wandb),
                    wandb_project=args.wandb_project,
                    wandb_entity=args.wandb_entity,
                    wandb_group=args.wandb_group or f"htag-gap-s{seed}",
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

    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=True) + "\n")
    print(f"wrote {len(tasks)} tasks to {out}")
    return 0


def _sanitize(s: str) -> str:
    return s.replace("_", "-")


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


if __name__ == "__main__":
    raise SystemExit(main())
