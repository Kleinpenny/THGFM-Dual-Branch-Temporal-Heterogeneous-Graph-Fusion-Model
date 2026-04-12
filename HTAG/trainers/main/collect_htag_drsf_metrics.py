#!/usr/bin/env python3
"""
Read metrics JSONL files produced by HTAG PV trainers and print LaTeX-ready cells.

Expects task names from `build_htag_drsf_ablation_manifest.py`, e.g.
  model_save_shared/PV_HTAG_Arxiv_ssta_r001_s001_s43.jsonl

With `--legacy`, also checks older task_name filenames from prior manifests (same conv defaults).

Output columns: test_acc_high / test_acc_low (from test_best_* events), best_valid_acc (max over epochs).
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize HTAG DRSF appendix metrics from jsonl.")
    p.add_argument("--model-dir", default=str(HTAG_DIR / "model_save_shared"))
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--datasets", nargs="+", choices=["arxiv", "dblp"], default=["arxiv", "dblp"])
    p.add_argument(
        "--legacy",
        action="store_true",
        help="Fall back to older PV_HTAG_* jsonl filenames where the conv match is the same.",
    )
    p.add_argument(
        "--spec-suffixes",
        nargs="*",
        default=[],
        help="If set, only these spec suffixes (e.g. ssta_r001_s001). Default: all from build_htag_drsf_ablation_manifest.",
    )
    return p.parse_args()


def _default_suffixes() -> list[str]:
    import build_htag_drsf_ablation_manifest as m

    return [s.suffix for s in m.all_specs()]


# Older runs: filename suffix tags (e.g. r111_s111) came from build_task_pool_manifest heuristics, not RoTA CLI.
_LEGACY_JSONL: dict[tuple[str, str], tuple[str, ...]] = {
    ("dblp", "ssta_r001_s001"): ("PV_HTAG_DBLP_ssta_s001_s43.jsonl",),
    ("arxiv", "ssta_r001_s001"): ("PV_HTAG_Arxiv_ssta_s001_s43.jsonl",),
    ("dblp", "rtta_r001_s001"): ("PV_HTAG_DBLP_rtta_r001_s43.jsonl",),
    ("arxiv", "rtta_r001_s001"): ("PV_HTAG_Arxiv_rtta_r001_s43.jsonl",),
    ("dblp", "sf_r001_s001"): ("PV_HTAG_DBLP_drsf_sf_r001_s001_s43.jsonl",),
    ("arxiv", "sf_r001_s001"): ("PV_HTAG_Arxiv_drsf_sf_r001_s001_s43.jsonl",),
    ("dblp", "lf_r001_s001"): ("PV_HTAG_DBLP_drsf_r001_s001_s43.jsonl",),
    ("arxiv", "lf_r001_s001"): ("PV_HTAG_Arxiv_drsf_r001_s001_s43.jsonl",),
    ("dblp", "ngsf_r001_s001"): ("PV_HTAG_DBLP_drsf_ngsf_r111_s111_s43.jsonl",),
    ("arxiv", "ngsf_r001_s001"): ("PV_HTAG_Arxiv_drsf_ngsf_r111_s111_s43.jsonl",),
    ("dblp", "nglf_r001_s001"): ("PV_HTAG_DBLP_drsf_nglf_r111_s111_s43.jsonl",),
    ("arxiv", "nglf_r001_s001"): ("PV_HTAG_Arxiv_drsf_nglf_r111_s111_s43.jsonl",),
    ("dblp", "tcngsf_r001_s001"): ("PV_HTAG_DBLP_drsf_tcngsf_r111_s111_s43.jsonl",),
    ("arxiv", "tcngsf_r001_s001"): ("PV_HTAG_Arxiv_drsf_tcngsf_r111_s111_s43.jsonl",),
    ("dblp", "tcnglf_r001_s001"): ("PV_HTAG_DBLP_drsf_tcnglf_r111_s111_s43.jsonl",),
    ("arxiv", "tcnglf_r001_s001"): ("PV_HTAG_Arxiv_drsf_tcnglf_r111_s111_s43.jsonl",),
}


def _candidate_paths(model_dir: Path, *, prefix: str, did: str, suf: str, seed: int, legacy: bool) -> list[Path]:
    paths = [model_dir / f"{prefix}_{suf}_s{seed}.jsonl"]
    if legacy:
        for name in _LEGACY_JSONL.get((did, suf), ()):
            paths.append(model_dir / name)
    return paths


def _load_metrics(path: Path) -> tuple[float | None, float | None, float | None]:
    """Returns (test_high, test_low, best_val_acc)."""
    test_h = test_l = None
    best_val: float | None = None
    if not path.is_file():
        return None, None, None
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            ev = o.get("event")
            if ev == "epoch":
                v = o.get("best_valid_acc")
                if v is not None:
                    fv = float(v)
                    if best_val is None or fv > best_val:
                        best_val = fv
            elif ev == "test_best_valid_acc":
                test_h = float(o["test_acc"])
            elif ev == "test_best_valid_loss":
                test_l = float(o["test_acc"])
    return test_h, test_l, best_val


def _load_first_nonempty(paths: list[Path]) -> tuple[tuple[float | None, float | None, float | None], Path | None]:
    for p in paths:
        t = _load_metrics(p)
        if any(x is not None for x in t):
            return t, p
    return (None, None, None), None


def _fmt(x: float | None) -> str:
    if x is None:
        return "--"
    return f"{x:.4f}"


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()
    suffixes = list(args.spec_suffixes) if args.spec_suffixes else _default_suffixes()

    for did in args.datasets:
        prefix = "PV_HTAG_Arxiv" if did == "arxiv" else "PV_HTAG_DBLP"
        print(f"\n## {did} ({prefix}_*_s{args.seed}.jsonl)\n")
        for suf in suffixes:
            cands = _candidate_paths(
                model_dir, prefix=prefix, did=did, suf=suf, seed=args.seed, legacy=args.legacy
            )
            (th, tl, bv), used = _load_first_nonempty(cands)
            src = used.name if used is not None else cands[0].name
            print(f"{suf:28}  H={_fmt(th):>8}  L={_fmt(tl):>8}  val={_fmt(bv):>8}   [{src}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
