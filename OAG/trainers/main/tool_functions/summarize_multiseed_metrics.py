#!/usr/bin/env python3
import argparse
import glob
import json
import math
import os
import re
import statistics
from typing import Dict, List, Optional


def _read_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _extract_seed(rows: List[Dict], fallback_path: str) -> str:
    for row in rows:
        if row.get("event") == "run_start" and "seed" in row:
            return str(row["seed"])
    # fallback: try parse suffix like *_s43.jsonl
    stem = os.path.splitext(os.path.basename(fallback_path))[0]
    m = re.search(r"_s(\d+)$", stem)
    return m.group(1) if m else "unknown"


def _pick_event(rows: List[Dict], event_name: str) -> Optional[Dict]:
    picked = None
    for row in rows:
        if row.get("event") == event_name:
            picked = row
    return picked


def _fmt_mean_std(values: List[float]) -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.4f} ± 0.0000 (n=1)"
    return f"{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f} (n={len(values)})"


def main():
    parser = argparse.ArgumentParser(description="Aggregate multi-seed metrics from jsonl logs.")
    parser.add_argument(
        "--glob",
        dest="glob_pattern",
        required=True,
        help='Glob pattern for jsonl files, e.g. "model_save/drsf_gate/metrics/AD_XLNet_drsf_gate_s*.jsonl".',
    )
    parser.add_argument(
        "--event",
        default="best_test_(highest_valid_ndcg)",
        choices=[
            "best_test_(highest_valid_ndcg)",
            "best_test_(lowest_valid_loss)",
            "last_test",
        ],
        help="Which event to aggregate.",
    )
    args = parser.parse_args()

    paths = sorted(glob.glob(args.glob_pattern))
    if not paths:
        raise SystemExit(f"No files matched: {args.glob_pattern}")

    ndcgs: List[float] = []
    mrrs: List[float] = []
    rows_out = []
    for path in paths:
        rows = _read_jsonl(path)
        event_row = _pick_event(rows, args.event)
        if event_row is None:
            continue
        seed = _extract_seed(rows, path)
        ndcg = event_row.get("ndcg")
        mrr = event_row.get("mrr")
        if isinstance(ndcg, (int, float)) and math.isfinite(float(ndcg)):
            ndcgs.append(float(ndcg))
        if isinstance(mrr, (int, float)) and math.isfinite(float(mrr)):
            mrrs.append(float(mrr))
        rows_out.append((seed, ndcg, mrr, path))

    if not rows_out:
        raise SystemExit(f"Matched files found, but no '{args.event}' entries were found.")

    print(f"Event: {args.event}")
    print(f"Matched files: {len(paths)}")
    print(f"NDCG: {_fmt_mean_std(ndcgs)}")
    print(f"MRR : {_fmt_mean_std(mrrs)}")
    print("\nPer-run:")
    for seed, ndcg, mrr, path in rows_out:
        print(f"  seed={seed:>7}  ndcg={float(ndcg):.4f}  mrr={float(mrr):.4f}  file={path}")


if __name__ == "__main__":
    main()
