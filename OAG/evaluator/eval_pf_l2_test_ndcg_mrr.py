#!/usr/bin/env python3
"""PF-L2 (level-2 paper-field) test split: NDCG and MRR. See eval_pf_common.py."""

import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from eval_pf_common import run_main

if __name__ == "__main__":
    raise SystemExit(run_main("L2"))
