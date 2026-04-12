"""
Layout for evaluator outputs: <save_dir>/<TASK>/<MODEL>/eval_*.jsonl

TASK is PV, PF_L1, or PF_L2. MODEL is inferred from the checkpoint file's parent
directory (e.g. drsf_tcnglf -> TC-NGLF), with optional override via CLI.
"""

from __future__ import annotations

import os
import re

# Basename of the directory containing the checkpoint file -> subfolder under task dir
# (keys are matched case-insensitively)
_VARIANT_PARENT_TO_LABEL: dict[str, str] = {
    "ssta": "SSTA",
    "ssta_rota": "SSTA",
    "ssta_rte0": "SSTA",
    "rtta": "RTTA",
    "rtta_rota": "RTTA",
    "rtta_rte0": "RTTA",
    "drsf_sf": "SF",
    "drsf_sf_rota": "SF",
    "drsf_nglf": "NGLF",
    "drsf_nglf_rota": "NGLF",
    "drsf_ngsf": "NGSF",
    "drsf_ngsf_rota": "NGSF",
    "drsf_tcnglf": "TC-NGLF",
    "drsf_tcnglf_rota": "TC-NGLF",
    "drsf_tcngsf": "TC-NGSF",
    "drsf_tcngsf_rota": "TC-NGSF",
}


def sanitize_model_subdir(name: str) -> str:
    """Single path component safe for mkdir; falls back if empty."""
    s = name.strip().strip(".")
    s = re.sub(r"[\\/]+", "_", s)
    s = re.sub(r"[^\w.\-]+", "_", s, flags=re.UNICODE)
    s = s.strip("._") or "OTHER"
    if s in (".", ".."):
        return "OTHER"
    return s


def infer_model_subdir(checkpoint_path: str) -> str:
    """
    Infer MODEL folder from the resolved checkpoint file path.
    Uses the immediate parent directory name of the checkpoint file.
    """
    parent = os.path.basename(os.path.dirname(os.path.abspath(checkpoint_path)))
    key = parent.lower()
    if key in _VARIANT_PARENT_TO_LABEL:
        return _VARIANT_PARENT_TO_LABEL[key]
    return sanitize_model_subdir(parent)


def resolve_eval_output_dir(
    save_dir: str,
    *,
    task_subdir: str,
    model_subdir: str | None,
    checkpoint_path: str,
) -> str:
    """Directory in which the .jsonl summary file should be written."""
    model = (model_subdir or "").strip() or infer_model_subdir(checkpoint_path)
    model = sanitize_model_subdir(model)
    return os.path.join(save_dir, task_subdir, model)
