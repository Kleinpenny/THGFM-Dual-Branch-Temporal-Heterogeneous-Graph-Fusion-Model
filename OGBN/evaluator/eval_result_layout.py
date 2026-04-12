"""
OGBN-MAG evaluator outputs: <save_dir>/OGBN-MAG/<MODEL>/eval_*.json

MODEL is inferred from the checkpoint file's parent directory (same conventions as OAG table runs).
"""

from __future__ import annotations

import os
import re

TASK_SUBDIR_OGBN_MAG = "OGBN-MAG"

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


def sanitize_path_component(name: str) -> str:
    s = name.strip().strip(".")
    s = re.sub(r"[\\/]+", "_", s)
    s = re.sub(r"[^\w.\-]+", "_", s, flags=re.UNICODE)
    s = s.strip("._") or "OTHER"
    if s in (".", ".."):
        return "OTHER"
    return s


def infer_model_subdir(checkpoint_path: str) -> str:
    parent = os.path.basename(os.path.dirname(os.path.abspath(checkpoint_path)))
    key = parent.lower()
    if key in _VARIANT_PARENT_TO_LABEL:
        return _VARIANT_PARENT_TO_LABEL[key]
    return sanitize_path_component(parent)


def resolve_eval_output_dir(
    save_dir: str,
    *,
    model_subdir: str | None,
    checkpoint_path: str,
    flat_save: bool,
) -> str:
    if flat_save:
        return save_dir
    model = (model_subdir or "").strip() or infer_model_subdir(checkpoint_path)
    model = sanitize_path_component(model)
    return os.path.join(save_dir, TASK_SUBDIR_OGBN_MAG, model)
