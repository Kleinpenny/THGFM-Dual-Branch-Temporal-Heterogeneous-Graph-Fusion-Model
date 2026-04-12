"""
HTAG evaluator outputs: <save_dir>/<Arxiv|DBLP>/<MODEL>/eval_*.json

TASK is inferred from --shared_base_dir unless overridden.
MODEL is inferred from the checkpoint file's parent directory (e.g. conv variant folder).
"""

from __future__ import annotations

import os
import re


def sanitize_path_component(name: str) -> str:
    s = name.strip().strip(".")
    s = re.sub(r"[\\/]+", "_", s)
    s = re.sub(r"[^\w.\-]+", "_", s, flags=re.UNICODE)
    s = s.strip("._") or "OTHER"
    if s in (".", ".."):
        return "OTHER"
    return s


def infer_task_subdir(shared_base_dir: str) -> str:
    """Map shared PV base path to Arxiv or DBLP."""
    low = os.path.abspath(shared_base_dir).lower().replace("\\", "/")
    if "htag_dblp" in low or "/dblp_pv/" in low:
        return "DBLP"
    if "htag_arxiv" in low or "/arxiv_pv/" in low:
        return "Arxiv"
    if "dblp" in low and "arxiv" not in os.path.basename(low):
        return "DBLP"
    if "arxiv" in low:
        return "Arxiv"
    return "OTHER"


def infer_model_subdir(checkpoint_path: str) -> str:
    parent = os.path.basename(os.path.dirname(os.path.abspath(checkpoint_path)))
    return sanitize_path_component(parent)


def resolve_eval_output_dir(
    save_dir: str,
    *,
    task_subdir: str,
    model_subdir: str | None,
    checkpoint_path: str,
    flat_save: bool,
) -> str:
    if flat_save:
        return save_dir
    model = (model_subdir or "").strip() or infer_model_subdir(checkpoint_path)
    model = sanitize_path_component(model)
    return os.path.join(save_dir, task_subdir, model)
