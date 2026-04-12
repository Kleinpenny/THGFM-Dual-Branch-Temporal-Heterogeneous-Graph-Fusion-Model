#!/usr/bin/env python3
"""
Count trainable parameters for models saved under model_save/.

Two ways:
1. Load actual checkpoints: reports total (GNN + task head) and file size.
   Model file size on disk != parameter count (serialization, float32/16, etc.).
2. Build GNN only with fixed config: comparable GNN parameter count per conv type,
   independent of task (Classifier/Matcher) or graph (in_dim varies by embedding).
"""
import os
import sys
import torch

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_OAG_DIR = os.path.abspath(os.path.join(_CUR_DIR, "..", "..", ".."))
if _OAG_DIR not in sys.path:
    sys.path.insert(0, _OAG_DIR)

from THGFM.model import GNN

_TRAINERS_MAIN_DIR = os.path.abspath(os.path.join(_CUR_DIR, ".."))
MODEL_SAVE = os.path.join(_TRAINERS_MAIN_DIR, "model_save")
DEVICE = torch.device("cpu")

# One checkpoint per conv type (for method 1). Head size differs by task (AD vs PF_L2).
CHECKPOINTS = {
    "rtta": os.path.join(MODEL_SAVE, "rtta", "PF_Qwen3_fp16_L2"),
    "ssta": os.path.join(MODEL_SAVE, "ssta", "AD_Qwen3_fp16"),
    "drsf": os.path.join(MODEL_SAVE, "drsf", "PF_Qwen3_fp16_L2"),
}

# Representative trained PF-L2 checkpoints for method-level comparison.
# Note: this repo stores RTTA and DRSF variants in rtta_rota/drsf_gate folders.
METHOD_CHECKPOINTS = {
    "SSTA": os.path.join(MODEL_SAVE, "ssta", "PF_XLNet_L2_ssta_L2"),
    "RTTA": os.path.join(MODEL_SAVE, "rtta_rota", "PF_XLNet_L2_rtta_rota_qk_rte0_L2"),
    "DRSF": os.path.join(MODEL_SAVE, "drsf_gate", "PF_XLNet_L2_drsf_gate_L2"),
}

# in_dim = paper_emb_dim + 401. XLNet 768+401=1169; Qwen3-8B 4096+401=4497.
IN_DIM_XLNET = 1169    # 768 + 401
IN_DIM_QWEN3 = 4497    # 4096 + 401 (Qwen3 / Qwen3-fp16 同)
N_HID = 256
N_HEADS = 8
N_LAYERS = 3
NUM_TYPES = 5
NUM_RELATIONS = 24


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_ssta_rtta_drsf_in_m():
    """Count trainable parameters (in M) from trained checkpoints."""
    print("=" * 72)
    print("SSTA / RTTA / DRSF 可训练参数量（基于已训练权重，单位: M）")
    print("=" * 72)
    print(f"{'Method':<8}  {'Total(M)':>10}  {'GNN(M)':>10}  {'Head(M)':>10}  {'Checkpoint':<}")
    print("-" * 72)
    for method, path in METHOD_CHECKPOINTS.items():
        if not os.path.isfile(path):
            print(f"{method:<8}  {'N/A':>10}  {'N/A':>10}  {'N/A':>10}  file not found: {path}")
            continue
        try:
            obj = torch.load(path, map_location=DEVICE, weights_only=False)
            total = count_trainable_params(obj)
            gnn_only = count_trainable_params(obj[0]) if len(obj) >= 1 else total
            head_only = count_trainable_params(obj[1]) if len(obj) >= 2 else 0
            print(
                f"{method:<8}  {total/1e6:10.3f}  {gnn_only/1e6:10.3f}  {head_only/1e6:10.3f}  "
                f"{os.path.relpath(path, _CUR_DIR)}"
            )
        except Exception as e:
            print(f"{method:<8}  {'ERR':>10}  {'ERR':>10}  {'ERR':>10}  {e}")
    print("-" * 72)
    print()


def count_from_checkpoints():
    print("=" * 60)
    print("(1) From loaded checkpoints (total = GNN + task head)")
    print("    File size on disk ≠ parameter count.")
    print("=" * 60)
    for name, path in CHECKPOINTS.items():
        if not os.path.isfile(path):
            print(f"{name}: file not found")
            continue
        try:
            obj = torch.load(path, map_location=DEVICE, weights_only=False)
            if hasattr(obj, "parameters"):
                total = count_trainable_params(obj)
                gnn_only = count_trainable_params(obj[0]) if len(obj) >= 1 else total
                head_only = count_trainable_params(obj[1]) if len(obj) >= 2 else 0
            else:
                gnn_only = count_trainable_params(obj[0])
                head_only = count_trainable_params(obj[1])
                total = gnn_only + head_only
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"{name:14s}  total: {total:>12,}  GNN: {gnn_only:>12,}  head: {head_only:>10,}  file: {size_mb:.1f} MB")
        except Exception as e:
            print(f"{name}: error: {e}")
    print()


def count_gnn_per_embedding():
    """GNN 参数量按 embedding 类型分别计算（in_dim 不同）。"""
    conv_names = ["rtta", "rtta_rota", "ssta", "ssta_rota", "drsf", "drsf_rota", "drsf_gate", "drsf_gate_rota"]
    configs = [
        ("XLNet (in_dim=1169)", IN_DIM_XLNET),
        ("Qwen3 / Qwen3-fp16 (in_dim=3985)", IN_DIM_QWEN3),
    ]
    print("=" * 70)
    print("GNN 参数量（按 embedding 类型分别计算）")
    print(f"    n_hid={N_HID}, n_heads={N_HEADS}, n_layers={N_LAYERS}, "
          f"num_types={NUM_TYPES}, num_relations={NUM_RELATIONS}")
    print("=" * 70)
    for label, in_dim in configs:
        print(f"\n--- {label} ---")
        for conv_name in conv_names:
            try:
                gnn = GNN(
                    conv_name=conv_name,
                    in_dim=in_dim,
                    n_hid=N_HID,
                    n_heads=N_HEADS,
                    n_layers=N_LAYERS,
                    num_types=NUM_TYPES,
                    num_relations=NUM_RELATIONS,
                    dropout=0.2,
                )
                n = count_trainable_params(gnn)
                print(f"  {conv_name:14s}  {n:>12,}  ({n/1e6:.2f}M)")
            except Exception as e:
                print(f"  {conv_name:14s}  error: {e}")
    print("\n" + "=" * 70)
    # Summary table
    print("\n汇总表 (GNN 参数量):")
    print("-" * 70)
    print(f"{'模型':<16}  {'XLNet (1169)':>18}  {'Qwen3 (3985)':>18}")
    print("-" * 70)
    for conv_name in conv_names:
        row = [conv_name]
        for _, in_dim in configs:
            try:
                gnn = GNN(
                    conv_name=conv_name,
                    in_dim=in_dim,
                    n_hid=N_HID,
                    n_heads=N_HEADS,
                    n_layers=N_LAYERS,
                    num_types=NUM_TYPES,
                    num_relations=NUM_RELATIONS,
                    dropout=0.2,
                )
                n = count_trainable_params(gnn)
                row.append(f"{n:,} ({n/1e6:.2f}M)")
            except Exception:
                row.append("N/A")
        print(f"  {row[0]:<14}  {row[1]:>18}  {row[2]:>18}")
    print("-" * 70)


def main():
    count_ssta_rtta_drsf_in_m()
    count_from_checkpoints()
    count_gnn_per_embedding()


if __name__ == "__main__":
    main()
