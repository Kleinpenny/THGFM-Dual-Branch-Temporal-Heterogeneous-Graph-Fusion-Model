"""
Preprocess ogbn-mag into THGFM Graph format (same pipeline as preprocess_OAG_XLNet.py).

- Paper features: OGB 128-d word2vec (node-feat/paper/node-feat.csv), not XLNet.
- Heterogeneous edges: author–institution, author–paper, paper–paper (cite),
  paper–field, paper–venue (from venue labels).
- Propagates paper embeddings to other types via neighborhood averaging; uses
  time <= test_time_bar (default 2017) for venue/field, matching train papers
  before the 2018 validation split.

Output: graph_ogbn_mag_w2v.pk under --output_dir (default: OGBN/data/).
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import dill
import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
OGBN_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = OGBN_ROOT.parent
OAG_ROOT = PROJECT_ROOT / "OAG"
if str(OAG_ROOT) not in sys.path:
    sys.path.insert(0, str(OAG_ROOT))

from THGFM.data import Graph
from THGFM.utils import normalize

parser = argparse.ArgumentParser(
    description="Preprocess ogbn-mag (THGFM Graph, OGB word2vec features)"
)
parser.add_argument(
    "--ogbn_mag_dir",
    type=str,
    default=str(PROJECT_ROOT / "data" / "ogbn_mag"),
    help="Extracted ogbn-mag directory (raw/, split/, mapping/).",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=str(OGBN_ROOT / "data"),
    help="Directory for graph_ogbn_mag_w2v.pk",
)
parser.add_argument(
    "--test_time_bar",
    type=int,
    default=2017,
    help="Max paper year used when aggregating paper embeddings onto "
    "venue/field (train papers are before 2018 in OGB).",
)
args = parser.parse_args()


def _line_count(path: Path) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def main() -> None:
    ogbn_mag_dir = Path(args.ogbn_mag_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = ogbn_mag_dir / "raw"
    paper_feat_path = raw / "node-feat" / "paper" / "node-feat.csv"
    paper_year_path = raw / "node-feat" / "paper" / "node_year.csv"
    paper_label_path = raw / "node-label" / "paper" / "node-label.csv"

    print(f"[INFO] ogbn_mag_dir={ogbn_mag_dir}", flush=True)
    print(f"[INFO] output_dir={out_dir}", flush=True)

    paper_feat = np.loadtxt(paper_feat_path, delimiter=",", dtype=np.float32)
    if paper_feat.ndim == 1:
        paper_feat = paper_feat.reshape(1, -1)
    paper_year = np.loadtxt(paper_year_path, delimiter=",", dtype=np.int32)
    if paper_year.ndim == 0:
        paper_year = paper_year.reshape(1)
    paper_label_raw = np.loadtxt(paper_label_path, delimiter=",", dtype=np.int64)
    if paper_label_raw.ndim == 0:
        paper_label_raw = paper_label_raw.reshape(1)

    n_papers = paper_feat.shape[0]
    if paper_year.shape[0] != n_papers or paper_label_raw.shape[0] != n_papers:
        raise ValueError("paper feat / year / label length mismatch")

    unique_labels = np.sort(np.unique(paper_label_raw[paper_label_raw >= 0])).astype(
        np.int64, copy=False
    )
    label_to_vidx = {int(lbl): int(i) for i, lbl in enumerate(unique_labels)}
    paper_venue_idx = np.full((n_papers,), -1, dtype=np.int64)
    for lbl, vidx in label_to_vidx.items():
        paper_venue_idx[paper_label_raw == lbl] = vidx

    paper_nodes = {}
    for i in range(n_papers):
        pid = str(i)
        paper_nodes[pid] = {
            "id": pid,
            "type": "paper",
            "time": int(paper_year[i]),
            "emb": paper_feat[i].tolist(),
        }

    graph = Graph()

    # --- Paper -> venue (PV_Journal) ---
    print("[INFO] Paper–venue edges…", flush=True)
    for pid, pnode in tqdm(paper_nodes.items(), total=len(paper_nodes)):
        vidx = paper_venue_idx[int(pid)]
        if vidx < 0:
            continue
        venue_node = {"id": str(vidx), "type": "venue", "attr": "journal"}
        graph.add_edge(
            pnode,
            venue_node,
            time=pnode["time"],
            relation_type="PV_Journal",
        )

    # --- Paper -> paper (PP_cite) ---
    cite_path = raw / "relations" / "paper___cites___paper" / "edge.csv"
    n_cite = _line_count(cite_path)
    print(f"[INFO] Paper–paper cites from {cite_path} ({n_cite} lines)…", flush=True)
    with open(cite_path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, total=n_cite):
            line = line.strip()
            if not line:
                continue
            a, b = line.split(",")
            p1, p2 = str(int(a)), str(int(b))
            if p1 not in paper_nodes or p2 not in paper_nodes:
                continue
            n1, n2 = paper_nodes[p1], paper_nodes[p2]
            if n1["time"] >= n2["time"]:
                graph.add_edge(n1, n2, time=n1["time"], relation_type="PP_cite")

    # --- Paper -> field (PF_in_L2) ---
    pf_path = raw / "relations" / "paper___has_topic___field_of_study" / "edge.csv"
    n_pf = _line_count(pf_path)
    print(f"[INFO] Paper–field edges… ({n_pf} lines)", flush=True)
    with open(pf_path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, total=n_pf):
            line = line.strip()
            if not line:
                continue
            p, f = line.split(",")
            pid, fid = str(int(p)), str(int(f))
            if pid not in paper_nodes:
                continue
            pnode = paper_nodes[pid]
            field_node = {"id": fid, "type": "field", "attr": "L2"}
            graph.add_edge(
                pnode,
                field_node,
                time=pnode["time"],
                relation_type="PF_in_L2",
            )

    # --- Author -> affiliation (in); collect author order per paper ---
    ap_path = raw / "relations" / "author___writes___paper" / "edge.csv"
    n_ap = _line_count(ap_path)
    ai_path = raw / "relations" / "author___affiliated_with___institution" / "edge.csv"
    n_ai = _line_count(ai_path)

    paper_authors = defaultdict(dict)
    seen_ap = set()
    print(f"[INFO] Author–paper order + author–affiliation…", flush=True)
    with open(ap_path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, total=n_ap):
            line = line.strip()
            if not line:
                continue
            author_s, paper_s = line.split(",")
            aid, pid = int(author_s), int(paper_s)
            pid_str = str(pid)
            if pid_str not in paper_nodes:
                continue
            if (aid, pid) in seen_ap:
                continue
            seen_ap.add((aid, pid))
            pos = len(paper_authors[pid_str]) + 1
            paper_authors[pid_str][pos] = {"id": str(aid), "type": "author"}

    with open(ai_path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, total=n_ai):
            line = line.strip()
            if not line:
                continue
            author_s, inst_s = line.split(",")
            aid, iid = str(int(author_s)), str(int(inst_s))
            author_node = {"id": aid, "type": "author"}
            aff_node = {"id": iid, "type": "affiliation"}
            graph.add_edge(author_node, aff_node, relation_type="in")

    print("[INFO] Author–paper triples…", flush=True)
    for pid_str in tqdm(paper_authors):
        pnode = paper_nodes[pid_str]
        seq_keys = paper_authors[pid_str]
        if not seq_keys:
            continue
        max_seq = max(seq_keys.keys())
        for seq_i in seq_keys:
            author_node = seq_keys[seq_i]
            if seq_i == 1:
                rel = "AP_write_first"
            elif seq_i == max_seq:
                rel = "AP_write_last"
            else:
                rel = "AP_write_other"
            graph.add_edge(
                author_node,
                pnode,
                time=pnode["time"],
                relation_type=rel,
            )

    # --- Optional venue names ---
    vname_path = ogbn_mag_dir / "mapping" / "labelidx2venuename.csv"
    if vname_path.is_file():
        print("[INFO] Venue names from mapping…", flush=True)
        with open(vname_path, "r", encoding="utf-8") as fin:
            fin.readline()
            for line in fin:
                parts = line.strip().split(",", 1)
                if len(parts) < 2:
                    continue
                vidx_s, vname = parts[0].strip(), parts[1].strip()
                if vidx_s in graph.node_forward["venue"]:
                    idx = graph.node_forward["venue"][vidx_s]
                    graph.node_backward["venue"][idx]["name"] = vname

    # --- Citation counts (same as OAG) ---
    print("[INFO] Citation node attributes…", flush=True)
    for paper_idx, paper_node in enumerate(graph.node_backward["paper"]):
        paper_node["citation"] = len(
            graph.edge_list["paper"]["paper"]["PP_cite"][paper_idx]
        )

    for author_idx, author_node in enumerate(graph.node_backward["author"]):
        citation = 0
        for rel in graph.edge_list["author"]["paper"]:
            for paper_idx in graph.edge_list["author"]["paper"][rel][author_idx]:
                citation += graph.node_backward["paper"][paper_idx]["citation"]
        author_node["citation"] = citation

    for aff_idx, aff_node in enumerate(graph.node_backward["affiliation"]):
        citation = 0
        for author_idx in graph.edge_list["affiliation"]["author"]["in"][aff_idx]:
            citation += graph.node_backward["author"][author_idx]["citation"]
        aff_node["citation"] = citation

    for venue_idx, venue_node in enumerate(graph.node_backward["venue"]):
        citation = 0
        for rel in graph.edge_list["venue"]["paper"]:
            for paper_idx in graph.edge_list["venue"]["paper"][rel][venue_idx]:
                citation += graph.node_backward["paper"][paper_idx]["citation"]
        venue_node["citation"] = citation

    for field_idx, field_node in enumerate(graph.node_backward["field"]):
        citation = 0
        for rel in graph.edge_list["field"]["paper"]:
            for paper_idx in graph.edge_list["field"]["paper"][rel][field_idx]:
                citation += graph.node_backward["paper"][paper_idx]["citation"]
        field_node["citation"] = citation

    # --- DataFrames + embedding propagation ---
    print("[INFO] Propagating embeddings (non-paper types)…", flush=True)
    test_time_bar = args.test_time_bar
    df = pd.DataFrame(graph.node_backward["paper"])
    graph.node_feature = {"paper": df}
    paper_embeddings = np.array(list(df["emb"]), dtype=np.float32)

    for _type in graph.node_backward:
        if _type in ("paper", "affiliation"):
            continue
        df_t = pd.DataFrame(graph.node_backward[_type])
        node_pairs = []
        for _rel in graph.edge_list[_type]["paper"]:
            for target_idx in graph.edge_list[_type]["paper"][_rel]:
                for source_idx in graph.edge_list[_type]["paper"][_rel][target_idx]:
                    t = graph.edge_list[_type]["paper"][_rel][target_idx][source_idx]
                    if t is not None and t <= test_time_bar:
                        node_pairs.append([target_idx, source_idx])
        if not node_pairs:
            continue
        node_pairs = np.array(node_pairs, dtype=np.int64).T
        edge_count = node_pairs.shape[1]
        v = np.ones(edge_count)
        m = normalize(
            sp.coo_matrix(
                (v, (node_pairs[0], node_pairs[1])),
                shape=(
                    len(graph.node_backward[_type]),
                    len(graph.node_backward["paper"]),
                ),
            )
        )
        out = m.dot(paper_embeddings)
        df_t["emb"] = list(out)
        graph.node_feature[_type] = df_t

    author_embeddings = np.array(
        list(graph.node_feature["author"]["emb"]), dtype=np.float32
    )
    df_a = pd.DataFrame(graph.node_backward["affiliation"])
    node_pairs = []
    for _rel in graph.edge_list["affiliation"]["author"]:
        for target_idx in graph.edge_list["affiliation"]["author"][_rel]:
            for source_idx in graph.edge_list["affiliation"]["author"][_rel][
                target_idx
            ]:
                node_pairs.append([target_idx, source_idx])
    if node_pairs:
        node_pairs = np.array(node_pairs, dtype=np.int64).T
        edge_count = node_pairs.shape[1]
        v = np.ones(edge_count)
        m = normalize(
            sp.coo_matrix(
                (v, (node_pairs[0], node_pairs[1])),
                shape=(
                    len(graph.node_backward["affiliation"]),
                    len(graph.node_backward["author"]),
                ),
            )
        )
        df_a["emb"] = list(m.dot(author_embeddings))
    else:
        dim = paper_embeddings.shape[1]
        df_a["emb"] = [np.zeros(dim, dtype=np.float32)] * len(df_a)
    graph.node_feature["affiliation"] = df_a

    # --- Clean edge list (same as OAG) ---
    print("[INFO] Cleaning edge list…", flush=True)
    clean_edge_list = {}
    for k1 in graph.edge_list:
        clean_edge_list[k1] = {}
        for k2 in graph.edge_list[k1]:
            clean_edge_list[k1][k2] = {}
            for k3 in graph.edge_list[k1][k2]:
                clean_edge_list[k1][k2][k3] = {}
                triple_count = 0
                for e1 in graph.edge_list[k1][k2][k3]:
                    edge_count = len(graph.edge_list[k1][k2][k3][e1])
                    triple_count += edge_count
                    if edge_count == 0:
                        continue
                    clean_edge_list[k1][k2][k3][e1] = {}
                    for e2 in graph.edge_list[k1][k2][k3][e1]:
                        clean_edge_list[k1][k2][k3][e1][e2] = graph.edge_list[k1][k2][
                            k3
                        ][e1][e2]
                print(k1, k2, k3, triple_count)

    graph.edge_list = clean_edge_list

    print("Number of nodes:", flush=True)
    for node_type in graph.node_forward:
        print(f"  {node_type}: {len(graph.node_forward[node_type]):,}", flush=True)

    del graph.node_backward

    out_path = out_dir / "graph_ogbn_mag_w2v.pk"
    print(f"[INFO] Writing {out_path}", flush=True)
    with open(out_path, "wb") as f:
        dill.dump(graph, f)
    print("[OK] Done.", flush=True)


if __name__ == "__main__":
    main()
