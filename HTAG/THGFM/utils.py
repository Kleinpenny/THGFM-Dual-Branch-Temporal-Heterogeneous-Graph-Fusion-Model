import numpy as np
import scipy.sparse as sp
import torch

def dcg_at_k(r, k):
    r = np.asarray(r, dtype=float)[:k]
    if r.size:
        return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
    return 0.

def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max


def mean_reciprocal_rank(rs):
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return [1. / (r[0] + 1) if r.size else 0. for r in rs]


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def randint():
    return np.random.randint(2**32 - 1)


def build_feature_cache(graph):
    """
    Build per-type numpy caches to speed up repeated feature extraction during sampling.
    """
    cache = {}
    for node_type, df in graph.node_feature.items():
        if len(df) == 0:
            continue
        type_cache = {}
        index_values = np.array(df.index)
        type_cache["id_to_pos"] = {idx: pos for pos, idx in enumerate(index_values)}
        if "emb" in df:
            type_cache["emb"] = np.asarray(list(df["emb"]), dtype=np.float32)
        if "node_emb" in df:
            type_cache["node_emb"] = np.asarray(list(df["node_emb"]), dtype=np.float32)
        if "citation" in df:
            type_cache["citation"] = np.asarray(list(df["citation"]), dtype=np.float32).reshape(-1, 1)
        if "title" in df:
            type_cache["title"] = np.asarray(list(df["title"]), dtype=object)
        cache[node_type] = type_cache
    graph._feature_cache = cache


def _positions_from_ids(id_to_pos, idxs):
    return np.fromiter((id_to_pos[i] for i in idxs), dtype=np.int64, count=len(idxs))


def _pad_or_trim_feat(feat: np.ndarray, dim: int) -> np.ndarray:
    if feat.shape[1] == dim:
        return feat
    if feat.shape[1] > dim:
        return feat[:, :dim]
    pad = np.zeros((feat.shape[0], dim - feat.shape[1]), dtype=np.float32)
    return np.concatenate([feat, pad], axis=1)


def _feature_mode(graph):
    return str(getattr(graph, "feature_mode", "oag")).lower()


def _graph_feature_dim(graph):
    dim = getattr(graph, "feature_dim", None)
    if dim is None:
        return None
    try:
        return int(dim)
    except Exception:
        return None


def _extract_feature_array(idxs, _type, graph, cache):
    desired_dim = _graph_feature_dim(graph)
    mode = _feature_mode(graph)
    use_oag_layout = (mode == "oag")

    if cache is not None and _type in cache:
        tc = cache[_type]
        pos = _positions_from_ids(tc["id_to_pos"], idxs)
        if use_oag_layout and ("emb" in tc) and ("citation" in tc):
            if "node_emb" in tc:
                node_emb = tc["node_emb"][pos]
            else:
                node_emb = np.zeros([len(idxs), 400], dtype=np.float32)
            emb = tc["emb"][pos]
            citation = tc["citation"][pos]
            feat = np.concatenate((node_emb, emb, np.log10(citation + 0.01)), axis=1)
        else:
            if "emb" in tc:
                feat = tc["emb"][pos]
            elif "node_emb" in tc:
                feat = tc["node_emb"][pos]
            elif desired_dim is not None:
                feat = np.zeros((len(idxs), desired_dim), dtype=np.float32)
            else:
                feat = np.zeros((len(idxs), 1), dtype=np.float32)
        if _type == "paper" and "title" in tc:
            texts = tc["title"][pos].astype(str)
        else:
            texts = []
        feat = np.asarray(feat, dtype=np.float32)
        if desired_dim is not None:
            feat = _pad_or_trim_feat(feat, desired_dim)
        return feat, texts

    sub_df = graph.node_feature[_type].loc[idxs]
    if use_oag_layout and ("emb" in sub_df) and ("citation" in sub_df):
        if "node_emb" in sub_df:
            node_emb = np.asarray(list(sub_df["node_emb"]), dtype=np.float32)
        else:
            node_emb = np.zeros([len(idxs), 400], dtype=np.float32)
        emb = np.asarray(list(sub_df["emb"]), dtype=np.float32)
        citation = np.asarray(list(sub_df["citation"]), dtype=np.float32).reshape(-1, 1)
        feat = np.concatenate((node_emb, emb, np.log10(citation + 0.01)), axis=1)
    else:
        if "emb" in sub_df:
            feat = np.asarray(list(sub_df["emb"]), dtype=np.float32)
        elif "node_emb" in sub_df:
            feat = np.asarray(list(sub_df["node_emb"]), dtype=np.float32)
        elif desired_dim is not None:
            feat = np.zeros((len(sub_df), desired_dim), dtype=np.float32)
        else:
            feat = np.zeros((len(sub_df), 1), dtype=np.float32)
    if _type == "paper" and "title" in sub_df:
        texts = np.asarray(list(sub_df["title"]), dtype=str)
    else:
        texts = []
    feat = np.asarray(feat, dtype=np.float32)
    if desired_dim is not None:
        feat = _pad_or_trim_feat(feat, desired_dim)
    return feat, texts



def feature_OAG(layer_data, graph):
    feature = {}
    times   = {}
    indxs   = {}
    texts   = []
    cache = getattr(graph, "_feature_cache", None)
    for _type in layer_data:
        if len(layer_data[_type]) == 0:
            continue
        idxs  = np.array(list(layer_data[_type].keys()))
        tims  = np.array(list(layer_data[_type].values()))[:,1]

        feat, maybe_texts = _extract_feature_array(
            idxs=idxs, _type=_type, graph=graph, cache=cache
        )
        feature[_type] = feat
        if _type == "paper":
            texts = maybe_texts

        times[_type]   = tims
        indxs[_type]   = idxs

    return feature, times, indxs, texts
