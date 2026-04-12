import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, uniform
from torch_geometric.utils import softmax
import math


_ROTA_MODES = {'none', 'k', 'qk'}


def _normalize_rota_mode(mode, field_name):
    if mode is None:
        return None
    mode = str(mode).lower()
    if mode not in _ROTA_MODES:
        raise ValueError(f"Unsupported {field_name}: {mode}. Expected one of {_ROTA_MODES}.")
    return mode


def _resolve_ablation_profile(conv_name, rtpa_mode, ssta_mode, rtpa_keep, ssta_keep):
    """
    Resolve table-friendly aliases to concrete conv names + branch settings.
    This keeps runtime selection hyper-parameter only.
    """
    alias = str(conv_name).lower()
    presets = {
        # Table row aliases.
        'rtta': dict(conv='rtpa_rota', rtpa_mode='qk', rtpa_keep=True),
        # DRSF variants with fixed branch setup R111/S001:
        # RoTA-Q(R/S)=1/0, RoTA-K(R/S)=1/0, RTE-R=1, RTE-S=1.
        'lf': dict(conv='drsf_rota', rtpa_mode='qk', ssta_mode='none', rtpa_keep=True, ssta_keep=True),
        'sf': dict(conv='drsf_sf_rota', rtpa_mode='qk', ssta_mode='none', rtpa_keep=True, ssta_keep=True),
        'cgsf': dict(conv='drsf_cgsf_rota', rtpa_mode='qk', ssta_mode='none', rtpa_keep=True, ssta_keep=True),
        'cglf': dict(conv='drsf_cglf_rota', rtpa_mode='qk', ssta_mode='none', rtpa_keep=True, ssta_keep=True),
        'ngsf': dict(conv='drsf_ngsf_rota', rtpa_mode='qk', ssta_mode='none', rtpa_keep=True, ssta_keep=True),
        'nglf': dict(conv='drsf_nglf_rota', rtpa_mode='qk', ssta_mode='none', rtpa_keep=True, ssta_keep=True),
        'tc-ngsf': dict(conv='drsf_tcngsf_rota', rtpa_mode='qk', ssta_mode='none', rtpa_keep=True, ssta_keep=True),
        'tc_ngsf': dict(conv='drsf_tcngsf_rota', rtpa_mode='qk', ssta_mode='none', rtpa_keep=True, ssta_keep=True),
        'tcngsf': dict(conv='drsf_tcngsf_rota', rtpa_mode='qk', ssta_mode='none', rtpa_keep=True, ssta_keep=True),
        'tc-nglf': dict(conv='drsf_tcnglf_rota', rtpa_mode='qk', ssta_mode='none', rtpa_keep=True, ssta_keep=True),
        'tc_nglf': dict(conv='drsf_tcnglf_rota', rtpa_mode='qk', ssta_mode='none', rtpa_keep=True, ssta_keep=True),
        'tcnglf': dict(conv='drsf_tcnglf_rota', rtpa_mode='qk', ssta_mode='none', rtpa_keep=True, ssta_keep=True),
        'ta-nglf': dict(conv='drsf_tanglf_rota', rtpa_mode='qk', ssta_mode='none', rtpa_keep=True, ssta_keep=True),
        'ta_nglf': dict(conv='drsf_tanglf_rota', rtpa_mode='qk', ssta_mode='none', rtpa_keep=True, ssta_keep=True),
        'tanglf': dict(conv='drsf_tanglf_rota', rtpa_mode='qk', ssta_mode='none', rtpa_keep=True, ssta_keep=True),
        'ta-ngsf': dict(conv='drsf_tangsf_rota', rtpa_mode='qk', ssta_mode='none', rtpa_keep=True, ssta_keep=True),
        'ta_ngsf': dict(conv='drsf_tangsf_rota', rtpa_mode='qk', ssta_mode='none', rtpa_keep=True, ssta_keep=True),
        'tangsf': dict(conv='drsf_tangsf_rota', rtpa_mode='qk', ssta_mode='none', rtpa_keep=True, ssta_keep=True),
    }
    preset = presets.get(alias)
    if preset is None:
        return conv_name, rtpa_mode, ssta_mode, rtpa_keep, ssta_keep
    resolved_name = preset.get('conv', conv_name)
    resolved_rtpa_mode = preset.get('rtpa_mode', rtpa_mode)
    resolved_ssta_mode = preset.get('ssta_mode', ssta_mode)
    resolved_rtpa_keep = preset.get('rtpa_keep', rtpa_keep)
    resolved_ssta_keep = preset.get('ssta_keep', ssta_keep)
    return resolved_name, resolved_rtpa_mode, resolved_ssta_mode, resolved_rtpa_keep, resolved_ssta_keep


class RTTAConv(MessagePassing):
    """
    Faster RTTA attention block with equivalent math.

    It precomputes per-edge Q/K/V by node type first, then applies
    relation-specific transforms in a single loop over relation ids.
    """
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, dropout = 0.2, use_norm = True, use_RTE = True, **kwargs):
        super(RTTAConv, self).__init__(node_dim=0, aggr='add', **kwargs)

        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.num_types     = num_types
        self.num_relations = num_relations
        self.total_rel     = num_types * num_relations * num_types
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        self.use_norm      = use_norm
        self.use_RTE       = use_RTE
        self.att           = None

        self.k_linears   = nn.ModuleList()
        self.q_linears   = nn.ModuleList()
        self.v_linears   = nn.ModuleList()
        self.a_linears   = nn.ModuleList()
        self.norms       = nn.ModuleList()

        for t in range(num_types):
            self.k_linears.append(nn.Linear(in_dim,   out_dim))
            self.q_linears.append(nn.Linear(in_dim,   out_dim))
            self.v_linears.append(nn.Linear(in_dim,   out_dim))
            self.a_linears.append(nn.Linear(out_dim,  out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri   = nn.Parameter(torch.ones(num_relations, self.n_heads))
        self.relation_att   = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg   = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.skip           = nn.Parameter(torch.ones(num_types))
        self.drop           = nn.Dropout(dropout)

        if self.use_RTE:
            self.emb = RelTemporalEncoding(in_dim)

        glorot(self.relation_att)
        glorot(self.relation_msg)

    def forward(self, node_inp, node_type, edge_index, edge_type, edge_time):
        return self.propagate(edge_index, node_inp=node_inp, node_type=node_type, \
                              edge_type=edge_type, edge_time=edge_time)

    def message(self, edge_index_i, node_inp_i, node_inp_j, node_type_i, node_type_j, edge_type, edge_time):
        data_size = edge_index_i.size(0)
        q_all = torch.zeros(data_size, self.n_heads, self.d_k).to(node_inp_i.device)
        k_all = torch.zeros(data_size, self.n_heads, self.d_k).to(node_inp_i.device)
        v_all = torch.zeros(data_size, self.n_heads, self.d_k).to(node_inp_i.device)

        # Target-side q per edge.
        for target_type in range(self.num_types):
            target_mask = (node_type_i == int(target_type))
            if target_mask.sum() == 0:
                continue
            q_all[target_mask] = self.q_linears[target_type](node_inp_i[target_mask]).view(-1, self.n_heads, self.d_k)

        # Source-side k/v per edge; RTE can be applied once globally.
        source_input = self.emb(node_inp_j, edge_time) if self.use_RTE else node_inp_j
        for source_type in range(self.num_types):
            source_mask = (node_type_j == int(source_type))
            if source_mask.sum() == 0:
                continue
            src_vec = source_input[source_mask]
            k_all[source_mask] = self.k_linears[source_type](src_vec).view(-1, self.n_heads, self.d_k)
            v_all[source_mask] = self.v_linears[source_type](src_vec).view(-1, self.n_heads, self.d_k)

        # Relation transforms only need one loop over relation ids.
        res_att = torch.zeros(data_size, self.n_heads).to(node_inp_i.device)
        res_msg = torch.zeros(data_size, self.n_heads, self.d_k).to(node_inp_i.device)
        for relation_type in range(self.num_relations):
            rel_mask = (edge_type == int(relation_type))
            if rel_mask.sum() == 0:
                continue
            q_rel = q_all[rel_mask]
            k_rel = k_all[rel_mask]
            v_rel = v_all[rel_mask]
            k_rel = torch.bmm(k_rel.transpose(1, 0), self.relation_att[relation_type]).transpose(1, 0)
            v_rel = torch.bmm(v_rel.transpose(1, 0), self.relation_msg[relation_type]).transpose(1, 0)
            res_att[rel_mask] = (q_rel * k_rel).sum(dim=-1) * self.relation_pri[relation_type] / self.sqrt_dk
            res_msg[rel_mask] = v_rel

        # TCA: normalize separately for each source type bucket.
        self.att = torch.zeros_like(res_att)
        for source_type in range(self.num_types):
            type_mask = (node_type_j == int(source_type))
            if type_mask.sum() == 0:
                continue
            self.att[type_mask] = softmax(res_att[type_mask], edge_index_i[type_mask])

        res = res_msg * self.att.view(-1, self.n_heads, 1)
        del q_all, k_all, v_all, res_att, res_msg
        return res.view(-1, self.out_dim)

    def update(self, aggr_out, node_inp, node_type):
        aggr_out = F.gelu(aggr_out)
        res = torch.zeros(aggr_out.size(0), self.out_dim).to(node_inp.device)
        for target_type in range(self.num_types):
            idx = (node_type == int(target_type))
            if idx.sum() == 0:
                continue
            trans_out = self.drop(self.a_linears[target_type](aggr_out[idx]))
            alpha = torch.sigmoid(self.skip[target_type])
            if self.use_norm:
                res[idx] = self.norms[target_type](trans_out * alpha + node_inp[idx] * (1 - alpha))
            else:
                res[idx] = trans_out * alpha + node_inp[idx] * (1 - alpha)
        return res

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_types={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_relations)


class RTTAConv_RoTA(RTTAConv):
    """
    RTTA with rotary temporal phase injected into attention Q/K.
    """
    def __init__(
        self,
        in_dim,
        out_dim,
        num_types,
        num_relations,
        n_heads,
        dropout = 0.2,
        use_norm = True,
        use_RTE = True,
        rota_mode = 'qk',
        rota_keep_rte = True,
        **kwargs
    ):
        super(RTTAConv_RoTA, self).__init__(
            in_dim, out_dim, num_types, num_relations, n_heads,
            dropout=dropout, use_norm=use_norm, use_RTE=use_RTE, **kwargs
        )
        if rota_mode not in _ROTA_MODES:
            raise ValueError(f"Unsupported rota_mode: {rota_mode}")
        self.rota_mode = rota_mode
        self.rota_keep_rte = rota_keep_rte
        rope_dim = self.d_k - (self.d_k % 2)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, rope_dim, 2).float() / max(rope_dim, 1)))
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)

    def _apply_rota(self, x, phase):
        rope_dim = self.d_k - (self.d_k % 2)
        if rope_dim == 0:
            return x
        freqs = phase.unsqueeze(-1) * self.rope_inv_freq.to(device=x.device, dtype=x.dtype).unsqueeze(0)
        cos = torch.cos(freqs).unsqueeze(1)
        sin = torch.sin(freqs).unsqueeze(1)
        x_main = x[..., :rope_dim].view(x.size(0), x.size(1), rope_dim // 2, 2)
        x_even = x_main[..., 0]
        x_odd  = x_main[..., 1]
        y_even = x_even * cos - x_odd * sin
        y_odd  = x_even * sin + x_odd * cos
        y_main = torch.stack([y_even, y_odd], dim=-1).view(x.size(0), x.size(1), rope_dim)
        if rope_dim == self.d_k:
            return y_main
        return torch.cat([y_main, x[..., rope_dim:]], dim=-1)

    def message(self, edge_index_i, node_inp_i, node_inp_j, node_type_i, node_type_j, edge_type, edge_time):
        data_size = edge_index_i.size(0)
        q_all = torch.zeros(data_size, self.n_heads, self.d_k).to(node_inp_i.device)
        k_all = torch.zeros(data_size, self.n_heads, self.d_k).to(node_inp_i.device)
        v_all = torch.zeros(data_size, self.n_heads, self.d_k).to(node_inp_i.device)

        for target_type in range(self.num_types):
            target_mask = (node_type_i == int(target_type))
            if target_mask.sum() == 0:
                continue
            q_all[target_mask] = self.q_linears[target_type](node_inp_i[target_mask]).view(-1, self.n_heads, self.d_k)

        source_input = self.emb(node_inp_j, edge_time) if (self.use_RTE and self.rota_keep_rte) else node_inp_j
        for source_type in range(self.num_types):
            source_mask = (node_type_j == int(source_type))
            if source_mask.sum() == 0:
                continue
            src_vec = source_input[source_mask]
            k_all[source_mask] = self.k_linears[source_type](src_vec).view(-1, self.n_heads, self.d_k)
            v_all[source_mask] = self.v_linears[source_type](src_vec).view(-1, self.n_heads, self.d_k)

        res_att = torch.zeros(data_size, self.n_heads).to(node_inp_i.device)
        res_msg = torch.zeros(data_size, self.n_heads, self.d_k).to(node_inp_i.device)
        for relation_type in range(self.num_relations):
            rel_mask = (edge_type == int(relation_type))
            if rel_mask.sum() == 0:
                continue
            q_rel = q_all[rel_mask]
            k_rel = k_all[rel_mask]
            v_rel = v_all[rel_mask]
            k_rel = torch.bmm(k_rel.transpose(1, 0), self.relation_att[relation_type]).transpose(1, 0)
            v_rel = torch.bmm(v_rel.transpose(1, 0), self.relation_msg[relation_type]).transpose(1, 0)

            phase = edge_time[rel_mask].to(dtype=q_rel.dtype)
            if self.rota_mode == 'qk':
                q_rel = self._apply_rota(q_rel, 0.5 * phase)
                k_rel = self._apply_rota(k_rel, -0.5 * phase)
            elif self.rota_mode == 'k':
                k_rel = self._apply_rota(k_rel, -phase)

            res_att[rel_mask] = (q_rel * k_rel).sum(dim=-1) * self.relation_pri[relation_type] / self.sqrt_dk
            res_msg[rel_mask] = v_rel

        self.att = torch.zeros_like(res_att)
        for source_type in range(self.num_types):
            type_mask = (node_type_j == int(source_type))
            if type_mask.sum() == 0:
                continue
            self.att[type_mask] = softmax(res_att[type_mask], edge_index_i[type_mask])

        res = res_msg * self.att.view(-1, self.n_heads, 1)
        del q_all, k_all, v_all, res_att, res_msg
        return res.view(-1, self.out_dim)


class SSTAConv(MessagePassing):
    """
    Type-Agnostic Attention (TAA) block in GraphBFF.

    Design goals reflected here:
    - Cross-type attention through shared Q/K/V matrices.
    - Parameter efficiency by sharing attention transforms across edge/node types.

    Pipeline:
      type-specific projection -> shared latent space -> shared Q/K/V attention
      -> aggregation over all incoming neighbors.
    """
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, dropout = 0.2, use_norm = True, use_RTE = True, **kwargs):
        super(SSTAConv, self).__init__(node_dim=0, aggr='add', **kwargs)

        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.num_types     = num_types
        self.num_relations = num_relations
        self.total_rel     = num_types * num_relations * num_types
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        self.use_norm      = use_norm
        self.use_RTE       = use_RTE
        self.att           = None
        self.ssta_att      = None
        # Type projection to a shared latent space.
        self.ssta_type_linears = nn.ModuleList()
        self.a_linears   = nn.ModuleList()
        self.norms       = nn.ModuleList()
        for t in range(num_types):
            self.ssta_type_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        # Shared attention matrices across all node/edge types.
        self.ssta_q_linear   = nn.Linear(out_dim, out_dim)
        self.ssta_k_linear   = nn.Linear(out_dim, out_dim)
        self.ssta_v_linear   = nn.Linear(out_dim, out_dim)
        self.skip           = nn.Parameter(torch.ones(num_types))
        self.drop           = nn.Dropout(dropout)

        if self.use_RTE:
            self.emb = RelTemporalEncoding(in_dim)

    def forward(self, node_inp, node_type, edge_index, edge_type, edge_time):
        return self.propagate(edge_index, node_inp=node_inp, node_type=node_type, \
                              edge_type=edge_type, edge_time=edge_time)

    def message(self, edge_index_i, node_inp_i, node_inp_j, node_type_i, node_type_j, edge_type, edge_time):
        data_size = edge_index_i.size(0)
        tgt_latent = torch.zeros(data_size, self.out_dim).to(node_inp_i.device)
        src_latent = torch.zeros(data_size, self.out_dim).to(node_inp_i.device)

        # Project each node type into a common latent space first.
        # This keeps type-aware input adaptation while enabling shared attention later.
        for source_type in range(self.num_types):
            source_mask = (node_type_j == int(source_type))
            if source_mask.sum() == 0:
                continue
            source_vec = node_inp_j[source_mask]
            if self.use_RTE:
                source_vec = self.emb(source_vec, edge_time[source_mask])
            src_latent[source_mask] = self.ssta_type_linears[source_type](source_vec)

        for target_type in range(self.num_types):
            target_mask = (node_type_i == int(target_type))
            if target_mask.sum() == 0:
                continue
            tgt_latent[target_mask] = self.ssta_type_linears[target_type](node_inp_i[target_mask])

        # Type-agnostic attention: all incoming neighbors share the same Q/K/V transforms.
        q_mat = self.ssta_q_linear(tgt_latent).view(-1, self.n_heads, self.d_k)
        k_mat = self.ssta_k_linear(src_latent).view(-1, self.n_heads, self.d_k)
        v_mat = self.ssta_v_linear(src_latent).view(-1, self.n_heads, self.d_k)
        att_score = (q_mat * k_mat).sum(dim=-1) / self.sqrt_dk
        self.ssta_att = softmax(att_score, edge_index_i)
        self.att = self.ssta_att
        res_msg = v_mat * self.ssta_att.view(-1, self.n_heads, 1)

        del tgt_latent, src_latent
        return res_msg.view(-1, self.out_dim)

    def update(self, aggr_out, node_inp, node_type):
        aggr_out = F.gelu(aggr_out)
        res = torch.zeros(aggr_out.size(0), self.out_dim).to(node_inp.device)
        for target_type in range(self.num_types):
            idx = (node_type == int(target_type))
            if idx.sum() == 0:
                continue
            trans_out = self.drop(self.a_linears[target_type](aggr_out[idx]))
            alpha = torch.sigmoid(self.skip[target_type])
            if self.use_norm:
                res[idx] = self.norms[target_type](trans_out * alpha + node_inp[idx] * (1 - alpha))
            else:
                res[idx] = trans_out * alpha + node_inp[idx] * (1 - alpha)
        return res

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_types={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_relations)


class SSTAConv_RoTA(SSTAConv):
    """
    SSTA variant with rotary temporal attention on Q/K.

    Keep SSTA's shared-space design, but inject edge-time phase directly
    into Q/K via RoPE-style 2D rotations before dot-product attention.
    """
    def __init__(
        self,
        in_dim,
        out_dim,
        num_types,
        num_relations,
        n_heads,
        dropout = 0.2,
        use_norm = True,
        use_RTE = True,
        rota_mode = 'qk',
        rota_keep_rte = True,
        **kwargs
    ):
        super(SSTAConv_RoTA, self).__init__(
            in_dim, out_dim, num_types, num_relations, n_heads,
            dropout=dropout, use_norm=use_norm, use_RTE=use_RTE, **kwargs
        )
        if rota_mode not in _ROTA_MODES:
            raise ValueError(f"Unsupported rota_mode: {rota_mode}")
        self.rota_mode = rota_mode
        self.rota_keep_rte = rota_keep_rte
        # Per-head rotary frequencies on the attention channel dimension.
        # We rotate pairs of channels (even/odd). If d_k is odd, the last
        # channel is kept unchanged.
        rope_dim = self.d_k - (self.d_k % 2)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, rope_dim, 2).float() / max(rope_dim, 1)))
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)

    def _apply_rota(self, x, phase):
        """
        x: [E, H, d_k], phase: [E]
        Apply RoPE rotation to paired channels using edge-wise phase.
        """
        rope_dim = self.d_k - (self.d_k % 2)
        if rope_dim == 0:
            return x
        # [E, rope_dim/2]
        freqs = phase.unsqueeze(-1) * self.rope_inv_freq.to(device=x.device, dtype=x.dtype).unsqueeze(0)
        cos = torch.cos(freqs).unsqueeze(1)  # [E, 1, rope_dim/2]
        sin = torch.sin(freqs).unsqueeze(1)  # [E, 1, rope_dim/2]

        x_main = x[..., :rope_dim].view(x.size(0), x.size(1), rope_dim // 2, 2)
        x_even = x_main[..., 0]
        x_odd  = x_main[..., 1]
        y_even = x_even * cos - x_odd * sin
        y_odd  = x_even * sin + x_odd * cos
        y_main = torch.stack([y_even, y_odd], dim=-1).view(x.size(0), x.size(1), rope_dim)
        if rope_dim == self.d_k:
            return y_main
        # Keep untouched tail channel when d_k is odd.
        return torch.cat([y_main, x[..., rope_dim:]], dim=-1)

    def message(self, edge_index_i, node_inp_i, node_inp_j, node_type_i, node_type_j, edge_type, edge_time):
        data_size = edge_index_i.size(0)
        tgt_latent = torch.zeros(data_size, self.out_dim).to(node_inp_i.device)
        src_latent = torch.zeros(data_size, self.out_dim).to(node_inp_i.device)

        for source_type in range(self.num_types):
            source_mask = (node_type_j == int(source_type))
            if source_mask.sum() == 0:
                continue
            source_vec = node_inp_j[source_mask]
            if self.use_RTE and self.rota_keep_rte:
                source_vec = self.emb(source_vec, edge_time[source_mask])
            src_latent[source_mask] = self.ssta_type_linears[source_type](source_vec)

        for target_type in range(self.num_types):
            target_mask = (node_type_i == int(target_type))
            if target_mask.sum() == 0:
                continue
            tgt_latent[target_mask] = self.ssta_type_linears[target_type](node_inp_i[target_mask])

        q_mat = self.ssta_q_linear(tgt_latent).view(-1, self.n_heads, self.d_k)
        k_mat = self.ssta_k_linear(src_latent).view(-1, self.n_heads, self.d_k)
        v_mat = self.ssta_v_linear(src_latent).view(-1, self.n_heads, self.d_k)

        # RoTA switch:
        # - qk: symmetric relative-phase rotation (q:+dt/2, k:-dt/2)
        # - k: rotate only key with edge phase
        phase = edge_time.to(dtype=q_mat.dtype)
        if self.rota_mode == 'qk':
            q_mat = self._apply_rota(q_mat, 0.5 * phase)
            k_mat = self._apply_rota(k_mat, -0.5 * phase)
        elif self.rota_mode == 'k':
            k_mat = self._apply_rota(k_mat, -phase)

        att_score = (q_mat * k_mat).sum(dim=-1) / self.sqrt_dk
        self.ssta_att = softmax(att_score, edge_index_i)
        self.att = self.ssta_att
        res_msg = v_mat * self.ssta_att.view(-1, self.n_heads, 1)
        del tgt_latent, src_latent
        return res_msg.view(-1, self.out_dim)


#Compare GRAM, Metrics, Speed, Memory, etc.
class DRSFConv(MessagePassing):
    """
    Aggressive-speed variant of DRSF fusion block.

    Main difference: it precomputes per-edge Q/K/V tensors by node type first, then
    applies relation-specific transforms in a single loop over relation ids.
    This removes the source_type x target_type x relation_type nested loops.
    """
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, dropout = 0.2, use_norm = True, use_RTE = True, **kwargs):
        super(DRSFConv, self).__init__(node_dim=0, aggr='add', **kwargs)

        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.num_types     = num_types
        self.num_relations = num_relations
        self.total_rel     = num_types * num_relations * num_types
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        self.use_norm      = use_norm
        self.use_RTE       = use_RTE
        self.att           = None
        self.ssta_att      = None
        self.k_linears   = nn.ModuleList()
        self.q_linears   = nn.ModuleList()
        self.v_linears   = nn.ModuleList()
        self.ssta_type_linears = nn.ModuleList()
        self.a_linears   = nn.ModuleList()
        self.norms       = nn.ModuleList()

        for t in range(num_types):
            self.k_linears.append(nn.Linear(in_dim,   out_dim))
            self.q_linears.append(nn.Linear(in_dim,   out_dim))
            self.v_linears.append(nn.Linear(in_dim,   out_dim))
            self.ssta_type_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim,  out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri   = nn.Parameter(torch.ones(num_relations, self.n_heads))
        self.relation_att   = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg   = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.skip           = nn.Parameter(torch.ones(num_types))
        self.drop           = nn.Dropout(dropout)
        self.ssta_q_linear   = nn.Linear(out_dim, out_dim)
        self.ssta_k_linear   = nn.Linear(out_dim, out_dim)
        self.ssta_v_linear   = nn.Linear(out_dim, out_dim)
        self.fuse_linear    = nn.Linear(out_dim * 2, out_dim)
        if self.use_RTE:
            self.emb = RelTemporalEncoding(in_dim)

        glorot(self.relation_att)
        glorot(self.relation_msg)

    def forward(self, node_inp, node_type, edge_index, edge_type, edge_time):
        return self.propagate(edge_index, node_inp=node_inp, node_type=node_type, \
                              edge_type=edge_type, edge_time=edge_time)

    def message(self, edge_index_i, node_inp_i, node_inp_j, node_type_i, node_type_j, edge_type, edge_time):
        data_size = edge_index_i.size(0)

        # Precompute per-edge RTTA q/k/v and SSTA latent tensors by node type.
        rtpa_q = torch.zeros(data_size, self.n_heads, self.d_k).to(node_inp_i.device)
        rtpa_k = torch.zeros(data_size, self.n_heads, self.d_k).to(node_inp_i.device)
        rtpa_v = torch.zeros(data_size, self.n_heads, self.d_k).to(node_inp_i.device)
        ssta_tgt_latent = torch.zeros(data_size, self.out_dim).to(node_inp_i.device)
        ssta_src_latent = torch.zeros(data_size, self.out_dim).to(node_inp_i.device)

        for target_type in range(self.num_types):
            target_mask = (node_type_i == int(target_type))
            if target_mask.sum() == 0:
                continue
            target_vec = node_inp_i[target_mask]
            rtpa_q[target_mask] = self.q_linears[target_type](target_vec).view(-1, self.n_heads, self.d_k)
            ssta_tgt_latent[target_mask] = self.ssta_type_linears[target_type](target_vec)

        for source_type in range(self.num_types):
            source_mask = (node_type_j == int(source_type))
            if source_mask.sum() == 0:
                continue
            source_vec = node_inp_j[source_mask]
            if self.use_RTE:
                source_vec = self.emb(source_vec, edge_time[source_mask])
            rtpa_k[source_mask] = self.k_linears[source_type](source_vec).view(-1, self.n_heads, self.d_k)
            rtpa_v[source_mask] = self.v_linears[source_type](source_vec).view(-1, self.n_heads, self.d_k)
            ssta_src_latent[source_mask] = self.ssta_type_linears[source_type](source_vec)

        # Relation-specific transform now loops only over relation ids.
        rtpa_res_att = torch.zeros(data_size, self.n_heads).to(node_inp_i.device)
        rtpa_res_msg = torch.zeros(data_size, self.n_heads, self.d_k).to(node_inp_i.device)
        for relation_type in range(self.num_relations):
            rel_mask = (edge_type == int(relation_type))
            if rel_mask.sum() == 0:
                continue
            q_rel = rtpa_q[rel_mask]
            k_rel = rtpa_k[rel_mask]
            v_rel = rtpa_v[rel_mask]
            k_rel = torch.bmm(k_rel.transpose(1, 0), self.relation_att[relation_type]).transpose(1, 0)
            v_rel = torch.bmm(v_rel.transpose(1, 0), self.relation_msg[relation_type]).transpose(1, 0)
            rtpa_res_att[rel_mask] = (q_rel * k_rel).sum(dim=-1) * self.relation_pri[relation_type] / self.sqrt_dk
            rtpa_res_msg[rel_mask] = v_rel

        # RTTA: per-source-type normalized sparse softmax.
        self.att = torch.zeros_like(rtpa_res_att)
        for source_type in range(self.num_types):
            type_mask = (node_type_j == int(source_type))
            if type_mask.sum() == 0:
                continue
            self.att[type_mask] = softmax(rtpa_res_att[type_mask], edge_index_i[type_mask])
        rtpa_res = rtpa_res_msg * self.att.view(-1, self.n_heads, 1)

        # SSTA: shared Q/K/V over all incoming neighbors.
        q_mat = self.ssta_q_linear(ssta_tgt_latent).view(-1, self.n_heads, self.d_k)
        k_mat = self.ssta_k_linear(ssta_src_latent).view(-1, self.n_heads, self.d_k)
        v_mat = self.ssta_v_linear(ssta_src_latent).view(-1, self.n_heads, self.d_k)
        ssta_score = (q_mat * k_mat).sum(dim=-1) / self.sqrt_dk
        self.ssta_att = softmax(ssta_score, edge_index_i)
        ssta_res = v_mat * self.ssta_att.view(-1, self.n_heads, 1)

        del rtpa_q, rtpa_k, rtpa_v, rtpa_res_att, rtpa_res_msg, ssta_tgt_latent, ssta_src_latent
        return torch.cat([rtpa_res.view(-1, self.out_dim), ssta_res.view(-1, self.out_dim)], dim=-1)

    def update(self, aggr_out, node_inp, node_type):
        rtpa_out, ssta_out = torch.split(aggr_out, self.out_dim, dim=-1)
        rtpa_out = F.gelu(rtpa_out)
        ssta_out = F.gelu(ssta_out)
        fused_out = self.drop(self.fuse_linear(torch.cat([rtpa_out, ssta_out], dim=-1)))
        return self._typewise_residual(fused_out, node_inp, node_type)

    def _typewise_residual(self, fused_out, node_inp, node_type):
        res = torch.zeros(fused_out.size(0), self.out_dim, device=node_inp.device, dtype=fused_out.dtype)
        for target_type in range(self.num_types):
            idx = (node_type == int(target_type))
            if idx.sum() == 0:
                continue
            alpha = torch.sigmoid(self.skip[target_type])
            if self.use_norm:
                res[idx] = self.norms[target_type](fused_out[idx] * alpha + node_inp[idx] * (1 - alpha))
            else:
                res[idx] = fused_out[idx] * alpha + node_inp[idx] * (1 - alpha)
        return res

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_types={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_relations)


class DRSFConv_RoTA(DRSFConv):
    """
    DRSF with RoTA on both RTTA (relation-aware) and SSTA (shared) branches.
    """
    def __init__(
        self,
        in_dim,
        out_dim,
        num_types,
        num_relations,
        n_heads,
        dropout = 0.2,
        use_norm = True,
        use_RTE = True,
        rota_mode = 'qk',
        rota_keep_rte = True,
        rtpa_rota_mode = None,
        ssta_rota_mode = None,
        rtpa_rota_keep_rte = None,
        ssta_rota_keep_rte = None,
        **kwargs
    ):
        super(DRSFConv_RoTA, self).__init__(
            in_dim, out_dim, num_types, num_relations, n_heads,
            dropout=dropout, use_norm=use_norm, use_RTE=use_RTE, **kwargs
        )
        self.rtpa_rota_mode = rtpa_rota_mode if rtpa_rota_mode is not None else rota_mode
        self.ssta_rota_mode = ssta_rota_mode if ssta_rota_mode is not None else rota_mode
        self.rtpa_rota_keep_rte = rota_keep_rte if rtpa_rota_keep_rte is None else rtpa_rota_keep_rte
        self.ssta_rota_keep_rte = rota_keep_rte if ssta_rota_keep_rte is None else ssta_rota_keep_rte
        if self.rtpa_rota_mode not in _ROTA_MODES:
            raise ValueError(f"Unsupported rtpa_rota_mode: {self.rtpa_rota_mode}")
        if self.ssta_rota_mode not in _ROTA_MODES:
            raise ValueError(f"Unsupported ssta_rota_mode: {self.ssta_rota_mode}")
        rope_dim = self.d_k - (self.d_k % 2)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, rope_dim, 2).float() / max(rope_dim, 1)))
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)

    def _apply_rota(self, x, phase):
        rope_dim = self.d_k - (self.d_k % 2)
        if rope_dim == 0:
            return x
        freqs = phase.unsqueeze(-1) * self.rope_inv_freq.to(device=x.device, dtype=x.dtype).unsqueeze(0)
        cos = torch.cos(freqs).unsqueeze(1)
        sin = torch.sin(freqs).unsqueeze(1)
        x_main = x[..., :rope_dim].view(x.size(0), x.size(1), rope_dim // 2, 2)
        x_even = x_main[..., 0]
        x_odd  = x_main[..., 1]
        y_even = x_even * cos - x_odd * sin
        y_odd  = x_even * sin + x_odd * cos
        y_main = torch.stack([y_even, y_odd], dim=-1).view(x.size(0), x.size(1), rope_dim)
        if rope_dim == self.d_k:
            return y_main
        return torch.cat([y_main, x[..., rope_dim:]], dim=-1)

    def message(self, edge_index_i, node_inp_i, node_inp_j, node_type_i, node_type_j, edge_type, edge_time):
        data_size = edge_index_i.size(0)

        rtpa_q = torch.zeros(data_size, self.n_heads, self.d_k).to(node_inp_i.device)
        rtpa_k = torch.zeros(data_size, self.n_heads, self.d_k).to(node_inp_i.device)
        rtpa_v = torch.zeros(data_size, self.n_heads, self.d_k).to(node_inp_i.device)
        ssta_tgt_latent = torch.zeros(data_size, self.out_dim).to(node_inp_i.device)
        ssta_src_latent = torch.zeros(data_size, self.out_dim).to(node_inp_i.device)

        for target_type in range(self.num_types):
            target_mask = (node_type_i == int(target_type))
            if target_mask.sum() == 0:
                continue
            target_vec = node_inp_i[target_mask]
            rtpa_q[target_mask] = self.q_linears[target_type](target_vec).view(-1, self.n_heads, self.d_k)
            ssta_tgt_latent[target_mask] = self.ssta_type_linears[target_type](target_vec)

        for source_type in range(self.num_types):
            source_mask = (node_type_j == int(source_type))
            if source_mask.sum() == 0:
                continue
            source_vec = node_inp_j[source_mask]
            source_time = edge_time[source_mask]
            rtpa_source_vec = source_vec
            ssta_source_vec = source_vec
            if self.use_RTE and self.rtpa_rota_keep_rte:
                rtpa_source_vec = self.emb(rtpa_source_vec, source_time)
            if self.use_RTE and self.ssta_rota_keep_rte:
                ssta_source_vec = self.emb(ssta_source_vec, source_time)
            rtpa_k[source_mask] = self.k_linears[source_type](rtpa_source_vec).view(-1, self.n_heads, self.d_k)
            rtpa_v[source_mask] = self.v_linears[source_type](rtpa_source_vec).view(-1, self.n_heads, self.d_k)
            ssta_src_latent[source_mask] = self.ssta_type_linears[source_type](ssta_source_vec)

        rtpa_res_att = torch.zeros(data_size, self.n_heads).to(node_inp_i.device)
        rtpa_res_msg = torch.zeros(data_size, self.n_heads, self.d_k).to(node_inp_i.device)
        for relation_type in range(self.num_relations):
            rel_mask = (edge_type == int(relation_type))
            if rel_mask.sum() == 0:
                continue
            q_rel = rtpa_q[rel_mask]
            k_rel = rtpa_k[rel_mask]
            v_rel = rtpa_v[rel_mask]
            k_rel = torch.bmm(k_rel.transpose(1, 0), self.relation_att[relation_type]).transpose(1, 0)
            v_rel = torch.bmm(v_rel.transpose(1, 0), self.relation_msg[relation_type]).transpose(1, 0)

            phase = edge_time[rel_mask].to(dtype=q_rel.dtype)
            if self.rtpa_rota_mode == 'qk':
                q_rel = self._apply_rota(q_rel, 0.5 * phase)
                k_rel = self._apply_rota(k_rel, -0.5 * phase)
            elif self.rtpa_rota_mode == 'k':
                k_rel = self._apply_rota(k_rel, -phase)

            rtpa_res_att[rel_mask] = (q_rel * k_rel).sum(dim=-1) * self.relation_pri[relation_type] / self.sqrt_dk
            rtpa_res_msg[rel_mask] = v_rel

        self.att = torch.zeros_like(rtpa_res_att)
        for source_type in range(self.num_types):
            type_mask = (node_type_j == int(source_type))
            if type_mask.sum() == 0:
                continue
            self.att[type_mask] = softmax(rtpa_res_att[type_mask], edge_index_i[type_mask])
        rtpa_res = rtpa_res_msg * self.att.view(-1, self.n_heads, 1)

        q_mat = self.ssta_q_linear(ssta_tgt_latent).view(-1, self.n_heads, self.d_k)
        k_mat = self.ssta_k_linear(ssta_src_latent).view(-1, self.n_heads, self.d_k)
        v_mat = self.ssta_v_linear(ssta_src_latent).view(-1, self.n_heads, self.d_k)
        phase = edge_time.to(dtype=q_mat.dtype)
        if self.ssta_rota_mode == 'qk':
            q_mat = self._apply_rota(q_mat, 0.5 * phase)
            k_mat = self._apply_rota(k_mat, -0.5 * phase)
        elif self.ssta_rota_mode == 'k':
            k_mat = self._apply_rota(k_mat, -phase)
        ssta_score = (q_mat * k_mat).sum(dim=-1) / self.sqrt_dk
        self.ssta_att = softmax(ssta_score, edge_index_i)
        ssta_res = v_mat * self.ssta_att.view(-1, self.n_heads, 1)

        del rtpa_q, rtpa_k, rtpa_v, rtpa_res_att, rtpa_res_msg, ssta_tgt_latent, ssta_src_latent
        return torch.cat([rtpa_res.view(-1, self.out_dim), ssta_res.view(-1, self.out_dim)], dim=-1)


class DRSFSFConv(DRSFConv):
    """
    SF (Sum Fusion): dropout(GELU(rtpa) + GELU(ssta)).
    """
    def update(self, aggr_out, node_inp, node_type):
        rtpa_out, ssta_out = torch.split(aggr_out, self.out_dim, dim=-1)
        rtpa_out = F.gelu(rtpa_out)
        ssta_out = F.gelu(ssta_out)
        fused_out = self.drop(rtpa_out + ssta_out)
        return self._typewise_residual(fused_out, node_inp, node_type)


class DRSFSFConv_RoTA(DRSFConv_RoTA):
    """
    SF with RoTA on both branches.
    """
    def update(self, aggr_out, node_inp, node_type):
        rtpa_out, ssta_out = torch.split(aggr_out, self.out_dim, dim=-1)
        rtpa_out = F.gelu(rtpa_out)
        ssta_out = F.gelu(ssta_out)
        fused_out = self.drop(rtpa_out + ssta_out)
        return self._typewise_residual(fused_out, node_inp, node_type)


class DRSFCGLFConv(DRSFConv):
    """
    CGLF (Competitive Gated Linear Fusion): feature-wise two-way softmax gate.
    """
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, dropout = 0.2, use_norm = True, use_RTE = True, **kwargs):
        super(DRSFCGLFConv, self).__init__(
            in_dim, out_dim, num_types, num_relations, n_heads,
            dropout=dropout, use_norm=use_norm, use_RTE=use_RTE, **kwargs
        )
        self.fuse_gate = None

    def update(self, aggr_out, node_inp, node_type):
        rtpa_out, ssta_out = torch.split(aggr_out, self.out_dim, dim=-1)
        rtpa_out = F.gelu(rtpa_out)
        ssta_out = F.gelu(ssta_out)

        comp = torch.softmax(torch.stack([rtpa_out, ssta_out], dim=1), dim=1)
        beta_rtpa = comp[:, 0, :]
        beta_ssta = comp[:, 1, :]
        self.fuse_gate = comp

        fused_out = self.drop(
            self.fuse_linear(torch.cat([beta_rtpa * rtpa_out, beta_ssta * ssta_out], dim=-1))
        )
        return self._typewise_residual(fused_out, node_inp, node_type)


class DRSFCGLFConv_RoTA(DRSFConv_RoTA):
    """
    CGLF with RoTA on both branches.
    """
    def __init__(
        self,
        in_dim,
        out_dim,
        num_types,
        num_relations,
        n_heads,
        dropout = 0.2,
        use_norm = True,
        use_RTE = True,
        rota_mode = 'qk',
        rota_keep_rte = True,
        rtpa_rota_mode = None,
        ssta_rota_mode = None,
        rtpa_rota_keep_rte = None,
        ssta_rota_keep_rte = None,
        **kwargs
    ):
        super(DRSFCGLFConv_RoTA, self).__init__(
            in_dim, out_dim, num_types, num_relations, n_heads,
            dropout=dropout, use_norm=use_norm, use_RTE=use_RTE,
            rota_mode=rota_mode, rota_keep_rte=rota_keep_rte,
            rtpa_rota_mode=rtpa_rota_mode, ssta_rota_mode=ssta_rota_mode,
            rtpa_rota_keep_rte=rtpa_rota_keep_rte, ssta_rota_keep_rte=ssta_rota_keep_rte,
            **kwargs
        )
        self.fuse_gate = None

    def update(self, aggr_out, node_inp, node_type):
        rtpa_out, ssta_out = torch.split(aggr_out, self.out_dim, dim=-1)
        rtpa_out = F.gelu(rtpa_out)
        ssta_out = F.gelu(ssta_out)

        comp = torch.softmax(torch.stack([rtpa_out, ssta_out], dim=1), dim=1)
        beta_rtpa = comp[:, 0, :]
        beta_ssta = comp[:, 1, :]
        self.fuse_gate = comp

        fused_out = self.drop(
            self.fuse_linear(torch.cat([beta_rtpa * rtpa_out, beta_ssta * ssta_out], dim=-1))
        )
        return self._typewise_residual(fused_out, node_inp, node_type)


class DRSFCGSFConv(DRSFConv):
    """
    CGSF (Competitive Gated Sum Fusion): same competitive gate as CGLF,
    but sums gated branches instead of linear fusion.
    """
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, dropout = 0.2, use_norm = True, use_RTE = True, **kwargs):
        super(DRSFCGSFConv, self).__init__(
            in_dim, out_dim, num_types, num_relations, n_heads,
            dropout=dropout, use_norm=use_norm, use_RTE=use_RTE, **kwargs
        )
        self.fuse_gate = None

    def update(self, aggr_out, node_inp, node_type):
        rtpa_out, ssta_out = torch.split(aggr_out, self.out_dim, dim=-1)
        rtpa_out = F.gelu(rtpa_out)
        ssta_out = F.gelu(ssta_out)

        comp = torch.softmax(torch.stack([rtpa_out, ssta_out], dim=1), dim=1)
        beta_rtpa = comp[:, 0, :]
        beta_ssta = comp[:, 1, :]
        self.fuse_gate = comp

        fused_out = self.drop(beta_rtpa * rtpa_out + beta_ssta * ssta_out)
        return self._typewise_residual(fused_out, node_inp, node_type)


class DRSFCGSFConv_RoTA(DRSFConv_RoTA):
    """
    CGSF with RoTA on both branches.
    """
    def __init__(
        self,
        in_dim,
        out_dim,
        num_types,
        num_relations,
        n_heads,
        dropout = 0.2,
        use_norm = True,
        use_RTE = True,
        rota_mode = 'qk',
        rota_keep_rte = True,
        rtpa_rota_mode = None,
        ssta_rota_mode = None,
        rtpa_rota_keep_rte = None,
        ssta_rota_keep_rte = None,
        **kwargs
    ):
        super(DRSFCGSFConv_RoTA, self).__init__(
            in_dim, out_dim, num_types, num_relations, n_heads,
            dropout=dropout, use_norm=use_norm, use_RTE=use_RTE,
            rota_mode=rota_mode, rota_keep_rte=rota_keep_rte,
            rtpa_rota_mode=rtpa_rota_mode, ssta_rota_mode=ssta_rota_mode,
            rtpa_rota_keep_rte=rtpa_rota_keep_rte, ssta_rota_keep_rte=ssta_rota_keep_rte,
            **kwargs
        )
        self.fuse_gate = None

    def update(self, aggr_out, node_inp, node_type):
        rtpa_out, ssta_out = torch.split(aggr_out, self.out_dim, dim=-1)
        rtpa_out = F.gelu(rtpa_out)
        ssta_out = F.gelu(ssta_out)

        comp = torch.softmax(torch.stack([rtpa_out, ssta_out], dim=1), dim=1)
        beta_rtpa = comp[:, 0, :]
        beta_ssta = comp[:, 1, :]
        self.fuse_gate = comp

        fused_out = self.drop(beta_rtpa * rtpa_out + beta_ssta * ssta_out)
        return self._typewise_residual(fused_out, node_inp, node_type)


class DRSFNGLFConv(DRSFConv):
    """
    NGLF (Non-Competitive Gated Linear Fusion): independent sigmoid dual gates.
    """
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, dropout = 0.2, use_norm = True, use_RTE = True, **kwargs):
        super(DRSFNGLFConv, self).__init__(
            in_dim, out_dim, num_types, num_relations, n_heads,
            dropout=dropout, use_norm=use_norm, use_RTE=use_RTE, **kwargs
        )
        self.nglf_gate_fc1 = nn.Linear(out_dim * 2, out_dim)
        self.nglf_gate_fc2 = nn.Linear(out_dim, out_dim * 2)
        self.fuse_gate = None

    def update(self, aggr_out, node_inp, node_type):
        rtpa_out, ssta_out = torch.split(aggr_out, self.out_dim, dim=-1)
        rtpa_out = F.gelu(rtpa_out)
        ssta_out = F.gelu(ssta_out)

        gate_inp = torch.cat([rtpa_out, ssta_out], dim=-1)
        gate_h = F.gelu(self.nglf_gate_fc1(gate_inp))
        dual_gate = torch.sigmoid(self.nglf_gate_fc2(gate_h))
        g_rtpa, g_ssta = torch.split(dual_gate, self.out_dim, dim=-1)
        self.fuse_gate = dual_gate

        fused_out = self.drop(
            self.fuse_linear(torch.cat([g_rtpa * rtpa_out, g_ssta * ssta_out], dim=-1))
        )
        return self._typewise_residual(fused_out, node_inp, node_type)


class DRSFNGLFConv_RoTA(DRSFConv_RoTA):
    """
    NGLF with RoTA on both branches.
    """
    def __init__(
        self,
        in_dim,
        out_dim,
        num_types,
        num_relations,
        n_heads,
        dropout = 0.2,
        use_norm = True,
        use_RTE = True,
        rota_mode = 'qk',
        rota_keep_rte = True,
        rtpa_rota_mode = None,
        ssta_rota_mode = None,
        rtpa_rota_keep_rte = None,
        ssta_rota_keep_rte = None,
        **kwargs
    ):
        super(DRSFNGLFConv_RoTA, self).__init__(
            in_dim, out_dim, num_types, num_relations, n_heads,
            dropout=dropout, use_norm=use_norm, use_RTE=use_RTE,
            rota_mode=rota_mode, rota_keep_rte=rota_keep_rte,
            rtpa_rota_mode=rtpa_rota_mode, ssta_rota_mode=ssta_rota_mode,
            rtpa_rota_keep_rte=rtpa_rota_keep_rte, ssta_rota_keep_rte=ssta_rota_keep_rte,
            **kwargs
        )
        self.nglf_gate_fc1 = nn.Linear(out_dim * 2, out_dim)
        self.nglf_gate_fc2 = nn.Linear(out_dim, out_dim * 2)
        self.fuse_gate = None

    def update(self, aggr_out, node_inp, node_type):
        rtpa_out, ssta_out = torch.split(aggr_out, self.out_dim, dim=-1)
        rtpa_out = F.gelu(rtpa_out)
        ssta_out = F.gelu(ssta_out)

        gate_inp = torch.cat([rtpa_out, ssta_out], dim=-1)
        gate_h = F.gelu(self.nglf_gate_fc1(gate_inp))
        dual_gate = torch.sigmoid(self.nglf_gate_fc2(gate_h))
        g_rtpa, g_ssta = torch.split(dual_gate, self.out_dim, dim=-1)
        self.fuse_gate = dual_gate

        fused_out = self.drop(
            self.fuse_linear(torch.cat([g_rtpa * rtpa_out, g_ssta * ssta_out], dim=-1))
        )
        return self._typewise_residual(fused_out, node_inp, node_type)


class DRSFNGSFConv(DRSFConv):
    """
    NGSF (Non-Competitive Gated Sum Fusion): same dual sigmoid gates as NGLF,
    but sums gated branches instead of linear fusion.
    """
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, dropout = 0.2, use_norm = True, use_RTE = True, **kwargs):
        super(DRSFNGSFConv, self).__init__(
            in_dim, out_dim, num_types, num_relations, n_heads,
            dropout=dropout, use_norm=use_norm, use_RTE=use_RTE, **kwargs
        )
        self.nglf_gate_fc1 = nn.Linear(out_dim * 2, out_dim)
        self.nglf_gate_fc2 = nn.Linear(out_dim, out_dim * 2)
        self.fuse_gate = None

    def update(self, aggr_out, node_inp, node_type):
        rtpa_out, ssta_out = torch.split(aggr_out, self.out_dim, dim=-1)
        rtpa_out = F.gelu(rtpa_out)
        ssta_out = F.gelu(ssta_out)

        gate_inp = torch.cat([rtpa_out, ssta_out], dim=-1)
        gate_h = F.gelu(self.nglf_gate_fc1(gate_inp))
        dual_gate = torch.sigmoid(self.nglf_gate_fc2(gate_h))
        g_rtpa, g_ssta = torch.split(dual_gate, self.out_dim, dim=-1)
        self.fuse_gate = dual_gate

        fused_out = self.drop(g_rtpa * rtpa_out + g_ssta * ssta_out)
        return self._typewise_residual(fused_out, node_inp, node_type)


class DRSFNGSFConv_RoTA(DRSFConv_RoTA):
    """
    NGSF with RoTA on both branches.
    """
    def __init__(
        self,
        in_dim,
        out_dim,
        num_types,
        num_relations,
        n_heads,
        dropout = 0.2,
        use_norm = True,
        use_RTE = True,
        rota_mode = 'qk',
        rota_keep_rte = True,
        rtpa_rota_mode = None,
        ssta_rota_mode = None,
        rtpa_rota_keep_rte = None,
        ssta_rota_keep_rte = None,
        **kwargs
    ):
        super(DRSFNGSFConv_RoTA, self).__init__(
            in_dim, out_dim, num_types, num_relations, n_heads,
            dropout=dropout, use_norm=use_norm, use_RTE=use_RTE,
            rota_mode=rota_mode, rota_keep_rte=rota_keep_rte,
            rtpa_rota_mode=rtpa_rota_mode, ssta_rota_mode=ssta_rota_mode,
            rtpa_rota_keep_rte=rtpa_rota_keep_rte, ssta_rota_keep_rte=ssta_rota_keep_rte,
            **kwargs
        )
        self.nglf_gate_fc1 = nn.Linear(out_dim * 2, out_dim)
        self.nglf_gate_fc2 = nn.Linear(out_dim, out_dim * 2)
        self.fuse_gate = None

    def update(self, aggr_out, node_inp, node_type):
        rtpa_out, ssta_out = torch.split(aggr_out, self.out_dim, dim=-1)
        rtpa_out = F.gelu(rtpa_out)
        ssta_out = F.gelu(ssta_out)

        gate_inp = torch.cat([rtpa_out, ssta_out], dim=-1)
        gate_h = F.gelu(self.nglf_gate_fc1(gate_inp))
        dual_gate = torch.sigmoid(self.nglf_gate_fc2(gate_h))
        g_rtpa, g_ssta = torch.split(dual_gate, self.out_dim, dim=-1)
        self.fuse_gate = dual_gate

        fused_out = self.drop(g_rtpa * rtpa_out + g_ssta * ssta_out)
        return self._typewise_residual(fused_out, node_inp, node_type)


class DRSFTC_NGLFConv(DRSFConv):
    """
    TC-NGLF: NGLF conditioned by target-type embeddings.
    """
    def __init__(
        self,
        in_dim,
        out_dim,
        num_types,
        num_relations,
        n_heads,
        dropout = 0.2,
        use_norm = True,
        use_RTE = True,
        type_emb_dim = None,
        **kwargs
    ):
        super(DRSFTC_NGLFConv, self).__init__(
            in_dim, out_dim, num_types, num_relations, n_heads,
            dropout=dropout, use_norm=use_norm, use_RTE=use_RTE, **kwargs
        )
        self.type_emb_dim = max(1, out_dim // 4) if type_emb_dim is None else type_emb_dim
        self.type_embeddings = nn.Embedding(num_types, self.type_emb_dim)
        self.tc_nglf_gate_fc1 = nn.Linear(out_dim * 2 + self.type_emb_dim, out_dim)
        self.tc_nglf_gate_fc2 = nn.Linear(out_dim, out_dim * 2)
        self.fuse_gate = None

    def update(self, aggr_out, node_inp, node_type):
        rtpa_out, ssta_out = torch.split(aggr_out, self.out_dim, dim=-1)
        rtpa_out = F.gelu(rtpa_out)
        ssta_out = F.gelu(ssta_out)

        type_vec = self.type_embeddings(node_type.long())
        gate_inp = torch.cat([rtpa_out, ssta_out, type_vec], dim=-1)
        gate_h = F.gelu(self.tc_nglf_gate_fc1(gate_inp))
        dual_gate = torch.sigmoid(self.tc_nglf_gate_fc2(gate_h))
        g_rtpa, g_ssta = torch.split(dual_gate, self.out_dim, dim=-1)
        self.fuse_gate = dual_gate

        fused_out = self.drop(
            self.fuse_linear(torch.cat([g_rtpa * rtpa_out, g_ssta * ssta_out], dim=-1))
        )
        return self._typewise_residual(fused_out, node_inp, node_type)


class DRSFTC_NGLFConv_RoTA(DRSFConv_RoTA):
    """
    TC-NGLF with RoTA on both branches.
    """
    def __init__(
        self,
        in_dim,
        out_dim,
        num_types,
        num_relations,
        n_heads,
        dropout = 0.2,
        use_norm = True,
        use_RTE = True,
        rota_mode = 'qk',
        rota_keep_rte = True,
        rtpa_rota_mode = None,
        ssta_rota_mode = None,
        rtpa_rota_keep_rte = None,
        ssta_rota_keep_rte = None,
        type_emb_dim = None,
        **kwargs
    ):
        super(DRSFTC_NGLFConv_RoTA, self).__init__(
            in_dim, out_dim, num_types, num_relations, n_heads,
            dropout=dropout, use_norm=use_norm, use_RTE=use_RTE,
            rota_mode=rota_mode, rota_keep_rte=rota_keep_rte,
            rtpa_rota_mode=rtpa_rota_mode, ssta_rota_mode=ssta_rota_mode,
            rtpa_rota_keep_rte=rtpa_rota_keep_rte, ssta_rota_keep_rte=ssta_rota_keep_rte,
            **kwargs
        )
        self.type_emb_dim = max(1, out_dim // 4) if type_emb_dim is None else type_emb_dim
        self.type_embeddings = nn.Embedding(num_types, self.type_emb_dim)
        self.tc_nglf_gate_fc1 = nn.Linear(out_dim * 2 + self.type_emb_dim, out_dim)
        self.tc_nglf_gate_fc2 = nn.Linear(out_dim, out_dim * 2)
        self.fuse_gate = None

    def update(self, aggr_out, node_inp, node_type):
        rtpa_out, ssta_out = torch.split(aggr_out, self.out_dim, dim=-1)
        rtpa_out = F.gelu(rtpa_out)
        ssta_out = F.gelu(ssta_out)

        type_vec = self.type_embeddings(node_type.long())
        gate_inp = torch.cat([rtpa_out, ssta_out, type_vec], dim=-1)
        gate_h = F.gelu(self.tc_nglf_gate_fc1(gate_inp))
        dual_gate = torch.sigmoid(self.tc_nglf_gate_fc2(gate_h))
        g_rtpa, g_ssta = torch.split(dual_gate, self.out_dim, dim=-1)
        self.fuse_gate = dual_gate

        fused_out = self.drop(
            self.fuse_linear(torch.cat([g_rtpa * rtpa_out, g_ssta * ssta_out], dim=-1))
        )
        return self._typewise_residual(fused_out, node_inp, node_type)


class DRSFTC_NGSFConv(DRSFConv):
    """
    TC-NGSF: TC-NGLF gates conditioned by target-type embeddings, with
    sum fusion on gated branches.
    """
    def __init__(
        self,
        in_dim,
        out_dim,
        num_types,
        num_relations,
        n_heads,
        dropout = 0.2,
        use_norm = True,
        use_RTE = True,
        type_emb_dim = None,
        **kwargs
    ):
        super(DRSFTC_NGSFConv, self).__init__(
            in_dim, out_dim, num_types, num_relations, n_heads,
            dropout=dropout, use_norm=use_norm, use_RTE=use_RTE, **kwargs
        )
        self.type_emb_dim = max(1, out_dim // 4) if type_emb_dim is None else type_emb_dim
        self.type_embeddings = nn.Embedding(num_types, self.type_emb_dim)
        self.tc_nglf_gate_fc1 = nn.Linear(out_dim * 2 + self.type_emb_dim, out_dim)
        self.tc_nglf_gate_fc2 = nn.Linear(out_dim, out_dim * 2)
        self.fuse_gate = None

    def update(self, aggr_out, node_inp, node_type):
        rtpa_out, ssta_out = torch.split(aggr_out, self.out_dim, dim=-1)
        rtpa_out = F.gelu(rtpa_out)
        ssta_out = F.gelu(ssta_out)

        type_vec = self.type_embeddings(node_type.long())
        gate_inp = torch.cat([rtpa_out, ssta_out, type_vec], dim=-1)
        gate_h = F.gelu(self.tc_nglf_gate_fc1(gate_inp))
        dual_gate = torch.sigmoid(self.tc_nglf_gate_fc2(gate_h))
        g_rtpa, g_ssta = torch.split(dual_gate, self.out_dim, dim=-1)
        self.fuse_gate = dual_gate

        fused_out = self.drop(g_rtpa * rtpa_out + g_ssta * ssta_out)
        return self._typewise_residual(fused_out, node_inp, node_type)


class DRSFTC_NGSFConv_RoTA(DRSFConv_RoTA):
    """
    TC-NGSF with RoTA on both branches.
    """
    def __init__(
        self,
        in_dim,
        out_dim,
        num_types,
        num_relations,
        n_heads,
        dropout = 0.2,
        use_norm = True,
        use_RTE = True,
        rota_mode = 'qk',
        rota_keep_rte = True,
        rtpa_rota_mode = None,
        ssta_rota_mode = None,
        rtpa_rota_keep_rte = None,
        ssta_rota_keep_rte = None,
        type_emb_dim = None,
        **kwargs
    ):
        super(DRSFTC_NGSFConv_RoTA, self).__init__(
            in_dim, out_dim, num_types, num_relations, n_heads,
            dropout=dropout, use_norm=use_norm, use_RTE=use_RTE,
            rota_mode=rota_mode, rota_keep_rte=rota_keep_rte,
            rtpa_rota_mode=rtpa_rota_mode, ssta_rota_mode=ssta_rota_mode,
            rtpa_rota_keep_rte=rtpa_rota_keep_rte, ssta_rota_keep_rte=ssta_rota_keep_rte,
            **kwargs
        )
        self.type_emb_dim = max(1, out_dim // 4) if type_emb_dim is None else type_emb_dim
        self.type_embeddings = nn.Embedding(num_types, self.type_emb_dim)
        self.tc_nglf_gate_fc1 = nn.Linear(out_dim * 2 + self.type_emb_dim, out_dim)
        self.tc_nglf_gate_fc2 = nn.Linear(out_dim, out_dim * 2)
        self.fuse_gate = None

    def update(self, aggr_out, node_inp, node_type):
        rtpa_out, ssta_out = torch.split(aggr_out, self.out_dim, dim=-1)
        rtpa_out = F.gelu(rtpa_out)
        ssta_out = F.gelu(ssta_out)

        type_vec = self.type_embeddings(node_type.long())
        gate_inp = torch.cat([rtpa_out, ssta_out, type_vec], dim=-1)
        gate_h = F.gelu(self.tc_nglf_gate_fc1(gate_inp))
        dual_gate = torch.sigmoid(self.tc_nglf_gate_fc2(gate_h))
        g_rtpa, g_ssta = torch.split(dual_gate, self.out_dim, dim=-1)
        self.fuse_gate = dual_gate

        fused_out = self.drop(g_rtpa * rtpa_out + g_ssta * ssta_out)
        return self._typewise_residual(fused_out, node_inp, node_type)


class DRSFTANGLFConv(DRSFConv):
    """
    TA-NGLF (Type-Aware Non-Competitive Gated Linear Fusion): NGLF gates
    followed by a target-type-specific linear projection (a_linears) before
    dropout, mirroring the per-type output transform in the SSTA/RTTA update.
    """
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, dropout = 0.2, use_norm = True, use_RTE = True, **kwargs):
        super(DRSFTANGLFConv, self).__init__(
            in_dim, out_dim, num_types, num_relations, n_heads,
            dropout=dropout, use_norm=use_norm, use_RTE=use_RTE, **kwargs
        )
        self.nglf_gate_fc1 = nn.Linear(out_dim * 2, out_dim)
        self.nglf_gate_fc2 = nn.Linear(out_dim, out_dim * 2)
        self.fuse_gate = None

    def update(self, aggr_out, node_inp, node_type):
        rtpa_out, ssta_out = torch.split(aggr_out, self.out_dim, dim=-1)
        rtpa_out = F.gelu(rtpa_out)
        ssta_out = F.gelu(ssta_out)

        gate_inp = torch.cat([rtpa_out, ssta_out], dim=-1)
        gate_h = F.gelu(self.nglf_gate_fc1(gate_inp))
        dual_gate = torch.sigmoid(self.nglf_gate_fc2(gate_h))
        g_rtpa, g_ssta = torch.split(dual_gate, self.out_dim, dim=-1)
        self.fuse_gate = dual_gate

        pre_fused = self.fuse_linear(torch.cat([g_rtpa * rtpa_out, g_ssta * ssta_out], dim=-1))
        fused_out = torch.zeros_like(pre_fused)
        for target_type in range(self.num_types):
            idx = (node_type == int(target_type))
            if idx.sum() == 0:
                continue
            fused_out[idx] = self.drop(self.a_linears[target_type](pre_fused[idx]))
        return self._typewise_residual(fused_out, node_inp, node_type)


class DRSFTANGLFConv_RoTA(DRSFConv_RoTA):
    """
    TA-NGLF with RoTA on both branches.
    """
    def __init__(
        self,
        in_dim,
        out_dim,
        num_types,
        num_relations,
        n_heads,
        dropout = 0.2,
        use_norm = True,
        use_RTE = True,
        rota_mode = 'qk',
        rota_keep_rte = True,
        rtpa_rota_mode = None,
        ssta_rota_mode = None,
        rtpa_rota_keep_rte = None,
        ssta_rota_keep_rte = None,
        **kwargs
    ):
        super(DRSFTANGLFConv_RoTA, self).__init__(
            in_dim, out_dim, num_types, num_relations, n_heads,
            dropout=dropout, use_norm=use_norm, use_RTE=use_RTE,
            rota_mode=rota_mode, rota_keep_rte=rota_keep_rte,
            rtpa_rota_mode=rtpa_rota_mode, ssta_rota_mode=ssta_rota_mode,
            rtpa_rota_keep_rte=rtpa_rota_keep_rte, ssta_rota_keep_rte=ssta_rota_keep_rte,
            **kwargs
        )
        self.nglf_gate_fc1 = nn.Linear(out_dim * 2, out_dim)
        self.nglf_gate_fc2 = nn.Linear(out_dim, out_dim * 2)
        self.fuse_gate = None

    def update(self, aggr_out, node_inp, node_type):
        rtpa_out, ssta_out = torch.split(aggr_out, self.out_dim, dim=-1)
        rtpa_out = F.gelu(rtpa_out)
        ssta_out = F.gelu(ssta_out)

        gate_inp = torch.cat([rtpa_out, ssta_out], dim=-1)
        gate_h = F.gelu(self.nglf_gate_fc1(gate_inp))
        dual_gate = torch.sigmoid(self.nglf_gate_fc2(gate_h))
        g_rtpa, g_ssta = torch.split(dual_gate, self.out_dim, dim=-1)
        self.fuse_gate = dual_gate

        pre_fused = self.fuse_linear(torch.cat([g_rtpa * rtpa_out, g_ssta * ssta_out], dim=-1))
        fused_out = torch.zeros_like(pre_fused)
        for target_type in range(self.num_types):
            idx = (node_type == int(target_type))
            if idx.sum() == 0:
                continue
            fused_out[idx] = self.drop(self.a_linears[target_type](pre_fused[idx]))
        return self._typewise_residual(fused_out, node_inp, node_type)


class DRSFTANGSFConv(DRSFConv):
    """
    TA-NGSF (Type-Aware Non-Competitive Gated Sum Fusion): NGSF gates
    followed by a target-type-specific linear projection (a_linears) before
    dropout, mirroring the per-type output transform in the SSTA/RTTA update.
    """
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, dropout = 0.2, use_norm = True, use_RTE = True, **kwargs):
        super(DRSFTANGSFConv, self).__init__(
            in_dim, out_dim, num_types, num_relations, n_heads,
            dropout=dropout, use_norm=use_norm, use_RTE=use_RTE, **kwargs
        )
        self.nglf_gate_fc1 = nn.Linear(out_dim * 2, out_dim)
        self.nglf_gate_fc2 = nn.Linear(out_dim, out_dim * 2)
        self.fuse_gate = None

    def update(self, aggr_out, node_inp, node_type):
        rtpa_out, ssta_out = torch.split(aggr_out, self.out_dim, dim=-1)
        rtpa_out = F.gelu(rtpa_out)
        ssta_out = F.gelu(ssta_out)

        gate_inp = torch.cat([rtpa_out, ssta_out], dim=-1)
        gate_h = F.gelu(self.nglf_gate_fc1(gate_inp))
        dual_gate = torch.sigmoid(self.nglf_gate_fc2(gate_h))
        g_rtpa, g_ssta = torch.split(dual_gate, self.out_dim, dim=-1)
        self.fuse_gate = dual_gate

        pre_fused = g_rtpa * rtpa_out + g_ssta * ssta_out
        fused_out = torch.zeros_like(pre_fused)
        for target_type in range(self.num_types):
            idx = (node_type == int(target_type))
            if idx.sum() == 0:
                continue
            fused_out[idx] = self.drop(self.a_linears[target_type](pre_fused[idx]))
        return self._typewise_residual(fused_out, node_inp, node_type)


class DRSFTANGSFConv_RoTA(DRSFConv_RoTA):
    """
    TA-NGSF with RoTA on both branches.
    """
    def __init__(
        self,
        in_dim,
        out_dim,
        num_types,
        num_relations,
        n_heads,
        dropout = 0.2,
        use_norm = True,
        use_RTE = True,
        rota_mode = 'qk',
        rota_keep_rte = True,
        rtpa_rota_mode = None,
        ssta_rota_mode = None,
        rtpa_rota_keep_rte = None,
        ssta_rota_keep_rte = None,
        **kwargs
    ):
        super(DRSFTANGSFConv_RoTA, self).__init__(
            in_dim, out_dim, num_types, num_relations, n_heads,
            dropout=dropout, use_norm=use_norm, use_RTE=use_RTE,
            rota_mode=rota_mode, rota_keep_rte=rota_keep_rte,
            rtpa_rota_mode=rtpa_rota_mode, ssta_rota_mode=ssta_rota_mode,
            rtpa_rota_keep_rte=rtpa_rota_keep_rte, ssta_rota_keep_rte=ssta_rota_keep_rte,
            **kwargs
        )
        self.nglf_gate_fc1 = nn.Linear(out_dim * 2, out_dim)
        self.nglf_gate_fc2 = nn.Linear(out_dim, out_dim * 2)
        self.fuse_gate = None

    def update(self, aggr_out, node_inp, node_type):
        rtpa_out, ssta_out = torch.split(aggr_out, self.out_dim, dim=-1)
        rtpa_out = F.gelu(rtpa_out)
        ssta_out = F.gelu(ssta_out)

        gate_inp = torch.cat([rtpa_out, ssta_out], dim=-1)
        gate_h = F.gelu(self.nglf_gate_fc1(gate_inp))
        dual_gate = torch.sigmoid(self.nglf_gate_fc2(gate_h))
        g_rtpa, g_ssta = torch.split(dual_gate, self.out_dim, dim=-1)
        self.fuse_gate = dual_gate

        pre_fused = g_rtpa * rtpa_out + g_ssta * ssta_out
        fused_out = torch.zeros_like(pre_fused)
        for target_type in range(self.num_types):
            idx = (node_type == int(target_type))
            if idx.sum() == 0:
                continue
            fused_out[idx] = self.drop(self.a_linears[target_type](pre_fused[idx]))
        return self._typewise_residual(fused_out, node_inp, node_type)


class RelTemporalEncoding(nn.Module):
    '''
        Implement the Temporal Encoding (Sinusoid) function.
    '''
    def __init__(self, n_hid, max_len = 240, dropout = 0.2):
        super(RelTemporalEncoding, self).__init__()
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_hid, 2) *
                             -(math.log(10000.0) / n_hid))
        emb = nn.Embedding(max_len, n_hid)
        emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(n_hid)
        emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(n_hid)
        emb.requires_grad = False
        self.emb = emb
        self.lin = nn.Linear(n_hid, n_hid)
    def forward(self, x, t):
        return x + self.lin(self.emb(t))
    
    
    
class GeneralConv(nn.Module):
    def __init__(
        self,
        conv_name,
        in_hid,
        out_hid,
        num_types,
        num_relations,
        n_heads,
        dropout,
        use_norm = True,
        use_RTE = True,
        rota_mode = 'qk',
        rota_keep_rte = True,
        rtpa_rota_mode = None,
        ssta_rota_mode = None,
        rtpa_rota_keep_rte = None,
        ssta_rota_keep_rte = None,
        rtta_rota_mode = None,
        rtta_rota_keep_rte = None,
    ):
        super(GeneralConv, self).__init__()
        if rtpa_rota_mode is None and rtta_rota_mode is not None:
            rtpa_rota_mode = rtta_rota_mode
        if rtpa_rota_keep_rte is None and rtta_rota_keep_rte is not None:
            rtpa_rota_keep_rte = rtta_rota_keep_rte
        global_rota_mode = _normalize_rota_mode(rota_mode, 'rota_mode')
        rtpa_mode = rtpa_rota_mode if rtpa_rota_mode is not None else global_rota_mode
        ssta_mode = ssta_rota_mode if ssta_rota_mode is not None else global_rota_mode
        rtpa_mode = _normalize_rota_mode(rtpa_mode, 'rtpa_rota_mode')
        ssta_mode = _normalize_rota_mode(ssta_mode, 'ssta_rota_mode')
        rtpa_keep = bool(rota_keep_rte if rtpa_rota_keep_rte is None else rtpa_rota_keep_rte)
        ssta_keep = bool(rota_keep_rte if ssta_rota_keep_rte is None else ssta_rota_keep_rte)
        self.raw_conv_name = conv_name
        self.conv_name, rtpa_mode, ssta_mode, rtpa_keep, ssta_keep = _resolve_ablation_profile(
            conv_name, rtpa_mode, ssta_mode, rtpa_keep, ssta_keep
        )
        if self.conv_name == 'rtpa':
            self.base_conv = RTTAConv(in_hid, out_hid, num_types, num_relations, n_heads, dropout, use_norm, use_RTE)
        elif self.conv_name == 'rtpa_rte0':
            self.base_conv = RTTAConv(in_hid, out_hid, num_types, num_relations, n_heads, dropout, use_norm, False)
        elif self.conv_name == 'rtpa_rota':
            self.base_conv = RTTAConv_RoTA(
                in_hid, out_hid, num_types, num_relations, n_heads,
                dropout, use_norm, use_RTE,
                rota_mode=rtpa_mode, rota_keep_rte=rtpa_keep
            )
        elif self.conv_name == 'ssta':
            self.base_conv = SSTAConv(in_hid, out_hid, num_types, num_relations, n_heads, dropout, use_norm, use_RTE)
        elif self.conv_name == 'ssta_rte0':
            self.base_conv = SSTAConv(in_hid, out_hid, num_types, num_relations, n_heads, dropout, use_norm, False)
        elif self.conv_name == 'ssta_rota':
            self.base_conv = SSTAConv_RoTA(
                in_hid, out_hid, num_types, num_relations, n_heads,
                dropout, use_norm, use_RTE,
                rota_mode=ssta_mode, rota_keep_rte=ssta_keep
            )
        elif self.conv_name == 'drsf':
            self.base_conv = DRSFConv(in_hid, out_hid, num_types, num_relations, n_heads, dropout, use_norm, use_RTE)
        elif self.conv_name == 'drsf_rota':
            self.base_conv = DRSFConv_RoTA(
                in_hid, out_hid, num_types, num_relations, n_heads,
                dropout, use_norm, use_RTE,
                rota_mode=global_rota_mode, rota_keep_rte=bool(rota_keep_rte),
                rtpa_rota_mode=rtpa_mode, ssta_rota_mode=ssta_mode,
                rtpa_rota_keep_rte=rtpa_keep, ssta_rota_keep_rte=ssta_keep
            )
        elif self.conv_name in {'drsf_sf'}:
            self.base_conv = DRSFSFConv(
                in_hid, out_hid, num_types, num_relations, n_heads,
                dropout, use_norm, use_RTE
            )
        elif self.conv_name in {'drsf_sf_rota'}:
            self.base_conv = DRSFSFConv_RoTA(
                in_hid, out_hid, num_types, num_relations, n_heads,
                dropout, use_norm, use_RTE,
                rota_mode=global_rota_mode, rota_keep_rte=bool(rota_keep_rte),
                rtpa_rota_mode=rtpa_mode, ssta_rota_mode=ssta_mode,
                rtpa_rota_keep_rte=rtpa_keep, ssta_rota_keep_rte=ssta_keep
            )
        elif self.conv_name in {'drsf_cglf'}:
            self.base_conv = DRSFCGLFConv(
                in_hid, out_hid, num_types, num_relations, n_heads,
                dropout, use_norm, use_RTE
            )
        elif self.conv_name in {'drsf_cglf_rota'}:
            self.base_conv = DRSFCGLFConv_RoTA(
                in_hid, out_hid, num_types, num_relations, n_heads,
                dropout, use_norm, use_RTE,
                rota_mode=global_rota_mode, rota_keep_rte=bool(rota_keep_rte),
                rtpa_rota_mode=rtpa_mode, ssta_rota_mode=ssta_mode,
                rtpa_rota_keep_rte=rtpa_keep, ssta_rota_keep_rte=ssta_keep
            )
        elif self.conv_name in {'drsf_cgsf'}:
            self.base_conv = DRSFCGSFConv(
                in_hid, out_hid, num_types, num_relations, n_heads,
                dropout, use_norm, use_RTE
            )
        elif self.conv_name in {'drsf_cgsf_rota'}:
            self.base_conv = DRSFCGSFConv_RoTA(
                in_hid, out_hid, num_types, num_relations, n_heads,
                dropout, use_norm, use_RTE,
                rota_mode=global_rota_mode, rota_keep_rte=bool(rota_keep_rte),
                rtpa_rota_mode=rtpa_mode, ssta_rota_mode=ssta_mode,
                rtpa_rota_keep_rte=rtpa_keep, ssta_rota_keep_rte=ssta_keep
            )
        elif self.conv_name in {'drsf_nglf'}:
            self.base_conv = DRSFNGLFConv(
                in_hid, out_hid, num_types, num_relations, n_heads,
                dropout, use_norm, use_RTE
            )
        elif self.conv_name in {'drsf_nglf_rota'}:
            self.base_conv = DRSFNGLFConv_RoTA(
                in_hid, out_hid, num_types, num_relations, n_heads,
                dropout, use_norm, use_RTE,
                rota_mode=global_rota_mode, rota_keep_rte=bool(rota_keep_rte),
                rtpa_rota_mode=rtpa_mode, ssta_rota_mode=ssta_mode,
                rtpa_rota_keep_rte=rtpa_keep, ssta_rota_keep_rte=ssta_keep
            )
        elif self.conv_name in {'drsf_ngsf'}:
            self.base_conv = DRSFNGSFConv(
                in_hid, out_hid, num_types, num_relations, n_heads,
                dropout, use_norm, use_RTE
            )
        elif self.conv_name in {'drsf_ngsf_rota'}:
            self.base_conv = DRSFNGSFConv_RoTA(
                in_hid, out_hid, num_types, num_relations, n_heads,
                dropout, use_norm, use_RTE,
                rota_mode=global_rota_mode, rota_keep_rte=bool(rota_keep_rte),
                rtpa_rota_mode=rtpa_mode, ssta_rota_mode=ssta_mode,
                rtpa_rota_keep_rte=rtpa_keep, ssta_rota_keep_rte=ssta_keep
            )
        elif self.conv_name in {'drsf_tcnglf', 'drsf_tc_nglf', 'drsftc_nglf'}:
            self.base_conv = DRSFTC_NGLFConv(
                in_hid, out_hid, num_types, num_relations, n_heads,
                dropout, use_norm, use_RTE
            )
        elif self.conv_name in {'drsf_tcnglf_rota', 'drsf_tc_nglf_rota', 'drsftc_nglf_rota'}:
            self.base_conv = DRSFTC_NGLFConv_RoTA(
                in_hid, out_hid, num_types, num_relations, n_heads,
                dropout, use_norm, use_RTE,
                rota_mode=global_rota_mode, rota_keep_rte=bool(rota_keep_rte),
                rtpa_rota_mode=rtpa_mode, ssta_rota_mode=ssta_mode,
                rtpa_rota_keep_rte=rtpa_keep, ssta_rota_keep_rte=ssta_keep
            )
        elif self.conv_name in {'drsf_tcngsf', 'drsf_tc_ngsf', 'drsftc_ngsf'}:
            self.base_conv = DRSFTC_NGSFConv(
                in_hid, out_hid, num_types, num_relations, n_heads,
                dropout, use_norm, use_RTE
            )
        elif self.conv_name in {'drsf_tcngsf_rota', 'drsf_tc_ngsf_rota', 'drsftc_ngsf_rota'}:
            self.base_conv = DRSFTC_NGSFConv_RoTA(
                in_hid, out_hid, num_types, num_relations, n_heads,
                dropout, use_norm, use_RTE,
                rota_mode=global_rota_mode, rota_keep_rte=bool(rota_keep_rte),
                rtpa_rota_mode=rtpa_mode, ssta_rota_mode=ssta_mode,
                rtpa_rota_keep_rte=rtpa_keep, ssta_rota_keep_rte=ssta_keep
            )
        elif self.conv_name in {'drsf_tanglf'}:
            self.base_conv = DRSFTANGLFConv(
                in_hid, out_hid, num_types, num_relations, n_heads,
                dropout, use_norm, use_RTE
            )
        elif self.conv_name in {'drsf_tanglf_rota'}:
            self.base_conv = DRSFTANGLFConv_RoTA(
                in_hid, out_hid, num_types, num_relations, n_heads,
                dropout, use_norm, use_RTE,
                rota_mode=global_rota_mode, rota_keep_rte=bool(rota_keep_rte),
                rtpa_rota_mode=rtpa_mode, ssta_rota_mode=ssta_mode,
                rtpa_rota_keep_rte=rtpa_keep, ssta_rota_keep_rte=ssta_keep
            )
        elif self.conv_name in {'drsf_tangsf'}:
            self.base_conv = DRSFTANGSFConv(
                in_hid, out_hid, num_types, num_relations, n_heads,
                dropout, use_norm, use_RTE
            )
        elif self.conv_name in {'drsf_tangsf_rota'}:
            self.base_conv = DRSFTANGSFConv_RoTA(
                in_hid, out_hid, num_types, num_relations, n_heads,
                dropout, use_norm, use_RTE,
                rota_mode=global_rota_mode, rota_keep_rte=bool(rota_keep_rte),
                rtpa_rota_mode=rtpa_mode, ssta_rota_mode=ssta_mode,
                rtpa_rota_keep_rte=rtpa_keep, ssta_rota_keep_rte=ssta_keep
            )
        elif self.conv_name == 'gcn':
            self.base_conv = GCNConv(in_hid, out_hid)
        elif self.conv_name == 'gat':
            self.base_conv = GATConv(in_hid, out_hid // n_heads, heads=n_heads)
        else:
            raise ValueError(f"Unsupported conv_name: {self.conv_name}")
    def forward(self, meta_xs, node_type, edge_index, edge_type, edge_time):
        # Unified dispatcher so training scripts can switch conv variants by name.
        if self.conv_name in {
            'rtpa', 'rtpa_rte0', 'rtpa_rota',
            'ssta', 'ssta_rte0', 'ssta_rota',
            'drsf', 'drsf_rota',
            'drsf_sf', 'drsf_sf_rota',
            'drsf_cglf', 'drsf_cglf_rota',
            'drsf_cgsf', 'drsf_cgsf_rota',
            'drsf_nglf', 'drsf_nglf_rota',
            'drsf_ngsf', 'drsf_ngsf_rota',
            'drsf_tcnglf', 'drsf_tc_nglf', 'drsftc_nglf',
            'drsf_tcnglf_rota', 'drsf_tc_nglf_rota', 'drsftc_nglf_rota',
            'drsf_tcngsf', 'drsf_tc_ngsf', 'drsftc_ngsf',
            'drsf_tcngsf_rota', 'drsf_tc_ngsf_rota', 'drsftc_ngsf_rota',
            'drsf_tanglf', 'drsf_tanglf_rota',
            'drsf_tangsf', 'drsf_tangsf_rota'
        }:
            return self.base_conv(meta_xs, node_type, edge_index, edge_type, edge_time)
        elif self.conv_name == 'gcn':
            return self.base_conv(meta_xs, edge_index)
        elif self.conv_name == 'gat':
            return self.base_conv(meta_xs, edge_index)
        raise ValueError(f"Unsupported conv_name: {self.conv_name}")
    

