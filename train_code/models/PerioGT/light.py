"""
light.py — LiGhT Graph Transformer Model (PyTorch Geometric version)
"""

import torch
from torch import nn
from torch_geometric.nn import global_mean_pool, global_add_pool
import numpy as np

from .constants import VIRTUAL_ATOM_FEATURE_PLACEHOLDER, VIRTUAL_BOND_FEATURE_PLACEHOLDER


def init_params(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class Residual(nn.Module):
    def __init__(self, d_in_feats, d_out_feats, n_ffn_dense_layers, feat_drop, activation):
        super().__init__()
        self.norm = nn.LayerNorm(d_in_feats)
        self.in_proj = nn.Linear(d_in_feats, d_out_feats)
        self.ffn = MLP(d_out_feats, d_out_feats, n_ffn_dense_layers, activation, d_hidden_feats=d_out_feats * 4)
        self.feat_dropout = nn.Dropout(feat_drop)

    def forward(self, x, y):
        x = x + self.feat_dropout(self.in_proj(y))
        y = self.norm(x)
        y = self.ffn(y)
        y = self.feat_dropout(y)
        x = x + y
        return x


class MLP(nn.Module):
    def __init__(self, d_in_feats, d_out_feats, n_dense_layers, activation, d_hidden_feats=None):
        super().__init__()
        self.n_dense_layers = n_dense_layers
        self.d_hidden_feats = d_out_feats if d_hidden_feats is None else d_hidden_feats
        self.dense_layer_list = nn.ModuleList()
        self.in_proj = nn.Linear(d_in_feats, self.d_hidden_feats)
        for _ in range(self.n_dense_layers - 2):
            self.dense_layer_list.append(nn.Linear(self.d_hidden_feats, self.d_hidden_feats))
        self.out_proj = nn.Linear(self.d_hidden_feats, d_out_feats)
        self.act = activation

    def forward(self, feats):
        feats = self.act(self.in_proj(feats))
        for i in range(self.n_dense_layers - 2):
            feats = self.act(self.dense_layer_list[i](feats))
        feats = self.out_proj(feats)
        return feats


class TripletTransformer(nn.Module):
    def __init__(self, d_feats, d_hpath_ratio, path_length, n_heads,
                 n_ffn_dense_layers, feat_drop=0., attn_drop=0., activation=nn.GELU()):
        super().__init__()
        self.d_feats = d_feats
        self.d_trip_path = d_feats // d_hpath_ratio
        self.path_length = path_length
        self.n_heads = n_heads
        self.head_dim = d_feats // n_heads
        self.scale = d_feats ** (-0.5)

        self.attention_norm = nn.LayerNorm(d_feats)
        self.qkv = nn.Linear(d_feats, d_feats * 3)
        self.node_out_layer = Residual(d_feats, d_feats, n_ffn_dense_layers, feat_drop, activation)
        self.feat_dropout = nn.Dropout(p=feat_drop)
        self.attn_dropout = nn.Dropout(p=attn_drop)

    @staticmethod
    def _edge_softmax(attn: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Numerically stable softmax grouped by destination node.
        attn : [E, n_heads, 1]
        dst  : [E] — destination node index for each edge
        Returns [E, n_heads, 1]
        """
        E, n_heads, _ = attn.shape
        idx = dst.view(-1, 1, 1).expand(E, n_heads, 1)

        # Max per (dst, head) for numerical stability
        max_attn = torch.full((num_nodes, n_heads, 1), float('-inf'),
                              dtype=attn.dtype, device=attn.device)
        max_attn.scatter_reduce_(0, idx, attn, reduce='amax', include_self=True)
        max_attn = max_attn.clamp(min=-1e9)  # guard nodes with no incoming edges

        exp_attn = torch.exp(attn - max_attn[dst])  # [E, n_heads, 1]

        sum_exp = torch.zeros(num_nodes, n_heads, 1, dtype=attn.dtype, device=attn.device)
        sum_exp.scatter_add_(0, idx, exp_attn)

        return exp_attn / (sum_exp[dst] + 1e-9)

    def forward(self, edge_index: torch.Tensor, num_nodes: int,
                triplet_h: torch.Tensor,
                dist_attn: torch.Tensor, path_attn: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_index : [2, E]  (src, dst) in PyG format
            num_nodes  : total number of nodes N
            triplet_h  : [N, d_feats]
            dist_attn  : [E, n_heads]
            path_attn  : [E, n_heads]
        Returns:
            updated triplet_h : [N, d_feats]
        """
        src, dst = edge_index          # each [E]
        E = edge_index.size(1)

        new_triplet_h = self.attention_norm(triplet_h)
        # QKV projection
        qkv = (self.qkv(new_triplet_h)
               .reshape(num_nodes, 3, self.n_heads, self.head_dim)
               .permute(1, 0, 2, 3))            # [3, N, n_heads, head_dim]
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        # Dot-product attention per edge: Q[src] · K[dst]
        q_src = q[src]                           # [E, n_heads, head_dim]
        k_dst = k[dst]                           # [E, n_heads, head_dim]
        node_attn = (q_src * k_dst).sum(dim=-1, keepdim=True)  # [E, n_heads, 1]

        # Combine with distance and path attention
        attn = (node_attn
                + dist_attn.reshape(E, self.n_heads, 1)
                + path_attn.reshape(E, self.n_heads, 1))   # [E, n_heads, 1]

        # Softmax over edges sharing the same dst node
        sa = self.attn_dropout(self._edge_softmax(attn, dst, num_nodes))  # [E, n_heads, 1]

        # Weighted aggregation: v[src] * sa → sum at dst
        v_src = v[src]                           # [E, n_heads, head_dim]
        he = (v_src * sa).reshape(E, self.d_feats)         # [E, d_feats]

        agg_h = torch.zeros(num_nodes, self.d_feats, dtype=he.dtype, device=he.device)
        agg_h.scatter_add_(0, dst.unsqueeze(-1).expand(-1, self.d_feats), he)

        return self.node_out_layer(triplet_h, agg_h)


class LiGhT(nn.Module):
    def __init__(self, d_g_feats, d_hpath_ratio, path_length,
                 n_mol_layers=2, n_heads=4, n_ffn_dense_layers=4,
                 feat_drop=0., attn_drop=0., activation=nn.GELU()):
        super().__init__()
        self.n_mol_layers = n_mol_layers
        self.n_heads = n_heads
        self.path_length = path_length
        self.d_g_feats = d_g_feats
        self.d_trip_path = d_g_feats // d_hpath_ratio

        self.mask_emb = nn.Embedding(1, d_g_feats)
        self.path_len_emb = nn.Embedding(path_length + 1, d_g_feats)
        self.virtual_path_emb = nn.Embedding(1, d_g_feats)
        self.self_loop_emb = nn.Embedding(1, d_g_feats)
        self.dist_attn_layer = nn.Sequential(
            nn.Linear(d_g_feats, d_g_feats), activation, nn.Linear(d_g_feats, n_heads)
        )
        self.trip_fortrans = nn.ModuleList([
            MLP(d_g_feats, self.d_trip_path, 2, activation) for _ in range(path_length)
        ])
        self.path_attn_layer = nn.Sequential(
            nn.Linear(self.d_trip_path, self.d_trip_path), activation, nn.Linear(self.d_trip_path, n_heads)
        )
        self.mol_T_layers = nn.ModuleList([
            TripletTransformer(d_g_feats, d_hpath_ratio, path_length, n_heads,
                               n_ffn_dense_layers, feat_drop, attn_drop, activation)
            for _ in range(n_mol_layers)
        ])
        self.feat_dropout = nn.Dropout(p=feat_drop)

    def _featurize_path(self, path_indices: torch.Tensor,
                        vp: torch.Tensor, sl: torch.Tensor) -> torch.Tensor:
        """
        Args:
            path_indices : [E, path_length]  edge path indices (after preprocess_batch_light)
            vp           : [E] bool — virtual path flag
            sl           : [E] bool — self-loop flag
        Returns:
            path_feats   : [E, d_g_feats]
        """
        mask = (path_indices >= 0).to(torch.int32)       # [E, path_length]
        path_len = torch.sum(mask, dim=-1)                # [E]
        path_feats = self.path_len_emb(path_len)          # [E, d_g_feats]
        path_feats[vp] = self.virtual_path_emb.weight
        path_feats[sl] = self.self_loop_emb.weight
        return path_feats

    def _init_path(self, triplet_h: torch.Tensor,
                   path_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            triplet_h    : [N, d_g_feats]
            path_indices : [E, path_length]  (mutable copy, will be modified in-place)
        Returns:
            path_h       : [E, d_trip_path]
        """
        path_indices = path_indices.clone()
        # VIRTUAL_PATH_INDICATOR values (very negative) → sentinel -1
        path_indices[path_indices < -99] = -1

        # Append a zero row so index -1 maps to a zero vector
        path_h_list = []
        for i in range(self.path_length):
            extended = torch.cat([
                self.trip_fortrans[i](triplet_h),
                torch.zeros(1, self.d_trip_path, device=triplet_h.device, dtype=triplet_h.dtype)
            ], dim=0)                                      # [N+1, d_trip_path]
            path_h_list.append(extended[path_indices[:, i]])   # [E, d_trip_path]

        path_h = torch.stack(path_h_list, dim=-1)         # [E, d_trip_path, path_length]
        mask = (path_indices >= 0).to(triplet_h.dtype)    # [E, path_length]
        path_size = torch.sum(mask, dim=-1, keepdim=True).clamp(min=1.0)  # [E, 1]
        path_h = torch.sum(path_h, dim=-1) / path_size    # [E, d_trip_path]
        return path_h

    def forward(self, data, triplet_h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            data      : PyG Data / Batch with edge_index, path, vp, sl
            triplet_h : [N, d_g_feats]
        Returns:
            updated triplet_h : [N, d_g_feats]
        """
        edge_index = data.edge_index      # [2, E]
        path_indices = data.path          # [E, path_length]
        vp = data.vp                      # [E] bool
        sl = data.sl                      # [E] bool

        N = triplet_h.size(0)

        dist_h = self._featurize_path(path_indices, vp, sl)         # [E, d_g_feats]
        path_h = self._init_path(triplet_h, path_indices)            # [E, d_trip_path]
        dist_attn = self.dist_attn_layer(dist_h)                     # [E, n_heads]
        path_attn = self.path_attn_layer(path_h)                     # [E, n_heads]

        for layer in self.mol_T_layers:
            triplet_h = layer(edge_index, N, triplet_h, dist_attn, path_attn)

        return triplet_h

    def _device(self):
        return next(self.parameters()).device


# ======================== Embedding Module ========================

class AtomEmbedding(nn.Module):
    def __init__(self, d_atom_feats, d_g_feats, input_drop):
        super().__init__()
        self.in_proj = nn.Linear(d_atom_feats, d_g_feats)
        self.virtual_atom_emb = nn.Embedding(1, d_g_feats)
        self.input_dropout = nn.Dropout(input_drop)

    def forward(self, pair_node_feats, indicators):
        pair_node_h = self.in_proj(pair_node_feats)
        pair_node_h[indicators == VIRTUAL_ATOM_FEATURE_PLACEHOLDER, 1, :] = self.virtual_atom_emb.weight
        return torch.sum(self.input_dropout(pair_node_h), dim=-2)


class BondEmbedding(nn.Module):
    def __init__(self, d_bond_feats, d_g_feats, input_drop):
        super().__init__()
        self.in_proj = nn.Linear(d_bond_feats, d_g_feats)
        self.virtual_bond_emb = nn.Embedding(1, d_g_feats)
        self.input_dropout = nn.Dropout(input_drop)

    def forward(self, edge_feats, indicators):
        edge_h = self.in_proj(edge_feats)
        edge_h[indicators == VIRTUAL_BOND_FEATURE_PLACEHOLDER] = self.virtual_bond_emb.weight
        return self.input_dropout(edge_h)


class TripletEmbedding(nn.Module):
    """Original TripletEmbedding for pretraining (single component)"""
    def __init__(self, d_g_feats, d_fp_feats, d_md_feats, activation=nn.GELU()):
        super().__init__()
        self.in_proj = MLP(d_g_feats * 2, d_g_feats, 2, activation)
        self.fp_proj = MLP(d_fp_feats, d_g_feats, 2, activation)
        self.md_proj = MLP(d_md_feats, d_g_feats, 2, activation)

    def forward(self, node_h, edge_h, fp, md, indicators):
        triplet_h = torch.cat([node_h, edge_h], dim=-1)
        triplet_h = self.in_proj(triplet_h)
        triplet_h[indicators == 1] = self.fp_proj(fp)
        triplet_h[indicators == 2] = self.md_proj(md)
        return triplet_h


class TripletEmbeddingCopoly(nn.Module):
    """
    TripletEmbedding for copolymer fine-tuning.
    Supports 2-component copolymers (homopolymer treated as single-component special case).
    """
    def __init__(self, d_g_feats, d_fp_feats, d_md_feats, d_ratio_feats,
                 activation=nn.GELU()):
        super().__init__()
        self.in_proj = MLP(d_g_feats * 2, d_g_feats, 2, activation)
        self.fp_proj = MLP(d_fp_feats, d_g_feats, 2, activation)
        self.md_proj = MLP(d_md_feats, d_g_feats, 2, activation)
        self.ratio_proj = MLP(d_ratio_feats, d_g_feats, 2, activation)
        # Global node embedding (copolymer type / overall info)
        self.global_node_emb = nn.Embedding(3, d_g_feats)

    def forward(self, node_h, edge_h, fp_1, md_1, fp_2, md_2,
                ratio_vec_1, ratio_vec_2, global_type, indicators):
        """
        Args:
            ratio_vec_1: (batch, d_ratio_feats) ratio information vector for component 1
            ratio_vec_2: (batch, d_ratio_feats) ratio information vector for component 2
            global_type: (batch,) long global copolymer type
            indicators: node type labels
        """
        triplet_h = torch.cat([node_h, edge_h], dim=-1)
        triplet_h = self.in_proj(triplet_h)

        triplet_h[indicators == 1] = self.fp_proj(fp_1)
        triplet_h[indicators == 2] = self.md_proj(md_1)
        triplet_h[indicators == 3] = self.ratio_proj(ratio_vec_1)

        triplet_h[indicators == 4] = self.fp_proj(fp_2)
        triplet_h[indicators == 5] = self.md_proj(md_2)
        triplet_h[indicators == 6] = self.ratio_proj(ratio_vec_2)

        triplet_h[indicators == 7] = self.global_node_emb(global_type)
        return triplet_h


class NodeSelfAttention(nn.Module):
    """Periodic prompt attention module"""
    def __init__(self, dim_emb, dropout=0.1):
        super().__init__()
        self.dim_emb = dim_emb
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_emb))
        self.self_attn_1 = nn.MultiheadAttention(embed_dim=dim_emb, num_heads=8, batch_first=True)
        self.self_attn_2 = nn.MultiheadAttention(embed_dim=dim_emb, num_heads=8, batch_first=True)
        self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alpha.data.fill_(0.5)
        self.norm_1 = nn.LayerNorm(dim_emb)
        self.norm_2 = nn.LayerNorm(dim_emb)
        self.linear_1 = nn.Linear(dim_emb, dim_emb)
        self.linear_2 = nn.Linear(dim_emb, dim_emb)
        self.linear_3 = nn.Linear(dim_emb, dim_emb)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.virtual_node_emb = nn.Embedding(1, dim_emb)

    def forward(self, x, triplet_h, indicators):
        # Boolean advanced indexing creates a copy; must clone first, then assign via combined index
        x = x.clone()
        vn_weight = self.virtual_node_emb.weight[0]  # (dim_emb,)
        x[indicators == 3, 0, :] = vn_weight
        x[indicators == 6, 0, :] = vn_weight
        x[indicators == 7, 0, :] = vn_weight

        num_node, seq_len, _ = x.size()
        cls_tokens = self.cls_token.expand(num_node, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        mask = (x.sum(dim=-1) == 0)
        hidden_state, _ = self.self_attn_1(x, x, x, key_padding_mask=mask)
        hidden_state = self.linear_1(hidden_state)
        hidden_state = self.dropout_1(hidden_state)
        x = self.norm_1(hidden_state + x)

        hidden_state, _ = self.self_attn_2(x, x, x, key_padding_mask=mask)
        hidden_state = self.linear_2(hidden_state)
        hidden_state = self.dropout_2(hidden_state)
        hidden_state = self.norm_2(hidden_state + x)

        cls_output = hidden_state[:, 0, :]
        cls_output = self.linear_3(cls_output)
        return triplet_h + self.alpha * cls_output


# ======================== Main Prediction Model ========================

class LiGhTPredictor(nn.Module):
    """
    PerioGT main model.
    Pretraining mode uses standard TripletEmbedding + forward,
    fine-tuning mode uses TripletEmbeddingCopoly + forward_tune.
    """
    def __init__(self, d_node_feats=138, d_edge_feats=14, d_g_feats=768,
                 d_cl_feats=256, d_fp_feats=1191, d_md_feats=1613,
                 d_hpath_ratio=12, n_mol_layers=12, path_length=5,
                 n_heads=12, n_ffn_dense_layers=2,
                 input_drop=0., feat_drop=0., attn_drop=0.,
                 activation=nn.GELU(), n_node_types=1, readout_mode='mean'):
        super().__init__()
        self.d_g_feats = d_g_feats
        self.readout_mode = readout_mode

        self.node_emb = AtomEmbedding(d_node_feats, d_g_feats, input_drop)
        self.edge_emb = BondEmbedding(d_edge_feats, d_g_feats, input_drop)
        self.triplet_emb = TripletEmbedding(d_g_feats, d_fp_feats, d_md_feats, activation)
        self.mask_emb = nn.Embedding(1, d_g_feats)

        self.model = LiGhT(
            d_g_feats, d_hpath_ratio, path_length, n_mol_layers, n_heads,
            n_ffn_dense_layers, feat_drop, attn_drop, activation
        )

        self.node_predictor = nn.Sequential(
            nn.Linear(d_g_feats, d_g_feats), activation, nn.Linear(d_g_feats, n_node_types)
        )
        self.fp_predictor = nn.Sequential(
            nn.Linear(d_g_feats, d_g_feats), activation, nn.Linear(d_g_feats, d_fp_feats)
        )
        self.md_predictor = nn.Sequential(
            nn.Linear(d_g_feats, d_g_feats), activation, nn.Linear(d_g_feats, d_md_feats)
        )
        self.cl_projector = nn.Sequential(
            nn.Linear(d_g_feats * 3, d_g_feats), activation, nn.Linear(d_g_feats, d_cl_feats)
        )

        self.apply(lambda module: init_params(module))

    def forward(self, data, fp, md):
        """
        Pretraining forward pass (PyG version).
        data: PyG Data / Batch with begin_end, edge, vavn, node_mask, path, vp, sl
        """
        indicators = data.vavn
        node_h = self.node_emb(data.begin_end, indicators)
        edge_h = self.edge_emb(data.edge, indicators)
        triplet_h = self.triplet_emb(node_h, edge_h, fp, md, indicators)
        node_mask = data.node_mask if hasattr(data, 'node_mask') else torch.zeros_like(indicators)
        triplet_h[node_mask == 1] = self.mask_emb.weight
        triplet_h = self.model(data, triplet_h)

        fp_vn = triplet_h[indicators == 1]
        md_vn = triplet_h[indicators == 2]

        # Readout over real nodes only (indicators == 0)
        real_mask = (indicators == 0)
        if hasattr(data, 'batch') and data.batch is not None:
            batch_idx = data.batch
            B = int(data.num_graphs)
        else:
            batch_idx = torch.zeros(triplet_h.size(0), dtype=torch.long, device=triplet_h.device)
            B = 1
        if self.readout_mode == 'mean':
            readout = global_mean_pool(triplet_h[real_mask], batch_idx[real_mask], size=B)
        else:
            readout = global_add_pool(triplet_h[real_mask], batch_idx[real_mask], size=B)

        g_feats = torch.cat([fp_vn, md_vn, readout], dim=-1)

        return (self.node_predictor(triplet_h[node_mask >= 1]),
                self.fp_predictor(triplet_h[indicators == 1]),
                self.md_predictor(triplet_h[indicators == 2]),
                self.cl_projector(g_feats))

    def forward_tune(self, data, fp_1, md_1, fp_2, md_2,
                     ratio_vec_1, ratio_vec_2, global_type):
        """
        Fine-tuning forward pass (PyG version).
        data: PyG Data / Batch
        """
        indicators = data.vavn
        node_h = self.node_emb(data.begin_end, indicators)
        edge_h = self.edge_emb(data.edge, indicators)
        triplet_h = self.triplet_emb(
            node_h, edge_h, fp_1, md_1, fp_2, md_2,
            ratio_vec_1, ratio_vec_2, global_type, indicators
        )
        if getattr(self, 'use_prompt', True):
            triplet_h = self.node_attn(data.prompt.float(), triplet_h, indicators)
        triplet_h = self.model(data, triplet_h)

        fp_vn_1 = triplet_h[indicators == 1]
        md_vn_1 = triplet_h[indicators == 2]
        ratio_vn_1 = triplet_h[indicators == 3]
        fp_vn_2 = triplet_h[indicators == 4]
        md_vn_2 = triplet_h[indicators == 5]
        ratio_vn_2 = triplet_h[indicators == 6]
        copolym_type = triplet_h[indicators == 7]

        fp_vn = (fp_vn_1 + fp_vn_2) / 2
        md_vn = (md_vn_1 + md_vn_2) / 2
        ratio_vn = (ratio_vn_1 + ratio_vn_2) / 2

        # Readout over real nodes only (indicators == 0)
        real_mask = (indicators == 0)
        if hasattr(data, 'batch') and data.batch is not None:
            batch_idx = data.batch
            B = int(data.num_graphs)
        else:
            batch_idx = torch.zeros(triplet_h.size(0), dtype=torch.long, device=triplet_h.device)
            B = 1
        if self.readout_mode == 'mean':
            readout = global_mean_pool(triplet_h[real_mask], batch_idx[real_mask], size=B)
        else:
            readout = global_add_pool(triplet_h[real_mask], batch_idx[real_mask], size=B)

        g_feats = torch.cat([fp_vn, md_vn, ratio_vn, copolym_type, readout], dim=-1)
        return g_feats

    def generate_node_emb(self, data, fp, md):
        """
        Generate node embeddings (for periodic prompt computation, PyG version).
        data: PyG Data (single graph)
        """
        bids = data.bid
        indicators = data.vavn
        node_h = self.node_emb(data.begin_end, indicators)
        edge_h = self.edge_emb(data.edge, indicators)
        triplet_h = self.triplet_emb(node_h, edge_h, fp, md, indicators)
        triplet_h = self.model(data, triplet_h)
        return triplet_h, bids

