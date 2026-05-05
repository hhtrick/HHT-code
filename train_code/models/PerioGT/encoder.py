"""
encoder.py — PerioGT Structure Encoder Wrapper (PyTorch Geometric version)
Integrates the PerioGT model into the project framework.
PerioGT always trains full parameters (no frozen/LoRA support).
"""
import os

import torch
import torch.nn as nn
import numpy as np
import yaml
from torch_geometric.nn import global_mean_pool, global_add_pool
from typing import List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler

from .light import LiGhTPredictor, TripletEmbeddingCopoly, NodeSelfAttention, MLP, init_params
from .vocab import Vocab
from .graph_builder import PolyGraphBuilder, preprocess_batch_light, D_ATOM_FEATS, D_BOND_FEATS
from .features import precompute_features
from .aug import periodicity_augment_traverse, generate_multimer_smiles

# PerioGT default hyperparameters (from config.yaml base config)
PERIOGT_DEFAULT_CONFIG = {
    'd_node_feats': 138,
    'd_edge_feats': 14,
    'd_g_feats': 768,
    'd_cl_feats': 256,
    'd_hpath_ratio': 12,
    'n_mol_layers': 12,
    'path_length': 5,
    'n_heads': 12,
    'n_ffn_dense_layers': 2,
}

# PerioGT readout output dimension = d_g_feats * 5 (fp_vn + md_vn + ratio_vn + copolym_type + readout)
PERIOGT_HIDDEN_DIM = PERIOGT_DEFAULT_CONFIG['d_g_feats'] * 5  # 3840

D_FP_FEATS = 1191   # MACCS (167) + ECFP (1024)
D_MD_FEATS = 1613   # Mordred descriptor dimension


class PerioGTEncoder(nn.Module):
    """
    PerioGT structure encoder.
    Converts data entries into molecular graphs and produces high-dimensional vectors via Graph Transformer.
    
    Note: PerioGT always trains full parameters, no frozen backbone or LoRA.
    """

    def __init__(self, pretrained_path: str, d_ratio_feats: int,
                 config_path: Optional[str] = None, device: str = "cuda",
                 max_prompt: int = 10, dropout: float = 0.1,
                 use_prompt: bool = True):
        """
        Args:
            pretrained_path: path to pretrained weights (.pth)
            d_ratio_feats: ratio information vector dimension
            config_path: path to PerioGT config.yaml (optional, uses built-in config by default)
            device: device
            max_prompt: maximum number of periodic prompts
            dropout: dropout ratio
            use_prompt: whether to enable periodic prompt enhancement (when False, only uses initial node chemical features)
        """
        super().__init__()
        self.device = device
        self.max_prompt = max_prompt
        self.use_prompt = use_prompt

        # Load config
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f).get('base', PERIOGT_DEFAULT_CONFIG)
        else:
            config = PERIOGT_DEFAULT_CONFIG

        self.config = config
        d_g_feats = config['d_g_feats']
        self.d_g_feats = d_g_feats
        self.hidden_dim = d_g_feats * 5  # readout dimension

        vocab = Vocab()

        # Build main model
        self.model = LiGhTPredictor(
            d_node_feats=config['d_node_feats'],
            d_edge_feats=config['d_edge_feats'],
            d_g_feats=d_g_feats,
            d_fp_feats=D_FP_FEATS,
            d_md_feats=D_MD_FEATS,
            d_hpath_ratio=config['d_hpath_ratio'],
            n_mol_layers=config['n_mol_layers'],
            path_length=config['path_length'],
            n_heads=config['n_heads'],
            n_ffn_dense_layers=config['n_ffn_dense_layers'],
            input_drop=0, attn_drop=0, feat_drop=0,
            n_node_types=vocab.vocab_size
        )

        # Replace triplet_emb with copolymer version
        self.model.triplet_emb = TripletEmbeddingCopoly(
            d_g_feats, D_FP_FEATS, D_MD_FEATS, d_ratio_feats
        )
        self.model.triplet_emb.apply(lambda m: init_params(m))

        # Load pretrained weights (partial match, ignore missing new modules)
        if os.path.exists(pretrained_path):
            state_dict = torch.load(pretrained_path, map_location='cpu', weights_only=True)
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            key_remap = {
                'edge_emb.virutal_bond_emb.weight': 'edge_emb.virtual_bond_emb.weight',
            }
            for old_key, new_key in key_remap.items():
                if old_key in state_dict:
                    state_dict[new_key] = state_dict.pop(old_key)
            info = self.model.load_state_dict(state_dict, strict=False)
            if info.missing_keys:
                print(f"[PerioGT] Missing keys when loading pretrained weights "
                      f"({len(info.missing_keys)}): {info.missing_keys[:10]}")
            if info.unexpected_keys:
                print(f"[PerioGT] Unexpected keys in pretrained weights "
                      f"({len(info.unexpected_keys)}): {info.unexpected_keys[:10]}")

        # Add attention module (predictor not created here; external regression_head handles prediction)
        self.model.node_attn = NodeSelfAttention(d_g_feats, dropout)
        self.model.node_attn.apply(lambda m: init_params(m))

        # Pass use_prompt flag to underlying model (used in forward_tune)
        self.model.use_prompt = use_prompt

        # Remove prediction heads specific to pretraining stage
        if hasattr(self.model, 'md_predictor'):
            del self.model.md_predictor
        if hasattr(self.model, 'fp_predictor'):
            del self.model.fp_predictor
        if hasattr(self.model, 'node_predictor'):
            del self.model.node_predictor
        if hasattr(self.model, 'cl_projector'):
            del self.model.cl_projector

        # Graph builder
        self.graph_builder = PolyGraphBuilder(
            max_length=config['path_length'],
            n_local_nodes=3,  # fp, md, ratio
            n_global_nodes=1,  # copolymer type
            add_self_loop=True
        )

        # Descriptor normalizer (must be initialized with fit_scaler before first use)
        self.md_scaler = None

    def fit_md_scaler(self, all_md: np.ndarray):
        """Fit StandardScaler with descriptor data"""
        self.md_scaler = StandardScaler()
        self.md_scaler.fit(all_md)

    def get_embedding(self, graphs, fp_1, md_1, fp_2, md_2,
                      ratio_vec_1, ratio_vec_2, global_type):
        """
        Get the final embedding vector (bypassing the prediction head).
        graphs: PyG Batch object (already processed with preprocess_batch_light)
        Returns:
            g_feats: (B, d_g_feats * 5)
        """
        indicators = graphs.vavn
        node_h = self.model.node_emb(graphs.begin_end, indicators)
        edge_h = self.model.edge_emb(graphs.edge, indicators)
        triplet_h = self.model.triplet_emb(
            node_h, edge_h, fp_1, md_1, fp_2, md_2,
            ratio_vec_1, ratio_vec_2, global_type, indicators
        )
        if self.use_prompt:
            triplet_h = self.model.node_attn(
                graphs.prompt.float(), triplet_h, indicators
            )
        triplet_h = self.model.model(graphs, triplet_h)

        fp_vn = (triplet_h[indicators == 1] + triplet_h[indicators == 4]) / 2
        md_vn = (triplet_h[indicators == 2] + triplet_h[indicators == 5]) / 2
        ratio_vn = (triplet_h[indicators == 3] + triplet_h[indicators == 6]) / 2
        copolym_type = triplet_h[indicators == 7]

        # Readout over real (non-virtual) nodes only
        real_mask = (indicators == 0)
        batch_idx = graphs.batch          # [total_N]
        B = int(graphs.num_graphs)
        if self.model.readout_mode == 'mean':
            readout = global_mean_pool(triplet_h[real_mask], batch_idx[real_mask], size=B)
        else:
            readout = global_add_pool(triplet_h[real_mask], batch_idx[real_mask], size=B)

        g_feats = torch.cat([fp_vn, md_vn, ratio_vn, copolym_type, readout], dim=-1)
        return g_feats

