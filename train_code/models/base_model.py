"""
base_model.py — Core model: pooling layers, MLP projection head, hybrid-stream gating, PyTorch Lightning training module
"""
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from peft import get_peft_model, LoraConfig, TaskType
from typing import List, Optional, Dict, Any
from sklearn.metrics import r2_score

from models.qwen3_4b_instruct_2507 import Qwen3_4B_Instruct_2507
from models.qwen3_4b_thinking_2507 import Qwen3_4B_Thinking_2507
from models.qwen3_4b_base import Qwen3_4B_Base
from models.qwen3_8b_base import Qwen3_8B_Base
from models.qwen3_0_6b_base import Qwen3_0_6B_Base
from models.chemdfm_v1_5_8b import ChemDFM_v1_5_8B
from models.qwen3_4b_base_cpt import Qwen3_4B_Base_CPT
from models.polybert import PolyBERTEncoder, POLYBERT_HIDDEN_DIM
from models.PerioGT import PerioGTEncoder, PERIOGT_HIDDEN_DIM

# ======================== Structure Encoder Registry ========================
STRUCT_ENCODER_REGISTRY = {
    "polybert": {"encoder_cls": PolyBERTEncoder, "hidden_dim": POLYBERT_HIDDEN_DIM},
    "periogt": {"encoder_cls": PerioGTEncoder, "hidden_dim": PERIOGT_HIDDEN_DIM},
}

def get_struct_hidden_dim(encoder_name: str, cfg: dict = None) -> int:
    if encoder_name == "llm":
        from transformers import AutoConfig
        model_type = cfg.get("MODEL_TYPE", "qwen3_4b_base")
        auto_cfg = AutoConfig.from_pretrained(
            _resolve_config_path(model_type), trust_remote_code=True
        )
        return auto_cfg.hidden_size
    base_dim = STRUCT_ENCODER_REGISTRY[encoder_name]["hidden_dim"]
    # PolyBERT etc. encoders concatenate ratio information vectors, so ratio_dim must be added
    if encoder_name == "polybert":
        import os, json, warnings
        ratio_path = cfg.get("_RATIO_ENCODING_PATH") if cfg else None
        if ratio_path and os.path.exists(ratio_path):
            with open(ratio_path, 'r') as f:
                ratio_encoding = json.load(f)
            base_dim += ratio_encoding["vector_dim"]
        else:
            warnings.warn(
                f"Ratio encoding file not found: {ratio_path}. "
                f"Returning base dim={base_dim} without ratio info. "
                f"Ensure scan_and_save_ratio_encoding() is called before model initialization."
            )
    return base_dim

# ======================== Model Registry ========================
LLM_REGISTRY = {
    "qwen3_4b_instruct_2507": Qwen3_4B_Instruct_2507,
    "qwen3_4b_thinking_2507": Qwen3_4B_Thinking_2507,
    "qwen3_4b_base":          Qwen3_4B_Base,
    "qwen3_8b_base":          Qwen3_8B_Base,
    "qwen3_0_6b_base":        Qwen3_0_6B_Base,
    "chemdfm_v1_5_8b":        ChemDFM_v1_5_8B,
    "qwen3_4b_base_cpt_1":    Qwen3_4B_Base_CPT,
    "qwen3_4b_base_cpt_2":    Qwen3_4B_Base_CPT,
    "qwen3_4b_base_cpt_3":    Qwen3_4B_Base_CPT,
}

# CPT model -> base model mapping (CPT is a LoRA adapter directory, no standalone config.json)
_CPT_BASE_MAP = {
    "qwen3_4b_base_cpt_1": "qwen3_4b_base",
    "qwen3_4b_base_cpt_2": "qwen3_4b_base",
    "qwen3_4b_base_cpt_3": "qwen3_4b_base",
}

def _resolve_config_path(model_type: str) -> str:
    """CPT model is a LoRA adapter directory; fall back to the base model path to get config.json"""
    from config.path import MODEL_PATHS
    base_type = _CPT_BASE_MAP.get(model_type, model_type)
    return MODEL_PATHS[base_type]


def get_activation(name: str) -> nn.Module:
    return {"SiLU": nn.SiLU(), "ReLU": nn.ReLU(), "GELU": nn.GELU()}[name]


# ======================== Article-Aware Loss Functions ========================

def article_consistency_loss(preds: torch.Tensor, targets: torch.Tensor,
                              article_ids: torch.Tensor) -> torch.Tensor:
    """
    Article consistency loss: penalizes high variance in prediction errors within the same article.
    For each article with sample count > 1, compute the variance of prediction errors (pred - target),
    then return the mean variance across all articles. Returns 0 if no article has multiple samples.
    """
    errors = preds - targets
    unique_articles = article_ids.unique()
    variances = []
    for aid in unique_articles:
        mask = (article_ids == aid)
        count = mask.sum()
        if count > 1:
            article_errors = errors[mask]
            variances.append(article_errors.var(unbiased=False))
    if not variances:
        return torch.tensor(0.0, device=preds.device, requires_grad=True)
    return torch.stack(variances).mean()


def article_bias_loss(preds: torch.Tensor, targets: torch.Tensor,
                      article_ids: torch.Tensor) -> torch.Tensor:
    """
    Article bias loss: penalizes article-level systematic bias.
    Computes the mean prediction error for each article, then takes the mean of squared article means.
    If the model has no article-level bias, each article's mean error should be near zero.
    L = (1/|A|) * sum_a ( mean_{i in a}(pred_i - target_i) )^2
    """
    errors = preds - targets
    unique_articles = article_ids.unique()
    if unique_articles.numel() < 1:
        return torch.tensor(0.0, device=preds.device, requires_grad=True)
    squared_means = []
    for aid in unique_articles:
        mask = (article_ids == aid)
        mean_error = errors[mask].mean()
        squared_means.append(mean_error ** 2)
    return torch.stack(squared_means).mean()


def article_ranking_loss(preds: torch.Tensor, targets: torch.Tensor,
                         article_ids: torch.Tensor) -> torch.Tensor:
    """
    Article pairwise ranking loss: encourages the model to maintain correct relative ordering within the same article.
    For each article with sample count > 1, enumerates all ordered pairs (i, j) where y_i > y_j,
    computing L = (1/|P|) * Σ log(1 + exp(-(ŷ_i - ŷ_j)))
    where P is the set of valid pairs. Returns 0 if no valid pairs in batch.
    """
    unique_articles = article_ids.unique()
    losses = []
    for aid in unique_articles:
        mask = (article_ids == aid)
        if mask.sum() < 2:
            continue
        p = preds[mask]
        t = targets[mask]
        # Build difference matrices for all (i, j) pairs
        t_diff = t.unsqueeze(0) - t.unsqueeze(1)  # t_diff[i,j] = t_j - t_i
        p_diff = p.unsqueeze(0) - p.unsqueeze(1)  # p_diff[i,j] = p_j - p_i
        # Select pairs where t_i > t_j (upper-triangular positions where t_diff < 0 are equivalent to t_i > t_j)
        valid = (t_diff < 0)  # valid[i,j] means t_j < t_i, want p_i > p_j
        if valid.sum() == 0:
            continue
        # For valid pairs: loss = log(1 + exp(-(ŷ_i - ŷ_j))) = log(1 + exp(p_diff[i,j]))
        # Since p_diff[i,j] = p_j - p_i, we have -(p_i - p_j) = p_diff[i,j]
        # Use F.softplus instead of manual log1p(exp(...)) to avoid exp overflow -> NaN
        # clamp prevents extreme prediction differences from causing numerical issues during backprop
        pair_losses = torch.nn.functional.softplus(p_diff[valid].clamp(-50.0, 50.0))
        losses.append(pair_losses.mean())
    if not losses:
        return torch.tensor(0.0, device=preds.device, requires_grad=True)
    return torch.stack(losses).mean()


def _compute_article_range_hit_rate(preds_orig: np.ndarray,
                                    article_ids: np.ndarray,
                                    article_ranges: Dict[int, tuple]) -> float:
    """
    Article range hit rate (leak-split evaluation metric): for each val/test sample, check whether the
    prediction (original scale) falls within the [min, max] range of training targets for the same article.
    Only counts samples whose article_id is in article_ranges and whose training range is non-degenerate (min < max).
    """
    hits = 0
    total = 0
    for i, aid in enumerate(article_ids):
        aid_int = int(aid)
        if aid_int not in article_ranges:
            continue
        lo, hi = article_ranges[aid_int]
        if lo >= hi:
            continue  # Single-sample article has no range, skip
        total += 1
        if lo <= preds_orig[i] <= hi:
            hits += 1
    if total == 0:
        return float('nan')
    return hits / total


def _compute_article_nars(preds_orig: np.ndarray,
                          article_ids: np.ndarray,
                          article_ranges: Dict[int, tuple]) -> float:
    """
    Normalized Article Range Score (NARS): Gaussian kernel score centered on the article's training target range.
    For each valid sample, compute s = exp( -(pred - center)^2 / (2 * half_width^2) ),
    return the mean score across all valid samples.
    - Prediction exactly at range center → s = 1
    - Prediction at range edge → s ≈ 0.607
    - Prediction farther from range → s decays smoothly to 0
    - Automatically adapts to range width: narrow ranges penalize more sensitively, wide ranges are more tolerant
    Only counts samples whose article_id is in article_ranges and whose training range is non-degenerate (min < max).
    """
    scores = []
    for i, aid in enumerate(article_ids):
        aid_int = int(aid)
        if aid_int not in article_ranges:
            continue
        lo, hi = article_ranges[aid_int]
        if lo >= hi:
            continue
        center = (lo + hi) / 2.0
        half_width = (hi - lo) / 2.0
        s = np.exp(-((preds_orig[i] - center) ** 2) / (2.0 * half_width ** 2))
        scores.append(s)
    if not scores:
        return float('nan')
    return float(np.mean(scores))


# ======================== Pooling Module ========================

class LastTokenPooling(nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if attention_mask is not None:
            seq_lengths = attention_mask.sum(dim=1).long() - 1
            return hidden_states[torch.arange(hidden_states.size(0),
                                              device=hidden_states.device), seq_lengths]
        return hidden_states[:, -1, :]


class MeanPooling(nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return hidden_states.mean(dim=1)


class SumPooling(nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            return (hidden_states * mask).sum(dim=1)
        return hidden_states.sum(dim=1)


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int, proj_dim: int):
        super().__init__()
        self.w_proj = nn.Linear(hidden_dim, proj_dim)
        self.w_score = nn.Linear(proj_dim, 1)

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                return_weights: bool = False):
        e = self.w_score(torch.tanh(self.w_proj(hidden_states)))  # (B, n, 1)
        if attention_mask is not None:
            e = e.masked_fill(attention_mask.unsqueeze(-1) == 0, float("-inf"))
        alpha = torch.softmax(e, dim=1)  # (B, n, 1)
        pooled = (alpha * hidden_states).sum(dim=1)
        if return_weights:
            return pooled, alpha.squeeze(-1)  # (B, d), (B, n)
        return pooled


class SigmoidGatingPooling(nn.Module):
    def __init__(self, hidden_dim: int, proj_dim: int):
        super().__init__()
        self.w_proj = nn.Linear(hidden_dim, proj_dim)
        self.w_score = nn.Linear(proj_dim, 1)

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                return_weights: bool = False):
        e = self.w_score(torch.tanh(self.w_proj(hidden_states)))  # (B, n, 1)
        alpha = torch.sigmoid(e)  # (B, n, 1)
        if attention_mask is not None:
            alpha = alpha * attention_mask.unsqueeze(-1).float()
        pooled = (alpha * hidden_states).sum(dim=1)
        if return_weights:
            return pooled, alpha.squeeze(-1)  # (B, d), (B, n)
        return pooled


def build_pooling(pooling_type: str, hidden_dim: int, proj_dim: int) -> nn.Module:
    if pooling_type == "last_token":
        return LastTokenPooling()
    elif pooling_type == "mean_pooling":
        return MeanPooling()
    elif pooling_type == "sum_pooling":
        return SumPooling()
    elif pooling_type == "attention_pooling":
        return AttentionPooling(hidden_dim, proj_dim)
    elif pooling_type == "sigmoid_pooling":
        return SigmoidGatingPooling(hidden_dim, proj_dim)
    else:
        raise ValueError(f"Unknown pooling type: {pooling_type}")


# ======================== MLP Projection Head ========================

class RegressionHead(nn.Module):
    """
    MLP projection head: d0 -> d1 -> d2 -> 1
    """
    def __init__(self, input_dim: int, hidden_dims: List[int],
                 activation: str = "SiLU", dropout: float = 0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(get_activation(activation))
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).squeeze(-1)


# ======================== Hybrid-Stream Gating ========================

class HybridGate(nn.Module):
    """
    Dynamic gated fusion: beta * v'_text + (1-beta) * v'_struct
    β = σ(W2 · SiLU(W1 · [v_text ∥ v_struct] + b1) + b2)
    """
    def __init__(self, text_dim: int, struct_dim: int,
                 proj_dim: int, gate_hidden_dim: int):
        super().__init__()
        # Alignment projections
        self.text_proj = nn.Linear(text_dim, proj_dim)
        self.struct_proj = nn.Linear(struct_dim, proj_dim)
        self.text_ln = nn.LayerNorm(proj_dim)
        self.struct_ln = nn.LayerNorm(proj_dim)

        # Gate MLP: concat(v_text, v_struct) -> gate_hidden_dim (SiLU) -> 1 (Sigmoid)
        self.gate_mlp = nn.Sequential(
            nn.Linear(text_dim + struct_dim, gate_hidden_dim),
            nn.SiLU(),
            nn.Linear(gate_hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, v_text: torch.Tensor,
                v_struct: torch.Tensor,
                return_gate: bool = False):
        v_text_aligned = self.text_ln(self.text_proj(v_text))
        v_struct_aligned = self.struct_ln(self.struct_proj(v_struct))

        beta = self.gate_mlp(torch.cat([v_text, v_struct], dim=-1))  # (B, 1)
        v_final = beta * v_text_aligned + (1 - beta) * v_struct_aligned
        if return_gate:
            return v_final, beta.squeeze(-1)  # (B, d4), (B,)
        return v_final


# ======================== Struct Stream Pooling ========================

class StructPooling(nn.Module):
    """Struct stream pooling (mean/attention pooling over multiple SMILES embeddings)"""
    def __init__(self, pooling_type: str, hidden_dim: int, proj_dim: int):
        super().__init__()
        if pooling_type == "attention_pooling":
            self.pool = AttentionPooling(hidden_dim, proj_dim)
        else:
            self.pool = MeanPooling()

    def forward(self, embeddings: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        embeddings: (B, max_smiles, hidden_dim)
        mask: (B, max_smiles) — 1 for valid, 0 for pad
        """
        return self.pool(embeddings, mask)


# ======================== Lightning Module ========================

class PropertyPredictionModel(pl.LightningModule):
    """
    Polymer property prediction model (supports non-hybrid / hybrid modes)
    """
    strict_loading = False  # Allow partial loading (trainable params only)

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        frozen = cfg["FROZEN_BACKBONE"]
        model_type = cfg["MODEL_TYPE"]
        pooling_type = cfg["POOLING_TYPE"]
        proj_dim = cfg["PROJ_DIM"]
        mlp_dims = cfg["MLP_HIDDEN_DIM"]
        mlp_act = cfg["MLP_ACTIVATION"]
        mlp_drop = cfg["MLP_DROPOUT"]
        enable_hybrid = cfg["ENABLE_HYBRID_ENCODING"]

        # ---------- Determine LLM hidden_dim ----------
        from config.path import MODEL_PATHS
        llm_cls = LLM_REGISTRY[model_type]
        if not frozen:
            # Need to load model to get hidden_dim and apply LoRA
            self.llm = llm_cls(MODEL_PATHS[model_type], device="cpu")
            llm_hidden_dim = self.llm.hidden_dim
            # Apply LoRA
            lora_config = LoraConfig(
                r=cfg["LORA_R"],
                lora_alpha=cfg["LORA_ALPHA"],
                lora_dropout=cfg["LORA_DROPOUT"],
                target_modules=cfg["LORA_TARGET_MODULES_LLM"],
                task_type=TaskType.CAUSAL_LM,
            )
            self.llm.model = get_peft_model(self.llm.model, lora_config)
            self.llm.model.print_trainable_parameters()
            # Register as nn.Module submodule so Lightning can track parameters and move to GPU
            self._llm_backbone = self.llm.model
        else:
            # Frozen mode: get hidden_dim from config, do not load model
            # Use model config file to retrieve hidden_dim
            from transformers import AutoConfig
            auto_cfg = AutoConfig.from_pretrained(
                _resolve_config_path(model_type), trust_remote_code=True
            )
            llm_hidden_dim = auto_cfg.hidden_size
            self.llm = None

        self.llm_hidden_dim = llm_hidden_dim
        self.frozen = frozen

        # ---------- LayerNorm + Pooling ----------
        self.layer_norm = nn.LayerNorm(llm_hidden_dim)
        self.pooling = build_pooling(pooling_type, llm_hidden_dim, proj_dim)

        # ---------- Hybrid Stream ----------
        self.struct_layer_norm = None
        self.periogt = None
        if enable_hybrid:
            encoder_name = cfg.get("STRUCT_ENCODER", "polybert")
            struct_hidden = get_struct_hidden_dim(encoder_name, cfg)
            struct_pool_type = cfg["STRUCT_POOLING_TYPE"]
            struct_proj_dim = cfg["STRUCT_PROJ_DIM"]
            hybrid_proj_dim = cfg["HYBRID_PROJ_DIM"]
            gate_dims = cfg["HYBRID_GATE_DIM"]

            if encoder_name == "periogt":
                # PerioGT always trains full parameters (no frozen/LoRA support)
                from config.path import PERIOGT_CONFIG_PATH
                import json
                _ratio_path = cfg["_RATIO_ENCODING_PATH"]
                with open(_ratio_path, 'r') as f:
                    ratio_encoding = json.load(f)
                d_ratio_feats = ratio_encoding["vector_dim"]
                self.periogt = PerioGTEncoder(
                    MODEL_PATHS["periogt"], d_ratio_feats=d_ratio_feats,
                    config_path=PERIOGT_CONFIG_PATH, device="cpu",
                    use_prompt=cfg.get("USE_PERIOGT_PROMPT", True)
                )
                self.struct_encoder = None
                # PerioGT has built-in graph-level readout, no StructPooling needed
                self.struct_pooling = None
            elif encoder_name == "llm":
                # LLM struct stream: use the same LLM to process type 9 inputs, no separate encoder needed
                self.struct_encoder = None
                self.struct_layer_norm = nn.LayerNorm(struct_hidden)
                self.struct_pooling = StructPooling(
                    struct_pool_type, struct_hidden, struct_proj_dim
                )
            elif not frozen:
                # polybert non-frozen mode: load struct encoder and apply LoRA
                encoder_cls = STRUCT_ENCODER_REGISTRY[encoder_name]["encoder_cls"]
                self.struct_encoder = encoder_cls(MODEL_PATHS[encoder_name], device="cpu")
                lora_config_struct = LoraConfig(
                    r=cfg["LORA_R"],
                    lora_alpha=cfg["LORA_ALPHA"],
                    lora_dropout=cfg["LORA_DROPOUT"],
                    target_modules=cfg.get("LORA_TARGET_MODULES_STRUCT",
                                           ["query_proj", "key_proj", "value_proj"]),
                    task_type=TaskType.FEATURE_EXTRACTION,
                )
                self.struct_encoder.model = get_peft_model(
                    self.struct_encoder.model, lora_config_struct
                )
                self.struct_encoder.model.print_trainable_parameters()
                self._struct_backbone = self.struct_encoder.model
                self.struct_pooling = StructPooling(
                    struct_pool_type, struct_hidden, struct_proj_dim
                )
            else:
                self.struct_encoder = None
                self.struct_pooling = StructPooling(
                    struct_pool_type, struct_hidden, struct_proj_dim
                )

            self.hybrid_gate = HybridGate(
                text_dim=llm_hidden_dim,
                struct_dim=struct_hidden,
                proj_dim=hybrid_proj_dim,
                gate_hidden_dim=gate_dims,
            )
            regression_input_dim = hybrid_proj_dim
        else:
            self.struct_pooling = None
            self.hybrid_gate = None
            regression_input_dim = llm_hidden_dim

        # Cache struct stream vector dimension to avoid re-reading JSON in _encode_smiles_batch
        self._struct_hidden_dim = get_struct_hidden_dim(
            cfg.get("STRUCT_ENCODER", "polybert"), cfg
        ) if enable_hybrid else 0

        # ---------- Regression Head ----------
        self.regression_head = RegressionHead(
            regression_input_dim, mlp_dims, mlp_act, mlp_drop
        )

        self.loss_fn = nn.MSELoss()

        # scaler params (used for logging original-scale RMSE/MAE in TensorBoard)
        self._scaler_mean = cfg.get("_SCALER_MEAN", 0.0)
        self._scaler_std = cfg.get("_SCALER_STD", 1.0)

        # Accumulate predictions and targets each epoch for computing R2/RMSE
        self._train_preds = []
        self._train_targets = []
        self._train_article_ids = []
        self._val_preds = []
        self._val_targets = []
        self._val_article_ids = []
        self._test_preds = []
        self._test_targets = []
        self._train_article_ranges = {}  # Training target value range per article (for leak-split evaluation)

    def _encode_smiles_batch(self, smiles_lists: List[List[str]],
                             ratio_vectors_lists: Optional[List[List[torch.Tensor]]] = None) -> tuple:
        """
        Non-frozen hybrid mode: encode a batch of SMILES lists in real-time (with gradients).
        If ratio_vectors_lists is provided, concatenate the corresponding ratio info vectors after encoding each SMILES.
        """
        struct_hidden = self._struct_hidden_dim
        all_embs = []
        for b_idx, smiles_list in enumerate(smiles_lists):
            if not smiles_list:
                emb = torch.zeros(1, struct_hidden, device=self.device)
            else:
                inputs = self.struct_encoder.tokenizer(
                    smiles_list, padding=True, truncation=True,
                    max_length=512, return_tensors="pt"
                ).to(self.device)
                outputs = self.struct_encoder.model(**inputs)
                hidden_states = outputs.last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                emb = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                # Concatenate ratio information vectors; fill with zeros if missing to maintain dimension consistency
                ratio_dim = struct_hidden - emb.size(-1)
                if ratio_dim > 0:
                    if ratio_vectors_lists is not None and b_idx < len(ratio_vectors_lists):
                        rvecs = ratio_vectors_lists[b_idx]
                        if rvecs:
                            ratio_t = torch.stack(rvecs[:emb.size(0)]).to(self.device)
                            emb = torch.cat([emb, ratio_t], dim=-1)
                        else:
                            emb = torch.cat([emb, torch.zeros(emb.size(0), ratio_dim, device=self.device)], dim=-1)
                    else:
                        emb = torch.cat([emb, torch.zeros(emb.size(0), ratio_dim, device=self.device)], dim=-1)
            all_embs.append(emb)

        max_smiles = max(e.size(0) for e in all_embs)
        bsz = len(smiles_lists)
        struct_emb = torch.zeros(bsz, max_smiles, struct_hidden, device=self.device)
        struct_mask = torch.zeros(bsz, max_smiles, dtype=torch.long, device=self.device)
        for i, emb in enumerate(all_embs):
            n = emb.size(0)
            struct_emb[i, :n] = emb
            struct_mask[i, :n] = 1
        return struct_emb, struct_mask

    def forward(self, text_hidden: torch.Tensor,
                text_mask: Optional[torch.Tensor] = None,
                struct_emb: Optional[torch.Tensor] = None,
                struct_mask: Optional[torch.Tensor] = None,
                graph_data: Optional[Dict] = None,
                return_aux: bool = False):
        """
        Args:
            text_hidden: (B, seq_len, d_llm) — last LLM layer hidden states
            text_mask: (B, seq_len)
            struct_emb: (B, N, d_struct) — structure embeddings (PolyBERT) or LLM type 9 hidden states
            struct_mask: (B, N)
            graph_data: PerioGT graph data dict (used with hybrid stream + PerioGT)
            return_aux: whether to return auxiliary info (pooling weights, gate weights)
        """
        normed = self.layer_norm(text_hidden)
        aux = {}

        # Pooling (optionally return weights)
        pool_type = self.cfg["POOLING_TYPE"]
        if return_aux and pool_type in ("attention_pooling", "sigmoid_pooling"):
            v_text, pooling_weights = self.pooling(normed, text_mask, return_weights=True)
            aux["pooling_weights"] = pooling_weights  # (B, seq_len)
        else:
            v_text = self.pooling(normed, text_mask)

        if self.hybrid_gate is not None:
            if self.periogt is not None and graph_data is not None:
                v_struct = self.periogt.get_embedding(**graph_data)
            elif struct_emb is not None:
                if self.struct_layer_norm is not None:
                    struct_emb = self.struct_layer_norm(struct_emb)
                v_struct = self.struct_pooling(struct_emb, struct_mask)
            else:
                v_struct = None

            if v_struct is not None:
                if return_aux:
                    v_final, gate_beta = self.hybrid_gate(v_text, v_struct, return_gate=True)
                    aux["gate_beta"] = gate_beta  # (B,)
                else:
                    v_final = self.hybrid_gate(v_text, v_struct)
            else:
                v_final = v_text
        else:
            v_final = v_text

        preds = self.regression_head(v_final)
        if return_aux:
            return preds, aux
        return preds

    def _compute_noise_std(self):
        """Compute noise standard deviation for current step (cosine annealing)"""
        base_std = self.cfg.get("NOISE_STD", 0.1)
        if not self.cfg.get("NOISE_ANNEAL", True):
            return base_std
        total_steps = self.trainer.estimated_stepping_batches if self.trainer else 1
        progress = min(self.global_step / max(total_steps, 1), 1.0)
        return base_std * (1 + np.cos(np.pi * progress)) / 2

    def _shared_step(self, batch, stage: str):
        if self.frozen:
            text_hidden = batch["text_hidden"]
            text_mask = batch.get("text_mask", None)
        else:
            inputs = self.llm.tokenize(batch["text"])
            device = self.device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}
            text_hidden = self.llm.get_hidden_states(inputs).float()
            text_mask = inputs.get("attention_mask", None)

        struct_emb = batch.get("struct_emb", None)
        struct_mask = batch.get("struct_mask", None)
        graph_data = None

        # Hybrid + PerioGT: always encode in real-time (PerioGT trains full params, no cache mode)
        if self.hybrid_gate is not None and self.periogt is not None:
            graph_data = {
                'graphs': batch['graphs'].to(self.device),
                'fp_1': batch['fp_1'].to(self.device),
                'md_1': batch['md_1'].to(self.device),
                'fp_2': batch['fp_2'].to(self.device),
                'md_2': batch['md_2'].to(self.device),
                'ratio_vec_1': batch['ratio_vec_1'].to(self.device),
                'ratio_vec_2': batch['ratio_vec_2'].to(self.device),
                'global_type': batch['global_type'].to(self.device),
            }
        # Hybrid + non-frozen mode: encode structure in real-time (polybert / llm)
        elif self.hybrid_gate is not None and struct_emb is None and not self.frozen:
            encoder_name = self.cfg.get("STRUCT_ENCODER", "polybert")
            if encoder_name == "llm":
                # LLM struct stream (non-frozen): use the same LLM for type 9 inputs
                type9_texts = batch.get("struct_texts", None)
                if type9_texts is not None:
                    type9_inputs = self.llm.tokenize(type9_texts)
                    type9_inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                                    for k, v in type9_inputs.items()}
                    struct_emb = self.llm.get_hidden_states(type9_inputs).float()
                    struct_mask = type9_inputs.get("attention_mask", None)
            else:
                smiles_lists = batch.get("smiles_lists", None)
                if smiles_lists is not None:
                    ratio_vectors_lists = batch.get("ratio_vectors_lists", None)
                    struct_emb, struct_mask = self._encode_smiles_batch(
                        smiles_lists, ratio_vectors_lists=ratio_vectors_lists)

        targets = batch["target"]

        preds = self(text_hidden, text_mask, struct_emb, struct_mask, graph_data=graph_data)

        # Gaussian noise (training only)
        if self.cfg.get("NOISE_ENABLED", False) and stage == "train":
            noise_std = self._compute_noise_std()
            noise_mean = self.cfg.get("NOISE_MEAN", 0.0)
            noise = torch.randn_like(preds) * noise_std + noise_mean
            preds_for_loss = preds + noise
        else:
            preds_for_loss = preds

        mse_loss = self.loss_fn(preds_for_loss, targets)
        w_mse = self.cfg.get("MSE_WEIGHT", 1.0)
        loss = w_mse * mse_loss

        # All losses written to TensorBoard only at epoch end (on_step=False),
        # epoch-level aggregated metrics computed in on_*_epoch_end, no step-level logging needed

        # Article-aware loss (computed when batch contains article_ids, applied during both training and validation)
        if "article_ids" in batch:
            article_ids = batch["article_ids"].to(self.device)
            w_consist = self.cfg.get("ARTICLE_CONSISTENCY_WEIGHT", 0)
            w_bias = self.cfg.get("ARTICLE_BIAS_WEIGHT", 0)
            w_rank = self.cfg.get("ARTICLE_RANKING_WEIGHT", 0)
            if w_consist > 0:
                l_consist = article_consistency_loss(preds_for_loss, targets, article_ids)
                loss = loss + w_consist * l_consist
                self.log(f"{stage}/article_consist_loss", l_consist, prog_bar=False, sync_dist=True,
                         on_step=False, on_epoch=True)
            if w_bias > 0:
                l_bias = article_bias_loss(preds_for_loss, targets, article_ids)
                loss = loss + w_bias * l_bias
                self.log(f"{stage}/article_bias_loss", l_bias, prog_bar=False, sync_dist=True,
                         on_step=False, on_epoch=True)
            if w_rank > 0:
                l_rank = article_ranking_loss(preds_for_loss, targets, article_ids)
                loss = loss + w_rank * l_rank
                self.log(f"{stage}/article_ranking_loss", l_rank, prog_bar=False, sync_dist=True,
                         on_step=False, on_epoch=True)
            self.log(f"{stage}/mse_loss", mse_loss, prog_bar=False, sync_dist=True,
                     on_step=False, on_epoch=True)

        self.log(f"{stage}/loss", loss, prog_bar=True, sync_dist=True,
                 on_step=False, on_epoch=True)
        result = {"loss": loss, "preds": preds.detach(), "targets": targets.detach()}
        if "article_ids" in batch:
            result["article_ids"] = batch["article_ids"].detach()
        return result

    def training_step(self, batch, batch_idx):
        result = self._shared_step(batch, "train")
        self._train_preds.append(result["preds"])
        self._train_targets.append(result["targets"])
        if "article_ids" in result:
            self._train_article_ids.append(result["article_ids"])
        return result

    def validation_step(self, batch, batch_idx):
        result = self._shared_step(batch, "val")
        self._val_preds.append(result["preds"])
        self._val_targets.append(result["targets"])
        if "article_ids" in result:
            self._val_article_ids.append(result["article_ids"])
        return result

    def test_step(self, batch, batch_idx):
        result = self._shared_step(batch, "test")
        self._test_preds.append(result["preds"])
        self._test_targets.append(result["targets"])
        return result

    def _inverse(self, arr: np.ndarray) -> np.ndarray:
        """Inverse-transform standardized values back to original scale"""
        return arr * self._scaler_std + self._scaler_mean

    def on_train_epoch_end(self):
        if self._train_preds:
            preds = torch.cat(self._train_preds).cpu().numpy()
            targets = torch.cat(self._train_targets).cpu().numpy()
            r2 = float(r2_score(targets, preds)) if len(targets) > 1 else 0.0
            preds_orig = self._inverse(preds)
            targets_orig = self._inverse(targets)
            rmse = float(np.sqrt(np.mean((preds_orig - targets_orig) ** 2)))
            mae = float(np.mean(np.abs(preds_orig - targets_orig)))
            self.log("train/rmse", rmse, prog_bar=False)
            self.log("train/r2", r2, prog_bar=False)
            self.log("train/mae", mae, prog_bar=False)
            # Store target value range per article in training set (for leak-split evaluation metrics)
            if self._train_article_ids:
                aids = torch.cat(self._train_article_ids).cpu().numpy()
                self._train_article_ranges = {}
                for aid in np.unique(aids):
                    mask = (aids == aid)
                    vals = targets_orig[mask]
                    self._train_article_ranges[int(aid)] = (float(vals.min()), float(vals.max()))
        self._train_preds.clear()
        self._train_targets.clear()
        self._train_article_ids.clear()

    def on_validation_epoch_end(self):
        if self._val_preds:
            preds = torch.cat(self._val_preds).cpu().numpy()
            targets = torch.cat(self._val_targets).cpu().numpy()
            r2 = float(r2_score(targets, preds)) if len(targets) > 1 else 0.0
            preds_orig = self._inverse(preds)
            targets_orig = self._inverse(targets)
            rmse = float(np.sqrt(np.mean((preds_orig - targets_orig) ** 2)))
            mae = float(np.mean(np.abs(preds_orig - targets_orig)))
            self.log("val/rmse", rmse, prog_bar=True)
            self.log("val/r2", r2, prog_bar=True)
            self.log("val/mae", mae, prog_bar=False)
            # Article-aware metrics (epoch-level only)
            if self._val_article_ids:
                aids = torch.cat(self._val_article_ids).cpu().numpy()
                hr = float('nan')
                # Article range hit rate (valid for leak-split: check if prediction falls within article training range)
                if self._train_article_ranges:
                    hr = _compute_article_range_hit_rate(preds_orig, aids, self._train_article_ranges)
                    if not np.isnan(hr):
                        self.log("val/article_range_hit_rate", hr, prog_bar=False)
                    # NARS (Normalized Article Range Score)
                    nars = _compute_article_nars(preds_orig, aids, self._train_article_ranges)
                    if not np.isnan(nars):
                        self.log("val/article_nars", nars, prog_bar=False)
                # Print article-aware metrics to console
                parts = []
                if not np.isnan(hr):
                    parts.append(f"RangeHR={hr:.4f}")
                if 'nars' in locals() and not np.isnan(nars):
                    parts.append(f"NARS={nars:.4f}")
                if parts:
                    print(f"  [Val] Article: {', '.join(parts)}")
        self._val_preds.clear()
        self._val_targets.clear()
        self._val_article_ids.clear()

    def on_test_epoch_end(self):
        if self._test_preds:
            preds = torch.cat(self._test_preds).cpu().numpy()
            targets = torch.cat(self._test_targets).cpu().numpy()
            r2 = float(r2_score(targets, preds)) if len(targets) > 1 else 0.0
            preds_orig = self._inverse(preds)
            targets_orig = self._inverse(targets)
            rmse = float(np.sqrt(np.mean((preds_orig - targets_orig) ** 2)))
            mae = float(np.mean(np.abs(preds_orig - targets_orig)))
            self.log("test/rmse", rmse)
            self.log("test/r2", r2)
            self.log("test/mae", mae)
        self._test_preds.clear()
        self._test_targets.clear()

    def on_save_checkpoint(self, checkpoint):
        """Save only trainable parameters to avoid saving full backbone"""
        trainable_keys = {n for n, p in self.named_parameters() if p.requires_grad}
        checkpoint["state_dict"] = {
            k: v for k, v in checkpoint["state_dict"].items()
            if k in trainable_keys
        }

    def configure_optimizers(self):
        lr = self.cfg["LR"]
        backbone_lr = self.cfg["BACKBONE_LR"]
        wd = self.cfg["WEIGHT_DECAY"]
        warmup = self.cfg["WARMUP_STEPS"]
        epochs = self.cfg["EPOCHS"]

        if not self.frozen and self.llm is not None:
            # LoRA / PerioGT backbone params use backbone_lr, rest use lr
            backbone_params = []
            head_params = []
            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                if "llm" in name or "_struct_backbone" in name or "periogt" in name:
                    backbone_params.append(param)
                else:
                    head_params.append(param)
            optimizer = torch.optim.AdamW([
                {"params": backbone_params, "lr": backbone_lr},
                {"params": head_params, "lr": lr},
            ], weight_decay=wd)
        elif self.periogt is not None:
            # Frozen LLM + PerioGT hybrid: PerioGT pretrained backbone uses backbone_lr, regression head etc. use lr
            backbone_params = []
            head_params = []
            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                if "periogt" in name:
                    backbone_params.append(param)
                else:
                    head_params.append(param)
            optimizer = torch.optim.AdamW([
                {"params": backbone_params, "lr": backbone_lr},
                {"params": head_params, "lr": lr},
            ], weight_decay=wd)
        else:
            trainable = [p for p in self.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=wd)

        # Compute total steps for cosine annealing
        trainer = self.trainer
        if trainer and trainer.estimated_stepping_batches:
            total_steps = trainer.estimated_stepping_batches
        else:
            total_steps = epochs * 100  # fallback

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps - warmup), eta_min=1e-7
        )

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup
        )

        combined = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, scheduler], milestones=[warmup]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": combined,
                "interval": "step",
            },
        }


# ======================== Structure-Only Model (Control Group) ========================

class StructureOnlyModel(pl.LightningModule):
    """
    Control model using only the structure encoder.
    Supports polybert / periogt / llm encoders:
    - polybert: frozen mode uses precomputed embeddings; finetune mode uses LoRA.
    - periogt: always trains full parameters (no frozen/LoRA).
    - llm: frozen mode uses precomputed type 9 embeddings; finetune mode uses LoRA.
    """
    strict_loading = False

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        encoder_name = cfg.get("STRUCT_ENCODER", "polybert")
        self.encoder_name = encoder_name
        mlp_dims = cfg["MLP_HIDDEN_DIM"]
        mlp_act = cfg["MLP_ACTIVATION"]
        mlp_drop = cfg["MLP_DROPOUT"]

        if encoder_name == "periogt":
            # PerioGT always trains full parameters, no frozen/LoRA support
            self.frozen = False
            from config.path import MODEL_PATHS, PERIOGT_CONFIG_PATH
            import json
            _ratio_path = cfg["_RATIO_ENCODING_PATH"]
            with open(_ratio_path, 'r') as f:
                ratio_encoding = json.load(f)
            d_ratio_feats = ratio_encoding["vector_dim"]
            self.periogt = PerioGTEncoder(
                MODEL_PATHS["periogt"], d_ratio_feats=d_ratio_feats,
                config_path=PERIOGT_CONFIG_PATH, device="cpu",
                use_prompt=cfg.get("USE_PERIOGT_PROMPT", True)
            )
            struct_hidden = PERIOGT_HIDDEN_DIM
            self.struct_pooling = None
            self.struct_encoder = None
            self.llm = None

        elif encoder_name == "llm":
            # LLM struct stream: use the LLM to process type 9 inputs
            self.frozen = cfg.get("FROZEN_BACKBONE", True)
            model_type = cfg["MODEL_TYPE"]
            from config.path import MODEL_PATHS
            if not self.frozen:
                llm_cls = LLM_REGISTRY[model_type]
                self.llm = llm_cls(MODEL_PATHS[model_type], device="cpu")
                struct_hidden = self.llm.hidden_dim
                lora_config = LoraConfig(
                    r=cfg["LORA_R"],
                    lora_alpha=cfg["LORA_ALPHA"],
                    lora_dropout=cfg["LORA_DROPOUT"],
                    target_modules=cfg.get("LORA_TARGET_MODULES_LLM",
                                           ["q_proj", "k_proj", "v_proj", "o_proj",
                                            "gate_proj", "up_proj", "down_proj"]),
                    task_type=TaskType.CAUSAL_LM,
                )
                self.llm.model = get_peft_model(self.llm.model, lora_config)
                self.llm.model.print_trainable_parameters()
                self._llm_backbone = self.llm.model
            else:
                from transformers import AutoConfig
                auto_cfg = AutoConfig.from_pretrained(
                    _resolve_config_path(model_type), trust_remote_code=True
                )
                struct_hidden = auto_cfg.hidden_size
                self.llm = None
            self.llm_layer_norm = nn.LayerNorm(struct_hidden)
            struct_pool_type = cfg["STRUCT_POOLING_TYPE"]
            struct_proj_dim = cfg["STRUCT_PROJ_DIM"]
            self.llm_pooling = build_pooling(struct_pool_type, struct_hidden, struct_proj_dim)
            self.struct_pooling = None
            self.struct_encoder = None
            self.periogt = None

        else:
            # polybert (default)
            self.frozen = cfg.get("FROZEN_BACKBONE", True)
            struct_hidden = get_struct_hidden_dim(encoder_name, cfg)
            struct_pool_type = cfg["STRUCT_POOLING_TYPE"]
            struct_proj_dim = cfg["STRUCT_PROJ_DIM"]

            if not self.frozen:
                from config.path import MODEL_PATHS
                encoder_cls = STRUCT_ENCODER_REGISTRY[encoder_name]["encoder_cls"]
                self.struct_encoder = encoder_cls(MODEL_PATHS[encoder_name], device="cpu")
                lora_config = LoraConfig(
                    r=cfg["LORA_R"],
                    lora_alpha=cfg["LORA_ALPHA"],
                    lora_dropout=cfg["LORA_DROPOUT"],
                    target_modules=cfg["LORA_TARGET_MODULES_STRUCT"],
                    task_type=TaskType.FEATURE_EXTRACTION,
                )
                self.struct_encoder.model = get_peft_model(self.struct_encoder.model, lora_config)
                self.struct_encoder.model.print_trainable_parameters()
                self._struct_backbone = self.struct_encoder.model
            else:
                self.struct_encoder = None

            self.struct_pooling = StructPooling(
                struct_pool_type, struct_hidden, struct_proj_dim
            )
            self.llm = None
            self.periogt = None

        self.struct_hidden = struct_hidden
        self.regression_head = RegressionHead(
            struct_hidden, mlp_dims, mlp_act, mlp_drop
        )
        self.loss_fn = nn.MSELoss()

        # scaler params (used for logging original-scale RMSE/MAE in TensorBoard)
        self._scaler_mean = cfg.get("_SCALER_MEAN", 0.0)
        self._scaler_std = cfg.get("_SCALER_STD", 1.0)

        self._train_preds = []
        self._train_targets = []
        self._train_article_ids = []
        self._val_preds = []
        self._val_targets = []
        self._val_article_ids = []
        self._test_preds = []
        self._test_targets = []
        self._train_article_ranges = {}  # Training target value range per article (for leak-split evaluation)

    def _encode_smiles_batch(self, smiles_lists: List[List[str]],
                             ratio_vectors_lists: Optional[List[List[torch.Tensor]]] = None) -> tuple:
        """
        Non-frozen mode (polybert): encode a batch of SMILES lists in real-time (with gradients).
        If ratio_vectors_lists is provided, concatenate the corresponding ratio info vectors after encoding each SMILES.
        """
        all_embs = []
        for b_idx, smiles_list in enumerate(smiles_lists):
            if not smiles_list:
                emb = torch.zeros(1, self.struct_hidden, device=self.device)
            else:
                inputs = self.struct_encoder.tokenizer(
                    smiles_list, padding=True, truncation=True,
                    max_length=512, return_tensors="pt"
                ).to(self.device)
                outputs = self.struct_encoder.model(**inputs)
                hidden_states = outputs.last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                emb = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                # Concatenate ratio information vectors; fill with zeros if missing to maintain dimension consistency
                ratio_dim = self.struct_hidden - emb.size(-1)
                if ratio_dim > 0:
                    if ratio_vectors_lists is not None and b_idx < len(ratio_vectors_lists):
                        rvecs = ratio_vectors_lists[b_idx]
                        if rvecs:
                            ratio_t = torch.stack(rvecs[:emb.size(0)]).to(self.device)
                            emb = torch.cat([emb, ratio_t], dim=-1)
                        else:
                            emb = torch.cat([emb, torch.zeros(emb.size(0), ratio_dim, device=self.device)], dim=-1)
                    else:
                        emb = torch.cat([emb, torch.zeros(emb.size(0), ratio_dim, device=self.device)], dim=-1)
            all_embs.append(emb)

        max_smiles = max(e.size(0) for e in all_embs)
        bsz = len(smiles_lists)
        struct_emb = torch.zeros(bsz, max_smiles, self.struct_hidden, device=self.device)
        struct_mask = torch.zeros(bsz, max_smiles, dtype=torch.long, device=self.device)
        for i, emb in enumerate(all_embs):
            n = emb.size(0)
            struct_emb[i, :n] = emb
            struct_mask[i, :n] = 1
        return struct_emb, struct_mask

    def forward(self, struct_emb: Optional[torch.Tensor] = None,
                struct_mask: Optional[torch.Tensor] = None,
                graph_data: Optional[Dict] = None,
                text_hidden: Optional[torch.Tensor] = None,
                text_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.encoder_name == "periogt" and graph_data is not None:
            g_feats = self.periogt.get_embedding(**graph_data)
            return self.regression_head(g_feats)
        elif self.encoder_name == "llm" and text_hidden is not None:
            normed = self.llm_layer_norm(text_hidden)
            v = self.llm_pooling(normed, text_mask)
            return self.regression_head(v)
        else:
            v_struct = self.struct_pooling(struct_emb, struct_mask)
            return self.regression_head(v_struct)

    def _compute_noise_std(self):
        """Compute noise standard deviation for current step (cosine annealing)"""
        base_std = self.cfg.get("NOISE_STD", 0.1)
        if not self.cfg.get("NOISE_ANNEAL", True):
            return base_std
        total_steps = self.trainer.estimated_stepping_batches if self.trainer else 1
        progress = min(self.global_step / max(total_steps, 1), 1.0)
        return base_std * (1 + np.cos(np.pi * progress)) / 2

    def _shared_step(self, batch, stage: str):
        if self.encoder_name == "periogt":
            preds = self(graph_data={
                'graphs': batch['graphs'].to(self.device),
                'fp_1': batch['fp_1'].to(self.device),
                'md_1': batch['md_1'].to(self.device),
                'fp_2': batch['fp_2'].to(self.device),
                'md_2': batch['md_2'].to(self.device),
                'ratio_vec_1': batch['ratio_vec_1'].to(self.device),
                'ratio_vec_2': batch['ratio_vec_2'].to(self.device),
                'global_type': batch['global_type'].to(self.device),
            })
        elif self.encoder_name == "llm":
            if self.frozen:
                text_hidden = batch["text_hidden"]
                text_mask = batch.get("text_mask", None)
            else:
                inputs = self.llm.tokenize(batch["text"])
                device = self.device
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                          for k, v in inputs.items()}
                text_hidden = self.llm.get_hidden_states(inputs).float()
                text_mask = inputs.get("attention_mask", None)
            preds = self(text_hidden=text_hidden, text_mask=text_mask)
        else:
            if self.frozen:
                struct_emb = batch["struct_emb"]
                struct_mask = batch.get("struct_mask", None)
            else:
                struct_emb, struct_mask = self._encode_smiles_batch(
                    batch["smiles_lists"],
                    ratio_vectors_lists=batch.get("ratio_vectors_lists"),
                )
            preds = self(struct_emb=struct_emb, struct_mask=struct_mask)

        targets = batch["target"]

        # Gaussian noise (training only)
        if self.cfg.get("NOISE_ENABLED", False) and stage == "train":
            noise_std = self._compute_noise_std()
            noise_mean = self.cfg.get("NOISE_MEAN", 0.0)
            noise = torch.randn_like(preds) * noise_std + noise_mean
            preds_for_loss = preds + noise
        else:
            preds_for_loss = preds

        mse_loss = self.loss_fn(preds_for_loss, targets)
        w_mse = self.cfg.get("MSE_WEIGHT", 1.0)
        loss = w_mse * mse_loss

        # All losses written to TensorBoard only at epoch end (on_step=False),
        # epoch-level aggregated metrics computed in on_*_epoch_end, no step-level logging needed

        # Article-aware loss (computed when batch contains article_ids, applied during both training and validation)
        if "article_ids" in batch:
            article_ids = batch["article_ids"].to(self.device)
            w_consist = self.cfg.get("ARTICLE_CONSISTENCY_WEIGHT", 0)
            w_bias = self.cfg.get("ARTICLE_BIAS_WEIGHT", 0)
            w_rank = self.cfg.get("ARTICLE_RANKING_WEIGHT", 0)
            if w_consist > 0:
                l_consist = article_consistency_loss(preds_for_loss, targets, article_ids)
                loss = loss + w_consist * l_consist
                self.log(f"{stage}/article_consist_loss", l_consist, prog_bar=False, sync_dist=True,
                         on_step=False, on_epoch=True)
            if w_bias > 0:
                l_bias = article_bias_loss(preds_for_loss, targets, article_ids)
                loss = loss + w_bias * l_bias
                self.log(f"{stage}/article_bias_loss", l_bias, prog_bar=False, sync_dist=True,
                         on_step=False, on_epoch=True)
            if w_rank > 0:
                l_rank = article_ranking_loss(preds_for_loss, targets, article_ids)
                loss = loss + w_rank * l_rank
                self.log(f"{stage}/article_ranking_loss", l_rank, prog_bar=False, sync_dist=True,
                         on_step=False, on_epoch=True)
            self.log(f"{stage}/mse_loss", mse_loss, prog_bar=False, sync_dist=True,
                     on_step=False, on_epoch=True)

        self.log(f"{stage}/loss", loss, prog_bar=True, sync_dist=True,
                 on_step=False, on_epoch=True)
        result = {"loss": loss, "preds": preds.detach(), "targets": targets.detach()}
        if "article_ids" in batch:
            result["article_ids"] = batch["article_ids"].detach()
        return result

    def training_step(self, batch, batch_idx):
        result = self._shared_step(batch, "train")
        self._train_preds.append(result["preds"])
        self._train_targets.append(result["targets"])
        if "article_ids" in result:
            self._train_article_ids.append(result["article_ids"])
        return result

    def validation_step(self, batch, batch_idx):
        result = self._shared_step(batch, "val")
        self._val_preds.append(result["preds"])
        self._val_targets.append(result["targets"])
        if "article_ids" in result:
            self._val_article_ids.append(result["article_ids"])
        return result

    def test_step(self, batch, batch_idx):
        result = self._shared_step(batch, "test")
        self._test_preds.append(result["preds"])
        self._test_targets.append(result["targets"])
        return result

    def _inverse(self, arr: np.ndarray) -> np.ndarray:
        return arr * self._scaler_std + self._scaler_mean

    def on_train_epoch_end(self):
        if self._train_preds:
            preds = torch.cat(self._train_preds).cpu().numpy()
            targets = torch.cat(self._train_targets).cpu().numpy()
            r2 = float(r2_score(targets, preds)) if len(targets) > 1 else 0.0
            preds_orig = self._inverse(preds)
            targets_orig = self._inverse(targets)
            rmse = float(np.sqrt(np.mean((preds_orig - targets_orig) ** 2)))
            mae = float(np.mean(np.abs(preds_orig - targets_orig)))
            self.log("train/rmse", rmse, prog_bar=False)
            self.log("train/r2", r2, prog_bar=False)
            self.log("train/mae", mae, prog_bar=False)
            # Store target value range per article in training set (for leak-split evaluation metrics)
            if self._train_article_ids:
                aids = torch.cat(self._train_article_ids).cpu().numpy()
                self._train_article_ranges = {}
                for aid in np.unique(aids):
                    mask = (aids == aid)
                    vals = targets_orig[mask]
                    self._train_article_ranges[int(aid)] = (float(vals.min()), float(vals.max()))
        self._train_preds.clear()
        self._train_targets.clear()
        self._train_article_ids.clear()

    def on_validation_epoch_end(self):
        if self._val_preds:
            preds = torch.cat(self._val_preds).cpu().numpy()
            targets = torch.cat(self._val_targets).cpu().numpy()
            r2 = float(r2_score(targets, preds)) if len(targets) > 1 else 0.0
            preds_orig = self._inverse(preds)
            targets_orig = self._inverse(targets)
            rmse = float(np.sqrt(np.mean((preds_orig - targets_orig) ** 2)))
            mae = float(np.mean(np.abs(preds_orig - targets_orig)))
            self.log("val/rmse", rmse, prog_bar=True)
            self.log("val/r2", r2, prog_bar=True)
            self.log("val/mae", mae, prog_bar=False)
            # Article-aware metrics (epoch-level only)
            if self._val_article_ids:
                aids = torch.cat(self._val_article_ids).cpu().numpy()
                hr = float('nan')
                # Article range hit rate (valid for leak-split: check if prediction falls within article training range)
                if self._train_article_ranges:
                    hr = _compute_article_range_hit_rate(preds_orig, aids, self._train_article_ranges)
                    if not np.isnan(hr):
                        self.log("val/article_range_hit_rate", hr, prog_bar=False)
                    # NARS (Normalized Article Range Score)
                    nars = _compute_article_nars(preds_orig, aids, self._train_article_ranges)
                    if not np.isnan(nars):
                        self.log("val/article_nars", nars, prog_bar=False)
                # Print article-aware metrics to console
                parts = []
                if not np.isnan(hr):
                    parts.append(f"RangeHR={hr:.4f}")
                if 'nars' in locals() and not np.isnan(nars):
                    parts.append(f"NARS={nars:.4f}")
                if parts:
                    print(f"  [Val] Article: {', '.join(parts)}")
        self._val_preds.clear()
        self._val_targets.clear()
        self._val_article_ids.clear()

    def on_test_epoch_end(self):
        if self._test_preds:
            preds = torch.cat(self._test_preds).cpu().numpy()
            targets = torch.cat(self._test_targets).cpu().numpy()
            r2 = float(r2_score(targets, preds)) if len(targets) > 1 else 0.0
            preds_orig = self._inverse(preds)
            targets_orig = self._inverse(targets)
            rmse = float(np.sqrt(np.mean((preds_orig - targets_orig) ** 2)))
            mae = float(np.mean(np.abs(preds_orig - targets_orig)))
            self.log("test/rmse", rmse)
            self.log("test/r2", r2)
            self.log("test/mae", mae)
        self._test_preds.clear()
        self._test_targets.clear()

    def on_save_checkpoint(self, checkpoint):
        trainable_keys = {n for n, p in self.named_parameters() if p.requires_grad}
        checkpoint["state_dict"] = {
            k: v for k, v in checkpoint["state_dict"].items()
            if k in trainable_keys
        }

    def configure_optimizers(self):
        lr = self.cfg["LR"]
        wd = self.cfg["WEIGHT_DECAY"]
        warmup = self.cfg["WARMUP_STEPS"]
        epochs = self.cfg["EPOCHS"]

        if self.encoder_name == "periogt":
            # PerioGT pretrained backbone uses backbone_lr, regression head uses lr
            backbone_lr = self.cfg.get("BACKBONE_LR", lr)
            backbone_params = []
            head_params = []
            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                if "periogt" in name:
                    backbone_params.append(param)
                else:
                    head_params.append(param)
            if backbone_params and head_params:
                optimizer = torch.optim.AdamW([
                    {"params": backbone_params, "lr": backbone_lr},
                    {"params": head_params, "lr": lr},
                ], weight_decay=wd)
            else:
                trainable = [p for p in self.parameters() if p.requires_grad]
                optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=wd)
        elif not self.frozen and (self.struct_encoder is not None or self.llm is not None):
            # LoRA params use backbone_lr, rest use lr
            backbone_lr = self.cfg["BACKBONE_LR"]
            backbone_params = []
            head_params = []
            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                if "_struct_backbone" in name or "_llm_backbone" in name:
                    backbone_params.append(param)
                else:
                    head_params.append(param)
            optimizer = torch.optim.AdamW([
                {"params": backbone_params, "lr": backbone_lr},
                {"params": head_params, "lr": lr},
            ], weight_decay=wd)
        else:
            trainable = [p for p in self.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=wd)

        trainer = self.trainer
        if trainer and trainer.estimated_stepping_batches:
            total_steps = trainer.estimated_stepping_batches
        else:
            total_steps = epochs * 100

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps - warmup), eta_min=1e-7
        )
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup
        )
        combined = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, scheduler], milestones=[warmup]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": combined,
                "interval": "step",
            },
        }


