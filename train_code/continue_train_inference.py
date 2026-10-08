"""
continue_train_inference.py — Continue-training fine-tuning inference script

Fine-tune a pre-trained model on a small amount of new data, then evaluate on test set.

Usage:
  1. Modify the parameter settings below: specify model weights directory and test set path
  2. Set training/validation data amounts in config/config_continue_train.py
  3. Run: python continue_train_inference.py

Pipeline:
  1. Load pre-trained model config (architecture and weights, or random initialization)
  2. Allocate training and validation sets from continue_train_dataset/{name}/{name}.json
  3. Use the JSON file under val_dataset/ as the test set
  4. Evaluate pre-fine-tuning baseline on the test set
  5. Fine-tune for a few epochs (with TensorBoard logging, recording loss / R2 / RMSE / MAE)
  6. Save best model weights
  7. Evaluate on test set and output CSV and JSON

Data sources:
  - continue_train_dataset/{name}/{name}.json: training + validation sets (allocated per config)
  - val_dataset/val_{name}_*.json: test set (used directly, no further splitting)
"""
import os
import sys
import json
import pickle
import copy
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from datetime import datetime
from typing import Dict, Any, List, Optional
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from target_scaling import (
    inject_scaler_config, inverse_from_parameters, validate_scaler_config,
    scaler_transform,
)

from config.path import (
    MODEL_PATHS, PERIOGT_CONFIG_PATH, VAL_DATASET_DIR,
    CONTINUE_TRAIN_DATASET_DIR, TB_LOG_DIR, RESULT_DIR, PROJECT_ROOT, ensure_dir,
)
from domain_embedding_cache import precompute_domain_embeddings, load_domain_cached_splits
from config.prompt import STRUCT_INPUT, build_prompt
from config import config_continue_train as ct_cfg
from utils import (
    compute_metrics, inverse_transform, load_ratio_encoding,
    fit_scaler, transform_targets, save_config_yaml,
    build_ratio_vector, load_trainable_weights,
    extract_article_ids, set_seed,
    periogt_collate_fn, precompute_periogt_data,
    scan_and_save_ratio_encoding,
)

# ======================== Inference Parameter Settings ========================
# Model weights folder (containing config.yaml, scaler.pkl, weights_seed*.pth, etc.)
WEIGHTS_DIR = "autodl-tmp/results/E_chemdfm_v1_5_8b_20260507_160338"

# Test set path (JSON file under val_dataset/, used directly as test set)
TEST_FILE = "val_dataset/val_E_article_1.json"

# Dataset name (used to find training data under continue_train_dataset/)
DATASET_NAME = "E_article"

# Input type
INPUT_TYPE = 2

# Seed used for inference (for weight loading and random seed setting)
SEED = 42

# Inference batch size
BATCH_SIZE = 4

# Used only when a requested domain LLM cache is missing.
EMBEDDING_BATCH_SIZE = 1

# Output directory
OUTPUT_DIR = "continue_train_results/continue"
# ==============================================================


# ======================== Helper Functions ========================

def load_config_from_yaml(yaml_path: str) -> Dict[str, Any]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_scaler(scaler_path: str):
    with open(scaler_path, "rb") as f:
        return pickle.load(f)


def is_struct_only(cfg: Dict[str, Any]) -> bool:
    return "STRUCT_ENCODER" in cfg and "ENABLE_HYBRID_ENCODING" not in cfg


def find_weight_file(weights_dir: str, seed: int) -> str:
    path = os.path.join(weights_dir, f"weights_seed{seed}.pth")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Weight file not found: {path}")
    return path


def load_external_llm(model_key: str, device: str = "cuda"):
    from models.base_model import LLM_REGISTRY
    llm_cls = LLM_REGISTRY[model_key]
    llm = llm_cls(MODEL_PATHS[model_key], device=device)
    llm.model.eval()
    return llm


def load_external_struct_encoder(encoder_name: str, device: str = "cuda"):
    from models.base_model import STRUCT_ENCODER_REGISTRY
    encoder_cls = STRUCT_ENCODER_REGISTRY[encoder_name]["encoder_cls"]
    return encoder_cls(MODEL_PATHS[encoder_name], device=device)


def encode_texts(llm, texts: List[str], device: str):
    inputs = llm.tokenize(texts)
    hidden = llm.get_hidden_states(inputs).float()
    mask = inputs["attention_mask"]
    return hidden.to(device), mask.to(device)


def encode_struct_batch(encoder, entries: List[Dict],
                        ratio_encoding: Optional[Dict], device: str):
    """Compute embeddings for a batch using the struct encoder, returns (emb, mask) both on device"""
    all_embs = []
    for entry in entries:
        emb = encoder.encode_entry(entry, ratio_info=ratio_encoding)
        all_embs.append(emb)
    max_n = max(e.size(0) for e in all_embs)
    dim = all_embs[0].size(-1)
    bsz = len(all_embs)
    struct_emb = torch.zeros(bsz, max_n, dim)
    struct_mask = torch.zeros(bsz, max_n, dtype=torch.long)
    for i, emb in enumerate(all_embs):
        n = emb.size(0)
        struct_emb[i, :n] = emb.cpu()
        struct_mask[i, :n] = 1
    return struct_emb.float().to(device), struct_mask.to(device)


def extract_smiles_data(entries: List[Dict], ratio_encoding: Optional[Dict]):
    """Extract SMILES lists and ratio vectors from a batch of data"""
    smiles_lists = []
    ratio_vectors_lists = []
    for entry in entries:
        chem = entry.get("chemical_composition", {})
        monomers = chem.get("monomers", [])
        is_homo = chem.get("is_homopolymer", False)
        sl, rvl = [], []
        for m in monomers:
            smi = m.get("smiles")
            if smi and smi.strip():
                sl.append(smi.strip())
                if ratio_encoding:
                    rv = build_ratio_vector(m, is_homo, ratio_encoding)
                    rvl.append(torch.tensor(rv, dtype=torch.float32))
        smiles_lists.append(sl)
        ratio_vectors_lists.append(rvl)
    return smiles_lists, ratio_vectors_lists


def infer_dataset_name(data_file: str) -> str:
    """
    Infer the dataset property name from the filename (i.e., the key in the properties dict).
    Supports the following naming patterns:
      - Tg_train.json         → Tg
      - val_Tg_1_train.json   → Tg
      - Tg.json               → Tg
    """
    basename = os.path.splitext(os.path.basename(data_file))[0]
    for suffix in ("_train", "_test"):
        if basename.endswith(suffix):
            basename = basename[:-len(suffix)]
            break
    m = re.match(r"^val_(.+?)(?:_\d+)?$", basename)
    if m:
        return m.group(1)
    return basename


# ======================== Datasets and DataLoader ========================

class ContinueTrainDataset(Dataset):
    """Continue-training dataset, supports precomputed embedding cache / PerioGT graph data / PolyBERT raw entries"""

    def __init__(self, entries: List[Dict], targets_scaled: np.ndarray,
                 dataset_name: str, input_type: int,
                 article_ids: Optional[np.ndarray] = None,
                 cached_embeddings: Optional[List[dict]] = None,
                 periogt_data: Optional[List[dict]] = None,
                 include_entry: bool = False):
        self.entries = entries
        self.targets = targets_scaled
        self.dataset_name = dataset_name
        self.input_type = input_type
        self.article_ids = article_ids
        self.cached_embeddings = cached_embeddings
        self.periogt_data = periogt_data
        self.include_entry = include_entry

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        item = {
            "target": torch.tensor(self.targets[idx], dtype=torch.float32),
        }
        if self.cached_embeddings is not None:
            item["text_hidden"] = self.cached_embeddings[idx]["hidden"]
            item["text_mask"] = self.cached_embeddings[idx]["mask"]
        else:
            entry = self.entries[idx]
            item["text"] = build_prompt(entry, self.dataset_name, self.input_type)
        if self.periogt_data is not None:
            item["periogt_item"] = self.periogt_data[idx]
        if self.include_entry:
            item["entry"] = self.entries[idx]
        if self.article_ids is not None:
            item["article_ids"] = torch.tensor(self.article_ids[idx], dtype=torch.long)
        return item


def continue_train_collate_fn(batch):
    """collate function supporting precomputed embeddings (padding), raw text, PerioGT graph data, and raw entries"""
    result = {
        "target": torch.stack([b["target"] for b in batch]),
    }
    if "text_hidden" in batch[0]:
        hiddens = [b["text_hidden"] for b in batch]
        masks = [b["text_mask"] for b in batch]
        max_len = max(h.size(0) for h in hiddens)
        dim = hiddens[0].size(-1)
        bsz = len(batch)
        padded_hidden = torch.zeros(bsz, max_len, dim)
        padded_mask = torch.zeros(bsz, max_len, dtype=masks[0].dtype)
        for i, (h, m) in enumerate(zip(hiddens, masks)):
            seq_len = h.size(0)
            padded_hidden[i, :seq_len] = h
            padded_mask[i, :seq_len] = m
        result["text_hidden"] = padded_hidden
        result["text_mask"] = padded_mask
    if "text" in batch[0]:
        result["texts"] = [b["text"] for b in batch]
    if "periogt_item" in batch[0]:
        pg_items = []
        for b in batch:
            d = {k: v for k, v in b["periogt_item"].items()}
            d["target"] = torch.tensor(0.0)
            pg_items.append(d)
        collated = periogt_collate_fn(pg_items)
        result["graph_data"] = {
            k: collated[k] for k in
            ['graphs', 'fp_1', 'md_1', 'fp_2', 'md_2',
             'ratio_vec_1', 'ratio_vec_2', 'global_type']
        }
    if "entry" in batch[0]:
        result["entries"] = [b["entry"] for b in batch]
    if "article_ids" in batch[0]:
        result["article_ids"] = torch.stack([b["article_ids"] for b in batch])
    return result


# ======================== Load Data from continue_train_dataset ========================

def load_continue_train_data(dataset_name: str) -> tuple:
    """
    Load all data from continue_train_dataset/{dataset_name}/{dataset_name}.json,
    and split train/val according to split_continue_train.pkl in the same directory.
    Returns (train_entries, train_targets, val_entries, val_targets, prop_name).
    If val list in pkl is empty, val_entries=[], val_targets=np.array([]).
    """
    json_path = os.path.join(CONTINUE_TRAIN_DATASET_DIR, dataset_name, f"{dataset_name}.json")
    if not os.path.exists(json_path):
        print(f"[Error] Training data not found: {json_path}")
        sys.exit(1)

    pkl_name = ct_cfg.CONTINUE_TRAIN_SPLIT_PKL_NAME
    pkl_path = os.path.join(CONTINUE_TRAIN_DATASET_DIR, dataset_name, pkl_name)
    if not os.path.exists(pkl_path):
        print(f"[Error] Split pkl not found: {pkl_path}")
        print("  Restore the matching published JSON/PKL pair. Offline recovery tools: "
              "../outputs/dataset_tools/README.md")
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    with open(pkl_path, "rb") as f:
        split = pickle.load(f)

    # Infer target property name
    prop_name = dataset_name
    if all_data:
        props = all_data[0].get("properties", {})
        if prop_name not in props:
            for key in props:
                if "value" in props[key]:
                    prop_name = key
                    break

    train_idx = list(split.get("train", []))
    val_idx = list(split.get("val", []))

    def _targets(indices):
        return np.array(
            [float(all_data[i]["properties"][prop_name]["value"][0]) for i in indices],
            dtype=np.float64,
        )

    train_entries = [all_data[i] for i in train_idx]
    train_targets = _targets(train_idx) if train_idx else np.array([])
    val_entries = [all_data[i] for i in val_idx]
    val_targets = _targets(val_idx) if val_idx else np.array([])

    print(f"[Split] Loaded {pkl_path}")
    print(f"  train={len(train_entries)}, val={len(val_entries)} (total in json={len(all_data)})")

    return train_entries, train_targets, val_entries, val_targets, prop_name


# ======================== Loss Computation ========================

def compute_weighted_loss(preds, targets, article_ids_batch, cfg,
                          article_ranges, device):
    """
    Compute the same weighted loss as the main training loop (MSE + article-aware losses),
    ensuring validation loss uses the same formula as training loss.
    Returns (total_loss, loss_dict).
    """
    from models.base_model import (
        article_consistency_loss, article_bias_loss,
        article_ranking_loss,
    )

    mse_fn = nn.MSELoss()
    w_mse = ct_cfg.MSE_WEIGHT
    total = w_mse * mse_fn(preds, targets)
    loss_dict = {"mse_loss": mse_fn(preds, targets).item()}

    if article_ids_batch is not None:
        article_ids_t = article_ids_batch.to(device)
        w_consist = ct_cfg.ARTICLE_CONSISTENCY_WEIGHT
        w_bias = ct_cfg.ARTICLE_BIAS_WEIGHT
        w_rank = ct_cfg.ARTICLE_RANKING_WEIGHT

        if w_consist > 0:
            l_consist = article_consistency_loss(preds, targets, article_ids_t)
            total = total + w_consist * l_consist
            loss_dict["article_consist_loss"] = l_consist.item()
        if w_bias > 0:
            l_bias = article_bias_loss(preds, targets, article_ids_t)
            total = total + w_bias * l_bias
            loss_dict["article_bias_loss"] = l_bias.item()
        if w_rank > 0:
            l_rank = article_ranking_loss(preds, targets, article_ids_t)
            total = total + w_rank * l_rank
            loss_dict["article_ranking_loss"] = l_rank.item()

    loss_dict["total_loss"] = total.item()
    return total, loss_dict


# ======================== Epoch-level Metric Computation ========================

def compute_epoch_metrics(model, data_loader, scaler_mean, scaler_std,
                          cfg, article_ranges, device, struct_only_flag,
                          llm=None, struct_encoder=None,
                          ratio_encoding=None):
    """
    Run forward pass over the entire loader, collect all predictions and targets,
    compute R2, RMSE, MAE (restored to original scale) and average loss.
    Returns (metrics_dict, avg_loss).
    """
    encoder_name = cfg.get("STRUCT_ENCODER", "polybert")
    frozen = cfg.get("FROZEN_BACKBONE", True)
    enable_hybrid = cfg.get("ENABLE_HYBRID_ENCODING", False)
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    total_n = 0

    with torch.no_grad():
        for batch in data_loader:
            targets = batch["target"].to(device)
            article_ids_batch = batch.get("article_ids")

            # ---- 1. Semantic stream ----
            text_hidden, text_mask = None, None
            if "text_hidden" in batch:
                text_hidden = batch["text_hidden"].float().to(device)
                text_mask = batch["text_mask"].to(device)
            elif struct_only_flag and encoder_name == "llm" and "texts" in batch:
                if llm is not None:
                    text_hidden, text_mask = encode_texts(llm, batch["texts"], device)
                else:
                    inputs = model.llm.tokenize(batch["texts"])
                    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                              for k, v in inputs.items()}
                    text_hidden = model.llm.get_hidden_states(inputs).float()
                    text_mask = inputs.get("attention_mask")
            elif not struct_only_flag and "texts" in batch:
                if llm is not None:
                    text_hidden, text_mask = encode_texts(llm, batch["texts"], device)
                else:
                    inputs = model.llm.tokenize(batch["texts"])
                    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                              for k, v in inputs.items()}
                    text_hidden = model.llm.get_hidden_states(inputs).float()
                    text_mask = inputs.get("attention_mask")

            # ---- 2. Structural stream ----
            struct_emb, struct_mask, graph_data = None, None, None
            if "graph_data" in batch:
                graph_data = {k: v.to(device) for k, v in batch["graph_data"].items()}
            elif encoder_name == "polybert" and struct_only_flag and "entries" in batch:
                if struct_encoder is not None:
                    struct_emb, struct_mask = encode_struct_batch(
                        struct_encoder, batch["entries"], ratio_encoding, device)
                else:
                    sl, rvl = extract_smiles_data(batch["entries"], ratio_encoding)
                    struct_emb, struct_mask = model._encode_smiles_batch(sl, rvl)

            # ---- 3. Forward ----
            if struct_only_flag:
                if encoder_name == "periogt":
                    preds = model(graph_data=graph_data)
                elif encoder_name == "llm":
                    preds = model(text_hidden=text_hidden, text_mask=text_mask)
                else:
                    preds = model(struct_emb=struct_emb, struct_mask=struct_mask)
            else:
                preds = model(text_hidden, text_mask, struct_emb, struct_mask,
                              graph_data=graph_data)

            v_loss, _ = compute_weighted_loss(
                preds, targets, article_ids_batch, cfg,
                article_ranges, device)
            total_loss += v_loss.item() * len(targets)
            total_n += len(targets)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = total_loss / max(total_n, 1)
    preds_scaled = np.concatenate(all_preds)
    targets_scaled = np.concatenate(all_targets)

    # Loss remains in standardized target space; metrics return to physical units.
    transform = cfg.get("_TARGET_TRANSFORM", "identity")
    preds_orig = inverse_from_parameters(preds_scaled, scaler_mean, scaler_std, transform)
    targets_orig = inverse_from_parameters(targets_scaled, scaler_mean, scaler_std, transform)

    metrics = compute_metrics(preds_orig, targets_orig)
    return metrics, avg_loss


# ======================== Continue-Training Loop ========================

def continue_train_loop(
    model, train_data: List[Dict], val_data: Optional[List[Dict]],
    train_targets_scaled: np.ndarray,
    val_targets_scaled: Optional[np.ndarray],
    dataset_name: str, cfg: Dict[str, Any],
    train_article_ids: Optional[np.ndarray] = None,
    val_article_ids: Optional[np.ndarray] = None,
    device: str = "cuda",
    llm=None,
    train_cached_embeddings: Optional[List[dict]] = None,
    val_cached_embeddings: Optional[List[dict]] = None,
    struct_encoder=None,
    ratio_encoding=None,
    train_periogt_data: Optional[List[dict]] = None,
    val_periogt_data: Optional[List[dict]] = None,
    tb_writer: Optional[SummaryWriter] = None,
    save_dir: Optional[str] = None,
):
    """
    Continue-training fine-tuning loop (native PyTorch loop).
    Validation loss uses the same weighted formula as training loss (MSE + article-aware losses).
    Supports TensorBoard logging and model weight saving.
    Returns the fine-tuned model.
    """
    model.train()
    model.to(device)

    # Override MLP dropout rate
    ct_dropout = ct_cfg.CONTINUE_TRAIN_MLP_DROPOUT
    if ct_dropout is not None:
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = ct_dropout

    # Get config
    epochs = ct_cfg.CONTINUE_TRAIN_EPOCHS
    patience = ct_cfg.CONTINUE_TRAIN_PATIENCE
    lr = ct_cfg.CONTINUE_TRAIN_LR
    batch_size = ct_cfg.CONTINUE_TRAIN_BATCH_SIZE
    clip_val = ct_cfg.CONTINUE_TRAIN_GRADIENT_CLIP_VAL
    accum_steps = ct_cfg.CONTINUE_TRAIN_GRADIENT_ACCUMULATION_STEPS

    # Determine model mode
    struct_only_flag = is_struct_only(cfg)
    encoder_name = cfg.get("STRUCT_ENCODER", "polybert")
    _need_entry = (struct_only_flag and encoder_name == "polybert")
    input_type = STRUCT_INPUT if (struct_only_flag and encoder_name == "llm") else INPUT_TYPE

    # Build training dataset
    train_ds = ContinueTrainDataset(
        train_data, train_targets_scaled, dataset_name, input_type,
        article_ids=train_article_ids,
        cached_embeddings=train_cached_embeddings,
        periogt_data=train_periogt_data,
        include_entry=_need_entry,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=continue_train_collate_fn, num_workers=0,
    )

    val_loader = None
    if val_data and val_targets_scaled is not None and len(val_data) > 0:
        val_ds = ContinueTrainDataset(
            val_data, val_targets_scaled, dataset_name, input_type,
            article_ids=val_article_ids,
            cached_embeddings=val_cached_embeddings,
            periogt_data=val_periogt_data,
            include_entry=_need_entry,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            collate_fn=continue_train_collate_fn, num_workers=0,
        )

    # Optimizer (only trainable parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-3)

    # Learning rate warmup scheduler
    warmup_steps = ct_cfg.CONTINUE_TRAIN_WARMUP_STEPS
    total_steps = len(train_loader) * epochs
    if warmup_steps > 0 and total_steps > warmup_steps:
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 1.0
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    scaler_mean = cfg.get("_SCALER_MEAN", 0.0)
    scaler_std = cfg.get("_SCALER_STD", 1.0)

    # Article ranges (for NARS loss)
    train_article_ranges = {}
    if train_article_ids is not None:
        scaler_mean = cfg.get("_SCALER_MEAN", 0.0)
        scaler_std = cfg.get("_SCALER_STD", 1.0)
        for i, aid in enumerate(train_article_ids):
            val_orig = float(inverse_from_parameters(
                train_targets_scaled[i], scaler_mean, scaler_std,
                cfg.get("_TARGET_TRANSFORM", "identity")))
            aid_int = int(aid)
            if aid_int not in train_article_ranges:
                train_article_ranges[aid_int] = (val_orig, val_orig)
            else:
                lo, hi = train_article_ranges[aid_int]
                train_article_ranges[aid_int] = (min(lo, val_orig), max(hi, val_orig))

    struct_only_flag = is_struct_only(cfg)

    print(f"\n[Continue-Train] epochs={epochs}, patience={patience}, "
          f"lr={lr}, batch_size={batch_size}, accum_steps={accum_steps}")
    print(f"  train={len(train_data)}, val={len(val_data) if val_data else 0}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            targets = batch["target"].to(device)
            article_ids_batch = batch.get("article_ids")

            # ---- 1. Semantic stream ----
            text_hidden, text_mask = None, None
            if "text_hidden" in batch:
                text_hidden = batch["text_hidden"].float().to(device)
                text_mask = batch["text_mask"].to(device)
            elif struct_only_flag and encoder_name == "llm" and "texts" in batch:
                if llm is not None:
                    text_hidden, text_mask = encode_texts(llm, batch["texts"], device)
                else:
                    inputs = model.llm.tokenize(batch["texts"])
                    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                              for k, v in inputs.items()}
                    text_hidden = model.llm.get_hidden_states(inputs).float()
                    text_mask = inputs.get("attention_mask")
            elif not struct_only_flag and "texts" in batch:
                if llm is not None:
                    text_hidden, text_mask = encode_texts(llm, batch["texts"], device)
                else:
                    inputs = model.llm.tokenize(batch["texts"])
                    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                              for k, v in inputs.items()}
                    text_hidden = model.llm.get_hidden_states(inputs).float()
                    text_mask = inputs.get("attention_mask")

            # ---- 2. Structural stream ----
            struct_emb, struct_mask, graph_data = None, None, None
            if "graph_data" in batch:
                graph_data = {k: v.to(device) for k, v in batch["graph_data"].items()}
            elif encoder_name == "polybert" and struct_only_flag and "entries" in batch:
                if struct_encoder is not None:
                    struct_emb, struct_mask = encode_struct_batch(
                        struct_encoder, batch["entries"], ratio_encoding, device)
                else:
                    sl, rvl = extract_smiles_data(batch["entries"], ratio_encoding)
                    struct_emb, struct_mask = model._encode_smiles_batch(sl, rvl)

            # ---- 3. Forward ----
            if struct_only_flag:
                if encoder_name == "periogt":
                    preds = model(graph_data=graph_data)
                elif encoder_name == "llm":
                    preds = model(text_hidden=text_hidden, text_mask=text_mask)
                else:
                    preds = model(struct_emb=struct_emb, struct_mask=struct_mask)
            else:
                preds = model(text_hidden, text_mask, struct_emb, struct_mask,
                              graph_data=graph_data)

            # Compute weighted loss
            loss, loss_dict = compute_weighted_loss(
                preds, targets, article_ids_batch, cfg,
                train_article_ranges, device)

            # Noise (optional) — added to predictions, not labels
            if ct_cfg.NOISE_ENABLED and epoch > 0:
                noise_std = ct_cfg.NOISE_STD
                if ct_cfg.NOISE_ANNEAL:
                    import math
                    progress = min(epoch / max(epochs, 1), 1.0)
                    noise_std = noise_std * (1 + math.cos(math.pi * progress)) / 2
                noise = torch.randn_like(preds) * noise_std + ct_cfg.NOISE_MEAN
                noisy_preds = preds + noise
                loss_noisy, _ = compute_weighted_loss(
                    noisy_preds, targets, article_ids_batch, cfg,
                    train_article_ranges, device)
                loss = (loss + loss_noisy) / 2

            # NaN/Inf protection: skip anomalous batches to prevent NaN contamination of model weights
            if not torch.isfinite(loss):
                optimizer.zero_grad()
                continue

            # Gradient accumulation
            scaled_loss = loss / accum_steps
            scaled_loss.backward()
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_params, clip_val if clip_val > 0 else float('inf'))
                if torch.isfinite(grad_norm):
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_loss / max(n_batches, 1)

        # ---- Epoch-level metric computation ----
        # Training set metrics
        train_metrics, _ = compute_epoch_metrics(
            model, train_loader, scaler_mean, scaler_std,
            cfg, train_article_ranges, device, struct_only_flag, llm=llm,
            struct_encoder=struct_encoder, ratio_encoding=ratio_encoding)

        if tb_writer is not None:
            tb_writer.add_scalar("train/loss", avg_train_loss, epoch)
            tb_writer.add_scalar("train/R2", train_metrics["R2"], epoch)
            tb_writer.add_scalar("train/RMSE", train_metrics["RMSE"], epoch)
            tb_writer.add_scalar("train/MAE", train_metrics["MAE"], epoch)

        # Validation
        val_loss = float('inf')
        if val_loader is not None:
            val_metrics, val_loss = compute_epoch_metrics(
                model, val_loader, scaler_mean, scaler_std,
                cfg, train_article_ranges, device, struct_only_flag, llm=llm,
                struct_encoder=struct_encoder, ratio_encoding=ratio_encoding)

            if tb_writer is not None:
                tb_writer.add_scalar("val/loss", val_loss, epoch)
                tb_writer.add_scalar("val/R2", val_metrics["R2"], epoch)
                tb_writer.add_scalar("val/RMSE", val_metrics["RMSE"], epoch)
                tb_writer.add_scalar("val/MAE", val_metrics["MAE"], epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                # Save best weights
                if save_dir:
                    best_path = os.path.join(save_dir, f"weights_seed{SEED}.pth")
                    torch.save(best_state, best_path)
            else:
                patience_counter += 1

            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f"  Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.6f}, "
                      f"val_loss={val_loss:.6f}, "
                      f"train_R2={train_metrics['R2']:.4f}, val_R2={val_metrics['R2']:.4f}, "
                      f"patience={patience_counter}/{patience}")

            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        else:
            # No validation set: save last state
            best_state = copy.deepcopy(model.state_dict())
            if save_dir:
                best_path = os.path.join(save_dir, f"weights_seed{SEED}.pth")
                torch.save(best_state, best_path)
            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f"  Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.6f}, "
                      f"train_R2={train_metrics['R2']:.4f}")

    # Load best state
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    return model


# ======================== Inference Core Loop ========================

def run_inference(model, data: List[Dict], dataset_name: str,
                  cfg: Dict[str, Any], device: str,
                  llm=None,
                  cached_embeddings: Optional[List[dict]] = None,
                  struct_encoder=None,
                  ratio_encoding: Optional[Dict] = None,
                  periogt_data: Optional[List[dict]] = None) -> np.ndarray:
    """
    Run inference on all data batch by batch, returning normalized predictions.
    Supports semantic stream, structural stream (periogt/polybert/llm), and hybrid stream.
    """
    struct_only_flag = is_struct_only(cfg)
    frozen = cfg.get("FROZEN_BACKBONE", True)
    enable_hybrid = cfg.get("ENABLE_HYBRID_ENCODING", False)
    encoder_name = cfg.get("STRUCT_ENCODER", "polybert")
    n = len(data)
    all_preds = []

    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        batch_entries = data[start:end]

        with torch.no_grad():
            # ---- 1. Semantic stream hidden states ----
            text_hidden, text_mask = None, None

            if struct_only_flag:
                if encoder_name == "llm":
                    texts = [build_prompt(e, dataset_name, STRUCT_INPUT) for e in batch_entries]
                    if cached_embeddings is not None:
                        text_hidden, text_mask = _unpack_cached(
                            cached_embeddings, start, end, device)
                    elif frozen and llm is not None:
                        text_hidden, text_mask = encode_texts(llm, texts, device)
                    elif model.llm is not None:
                        inputs = model.llm.tokenize(texts)
                        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                  for k, v in inputs.items()}
                        text_hidden = model.llm.get_hidden_states(inputs).float()
                        text_mask = inputs.get("attention_mask")
            else:
                texts = [build_prompt(e, dataset_name, INPUT_TYPE) for e in batch_entries]
                if cached_embeddings is not None:
                    text_hidden, text_mask = _unpack_cached(
                        cached_embeddings, start, end, device)
                elif frozen and llm is not None:
                    text_hidden, text_mask = encode_texts(llm, texts, device)
                elif model.llm is not None:
                    inputs = model.llm.tokenize(texts)
                    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                              for k, v in inputs.items()}
                    text_hidden = model.llm.get_hidden_states(inputs).float()
                    text_mask = inputs.get("attention_mask")

            # ---- 2. Structural stream data ----
            struct_emb, struct_mask, graph_data = None, None, None

            _need_struct = (
                (struct_only_flag and encoder_name != "llm") or
                (not struct_only_flag and enable_hybrid)
            )

            if _need_struct:
                if encoder_name == "periogt" and periogt_data is not None:
                    items = []
                    for item in periogt_data[start:end]:
                        d = {k: v for k, v in item.items()}
                        d["target"] = torch.tensor(0.0)
                        items.append(d)
                    collated = periogt_collate_fn(items)
                    graph_data = {
                        k: collated[k].to(device) for k in
                        ['graphs', 'fp_1', 'md_1', 'fp_2', 'md_2',
                         'ratio_vec_1', 'ratio_vec_2', 'global_type']
                    }
                elif encoder_name == "polybert":
                    if frozen and struct_encoder is not None:
                        struct_emb, struct_mask = encode_struct_batch(
                            struct_encoder, batch_entries, ratio_encoding, device)
                    elif hasattr(model, '_encode_smiles_batch'):
                        sl, rvl = extract_smiles_data(batch_entries, ratio_encoding)
                        struct_emb, struct_mask = model._encode_smiles_batch(sl, rvl)

            # ---- 3. Model forward inference ----
            if struct_only_flag:
                if encoder_name == "periogt":
                    preds = model(graph_data=graph_data)
                elif encoder_name == "llm":
                    preds = model(text_hidden=text_hidden, text_mask=text_mask)
                else:
                    preds = model(struct_emb=struct_emb, struct_mask=struct_mask)
            else:
                preds = model(text_hidden, text_mask, struct_emb, struct_mask,
                              graph_data=graph_data)

        all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_preds)


def _unpack_cached(cached_embeddings, start, end, device):
    """Unpack precomputed embeddings from cache and pad to uniform length"""
    hiddens = [cached_embeddings[i]["hidden"] for i in range(start, end)]
    masks = [cached_embeddings[i]["mask"] for i in range(start, end)]
    max_len = max(h.size(0) for h in hiddens)
    dim = hiddens[0].size(-1)
    bsz = len(hiddens)
    text_hidden = torch.zeros(bsz, max_len, dim)
    text_mask = torch.zeros(bsz, max_len, dtype=masks[0].dtype)
    for i, (h, m) in enumerate(zip(hiddens, masks)):
        sl = h.size(0)
        text_hidden[i, :sl] = h
        text_mask[i, :sl] = m
    return text_hidden.float().to(device), text_mask.to(device)


# ======================== Input Type Compatibility Check ========================

_ANALYSIS_TYPES = {
    3: "analysis_with_hierarchical_structure",
    4: "analysis_without_hierarchical_structure",
    5: "analysis_with_hierarchical_structure",
    6: "analysis_without_hierarchical_structure",
    7: "analysis_with_hierarchical_structure",
    8: "analysis_without_hierarchical_structure",
}


def validate_input_type(entries: List[Dict], input_type: int, label: str = "data"):
    required_field = _ANALYSIS_TYPES.get(input_type)
    if required_field is None:
        return
    sample = entries[0] if entries else {}
    if required_field not in sample or not sample[required_field]:
        print(f"[Error] INPUT_TYPE={input_type} requires field '{required_field}', "
              f"but {label} entries lack this field.")
        print(f"  Available fields: {list(sample.keys())}")
        print(f"  Supported types for this data: 1, 2, 9")
        sys.exit(1)


# ======================== Main Flow ========================

def main(*, config_path=None, output_dir_override=None):
    weights_dir = WEIGHTS_DIR
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---- Validate config file ----
    config_yaml_path = config_path or os.path.join(weights_dir, "config.yaml")
    if not os.path.exists(config_yaml_path):
        print(f"[Error] config.yaml not found in {weights_dir}")
        sys.exit(1)

    if not os.path.exists(TEST_FILE):
        print(f"[Error] Test file not found: {TEST_FILE}")
        sys.exit(1)

    # ---- Load config ----
    cfg = load_config_from_yaml(config_yaml_path)
    cfg.pop("USE_CHAT_TEMPLATE", None)  # Historical configs cannot change raw input formatting.
    input_type = STRUCT_INPUT if (is_struct_only(cfg) and cfg.get("STRUCT_ENCODER") == "llm") else INPUT_TYPE
    if ct_cfg.LOG_TARGET_DATASETS is not None:
        cfg["LOG_TARGET_DATASETS"] = ct_cfg.LOG_TARGET_DATASETS
    dataset_name = DATASET_NAME if DATASET_NAME else infer_dataset_name(TEST_FILE)

    # ---- Scaler handling ----
    if ct_cfg.RANDOM_INIT_WEIGHTS:
        # Random initialization mode: re-fit scaler from training data
        print("[Scaler] RANDOM_INIT_WEIGHTS=True, will fit scaler from training data")
        scaler = None  # Delay fitting until after training data is loaded
    else:
        scaler_path = os.path.join(weights_dir, "scaler.pkl")
        if not os.path.exists(scaler_path):
            scale_property = "E" if dataset_name == "E_article" else dataset_name
            scaler_path = os.path.join(weights_dir, f"scaler_{scale_property}.pkl")
        if not os.path.exists(scaler_path):
            print(f"[Error] scaler.pkl not found in {weights_dir}")
            sys.exit(1)
        scaler = load_scaler(scaler_path)

    print(f"[Config] {config_yaml_path}")
    print(f"[Dataset] {dataset_name}")
    print(f"[Test File] {TEST_FILE}")

    # ---- Output directory ----
    run_name = f"{dataset_name}_ct_{timestamp}"
    output_dir = output_dir_override or os.path.join(OUTPUT_DIR, run_name)
    ensure_dir(output_dir)

    # ---- TensorBoard ----
    tb_log_dir = os.path.join(TB_LOG_DIR, "continue_train", run_name)
    ensure_dir(tb_log_dir)
    tb_writer = SummaryWriter(log_dir=tb_log_dir)
    print(f"[TensorBoard] {tb_log_dir}")

    # ---- Load test data ----
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    validate_input_type(test_data, input_type, label="test")

    # Infer the actual property key (may differ from DATASET_NAME when the
    # continue_train_dataset folder name != the properties dict key, e.g.
    # DATASET_NAME="E_article" but the key in JSON is "E")
    test_prop_key = dataset_name
    if test_data:
        test_props = test_data[0].get("properties", {})
        if test_prop_key not in test_props:
            for key in test_props:
                if "value" in test_props[key]:
                    test_prop_key = key
                    break

    test_targets_raw = np.array(
        [float(e["properties"][test_prop_key]["value"][0]) for e in test_data],
        dtype=np.float64,
    )

    # ---- Load and split train/val from continue_train_dataset (based on pkl indices) ----
    all_train_entries, all_train_targets_raw, all_val_entries, all_val_targets_raw, prop_name = \
        load_continue_train_data(dataset_name)
    if prop_name != test_prop_key:
        raise ValueError(f"Training target {prop_name!r} differs from test target {test_prop_key!r}")
    validate_input_type(all_train_entries, input_type, label="train")
    if all_val_entries:
        validate_input_type(all_val_entries, input_type, label="val")

    print(f"\n[Train] {len(all_train_entries)} entries")
    print(f"[Val] {len(all_val_entries)} entries")

    if len(all_train_entries) == 0:
        print("[Error] No training data available.")
        sys.exit(1)

    # ---- Scaler fitting (random init mode) or use existing scaler ----
    if scaler is None:
        scaler = fit_scaler(all_train_targets_raw, list(range(len(all_train_targets_raw))),
                            cfg=cfg, dataset_name=prop_name)
        print(f"[Scaler] Fitted from training data: transform={scaler_transform(scaler)}, "
              f"mean={scaler.mean_[0]:.4f}, std={scaler.scale_[0]:.4f}")
    else:
        validate_scaler_config(scaler, cfg, prop_name, context="Continue training with pretrained head")
    inject_scaler_config(cfg, scaler, prop_name)
    # Store the exact scaler beside these weights even when inherited unchanged.
    scaler_save_path = os.path.join(output_dir, "scaler.pkl")
    with open(scaler_save_path, "wb") as f:
        pickle.dump(scaler, f)
    save_config_yaml(cfg, os.path.join(output_dir, "config.yaml"))

    # ---- Article ID mapping ----
    combined_article_ids = np.array(extract_article_ids(all_train_entries + all_val_entries))
    train_article_ids = combined_article_ids[:len(all_train_entries)]

    val_article_ids = None
    all_val_targets_scaled = None
    if all_val_entries:
        val_article_ids = combined_article_ids[len(all_train_entries):]
        all_val_targets_scaled = transform_targets(scaler, all_val_targets_raw)

    print(f"\n[Total] train={len(all_train_entries)}, val={len(all_val_entries)}, "
          f"test={len(test_data)}")

    # ---- Normalize target values ----
    all_train_targets_scaled = transform_targets(scaler, all_train_targets_raw)
    test_targets_scaled = transform_targets(scaler, test_targets_raw)

    # ---- Device ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    # ---- Load model ----
    set_seed(SEED)
    struct_only_flag = is_struct_only(cfg)
    frozen = cfg.get("FROZEN_BACKBONE", True)

    # ratio_encoding path: prefer the original path recorded in config.yaml during training,
    # then check the weights_dir copy, then search parent directories of weights_dir (grid search
    # saves ratio_encoding.json in the grid experiment directory, not in the per-trial directory)
    _ratio_candidates = [
        cfg.get("_RATIO_ENCODING_PATH"),                   # Original path written to config.yaml during training
        os.path.join(weights_dir, "ratio_encoding.json"),  # Inside weights_dir
    ]
    # Search up to 3 parent directories (covering grid experiment directory)
    _parent = weights_dir
    for _ in range(3):
        _parent = os.path.dirname(_parent)
        if _parent and _parent != os.path.dirname(_parent):
            _ratio_candidates.append(os.path.join(_parent, "ratio_encoding.json"))
    ratio_path = None
    for _cand in _ratio_candidates:
        if _cand and os.path.exists(_cand):
            ratio_path = _cand
            break
    if ratio_path:
        cfg["_RATIO_ENCODING_PATH"] = ratio_path

    llm = None
    need_external_llm = frozen and (
        (not struct_only_flag) or
        (struct_only_flag and cfg.get("STRUCT_ENCODER") == "llm")
    )
    train_cached_embeddings = None
    val_cached_embeddings = None
    test_cached_embeddings = None
    if need_external_llm:
        # Missing caches are populated before constructing the regression model;
        # complete caches never load the external backbone.
        model_key = cfg["MODEL_TYPE"]
        precompute_domain_embeddings(
            dataset_name, model_key, [input_type], batch_size=EMBEDDING_BATCH_SIZE,
            test_file=TEST_FILE,
        )
        train_cached_embeddings, val_cached_embeddings, test_cached_embeddings = \
            load_domain_cached_splits(
                dataset_name, model_key, input_type, test_file=TEST_FILE,
                split_pkl_name=ct_cfg.CONTINUE_TRAIN_SPLIT_PKL_NAME,
            )
        print(f"[Domain cache] Loaded train={len(train_cached_embeddings)}, "
              f"val={len(val_cached_embeddings)}, test={len(test_cached_embeddings)}")

    encoder_name = cfg.get("STRUCT_ENCODER", "polybert")
    enable_hybrid = cfg.get("ENABLE_HYBRID_ENCODING", False)

    # Structural encoder (needed for frozen polybert)
    struct_encoder_ext = None
    need_external_struct = frozen and (
        (enable_hybrid and encoder_name == "polybert") or
        (struct_only_flag and encoder_name == "polybert")
    )
    if need_external_struct:
        print(f"[Struct] Loading {encoder_name}...")
        struct_encoder_ext = load_external_struct_encoder(encoder_name, device=device)

    # ratio_encoding
    ratio_encoding = load_ratio_encoding(ratio_path) if ratio_path else None

    # PerioGT graph data
    train_periogt_data = None
    val_periogt_data = None
    test_periogt_data = None
    need_periogt = (
        (struct_only_flag and encoder_name == "periogt") or
        (enable_hybrid and encoder_name == "periogt")
    )
    if need_periogt:
        # ratio_encoding: prefer loading from training-time file (to ensure dimensions match model weights);
        # as fallback, scan from main dataset (must ensure scan range matches training-time scan)
        if ratio_encoding is None:
            fallback_ratio_path = os.path.join(output_dir, "ratio_encoding.json")
            print(f"[Ratio] ratio_encoding.json not found, scanning main dataset/{prop_name}/ ...")
            scan_and_save_ratio_encoding([prop_name], save_path=fallback_ratio_path)
            ratio_encoding = load_ratio_encoding(fallback_ratio_path)
            cfg["_RATIO_ENCODING_PATH"] = fallback_ratio_path
        else:
            print(f"[Ratio] Loaded ratio_encoding from: {ratio_path}")
        print("[PerioGT] Preparing graph data...")
        use_prompt = cfg.get("USE_PERIOGT_PROMPT", True)
        train_periogt_data = precompute_periogt_data(
            prop_name, all_train_entries, ratio_encoding,
            use_prompt=use_prompt)
        if all_val_entries:
            val_periogt_data = precompute_periogt_data(
                prop_name, all_val_entries, ratio_encoding,
                use_prompt=use_prompt)
        test_periogt_data = precompute_periogt_data(
            prop_name, test_data, ratio_encoding,
            use_prompt=use_prompt)

    from models.base_model import PropertyPredictionModel, StructureOnlyModel

    # Cache misses may instantiate a backbone; warm/cold caches must not alter
    # the random initialization of the trainable head or training RNG stream.
    set_seed(SEED)
    print("[Model] Building model...")
    if struct_only_flag:
        model = StructureOnlyModel(cfg)
    else:
        model = PropertyPredictionModel(cfg)
    model.to(device)

    if ct_cfg.RANDOM_INIT_WEIGHTS:
        print("[Weights] Using random initialization (RANDOM_INIT_WEIGHTS=True)")
    else:
        weight_path = find_weight_file(weights_dir, SEED)
        print(f"[Weights] Loading {weight_path}")
        load_trainable_weights(model, weight_path, device=device)

    # ---- Pre-finetune evaluation ----
    print("\n[Pre-FT Evaluation] Testing before fine-tuning...")
    model.eval()
    preds_scaled_pre = run_inference(
        model, test_data, prop_name, cfg, device,
        llm=llm, cached_embeddings=test_cached_embeddings,
        struct_encoder=struct_encoder_ext, ratio_encoding=ratio_encoding,
        periogt_data=test_periogt_data)
    preds_orig_pre = inverse_transform(scaler, preds_scaled_pre)
    metrics_pre = compute_metrics(preds_orig_pre, test_targets_raw)
    print(f"  R2: {metrics_pre['R2']:.4f}, RMSE: {metrics_pre['RMSE']:.4f}, MAE: {metrics_pre['MAE']:.4f}")

    # TensorBoard: pre-finetuning metrics
    if tb_writer is not None:
        tb_writer.add_scalar("test/R2_before_ft", metrics_pre["R2"], 0)
        tb_writer.add_scalar("test/RMSE_before_ft", metrics_pre["RMSE"], 0)
        tb_writer.add_scalar("test/MAE_before_ft", metrics_pre["MAE"], 0)

    # ---- Continue-training fine-tuning ----
    inject_scaler_config(cfg, scaler, prop_name)

    model = continue_train_loop(
        model, all_train_entries, all_val_entries if all_val_entries else None,
        all_train_targets_scaled,
        all_val_targets_scaled,
        prop_name, cfg,
        train_article_ids=train_article_ids,
        val_article_ids=val_article_ids,
        device=device,
        llm=llm,
        train_cached_embeddings=train_cached_embeddings,
        val_cached_embeddings=val_cached_embeddings,
        struct_encoder=struct_encoder_ext,
        ratio_encoding=ratio_encoding,
        train_periogt_data=train_periogt_data,
        val_periogt_data=val_periogt_data,
        tb_writer=tb_writer,
        save_dir=output_dir,
    )

    # ---- Post-finetune evaluation ----
    print("\n[Post-FT Evaluation] Testing after fine-tuning...")
    model.eval()
    preds_scaled_post = run_inference(
        model, test_data, prop_name, cfg, device,
        llm=llm, cached_embeddings=test_cached_embeddings,
        struct_encoder=struct_encoder_ext, ratio_encoding=ratio_encoding,
        periogt_data=test_periogt_data)
    preds_orig_post = inverse_transform(scaler, preds_scaled_post)
    metrics_post = compute_metrics(preds_orig_post, test_targets_raw)
    print(f"  R2: {metrics_post['R2']:.4f}, RMSE: {metrics_post['RMSE']:.4f}, MAE: {metrics_post['MAE']:.4f}")

    # TensorBoard: post-finetuning metrics
    if tb_writer is not None:
        tb_writer.add_scalar("test/R2_after_ft", metrics_post["R2"], 0)
        tb_writer.add_scalar("test/RMSE_after_ft", metrics_post["RMSE"], 0)
        tb_writer.add_scalar("test/MAE_after_ft", metrics_post["MAE"], 0)
        tb_writer.close()

    # ---- Free up GPU memory ----
    if llm is not None:
        del llm
    if struct_encoder_ext is not None:
        del struct_encoder_ext
    torch.cuda.empty_cache()

    # ---- Save results ----
    # Refresh after ratio-path resolution and actual continue-training options.
    cfg["_CONTINUE_TRAIN_DROPOUT_OVERRIDE"] = ct_cfg.CONTINUE_TRAIN_MLP_DROPOUT
    save_config_yaml(cfg, os.path.join(output_dir, "config.yaml"))
    # Save config snapshot
    cfg_snapshot = {
        "continue_train_config": {
            key: value for key, value in vars(ct_cfg).items() if key.isupper()
        },
        "weights_dir": weights_dir,
        "test_file": os.path.relpath(os.path.abspath(TEST_FILE), PROJECT_ROOT).replace("\\", "/"),
        "dataset_name": dataset_name,
        "target_property": prop_name,
        "target_transform": scaler_transform(scaler),
        "effective_log_target_datasets": cfg.get("LOG_TARGET_DATASETS", []),
        "input_format": "raw_text",
        "input_type": input_type,
        "seed": SEED,
        "inference_batch_size": BATCH_SIZE,
        "embedding_batch_size": EMBEDDING_BATCH_SIZE,
        "n_train": len(all_train_entries),
        "n_val": len(all_val_entries),
        "n_test": len(test_data),
    }
    config_save_path = os.path.join(output_dir, "continue_train_config.json")
    with open(config_save_path, "w", encoding="utf-8") as f:
        json.dump(cfg_snapshot, f, ensure_ascii=False, indent=2)

    # JSON: predictions
    predictions = [
        {
            "index": i,
            "true_value": float(test_targets_raw[i]),
            "predicted_value_before_ft": float(preds_orig_pre[i]),
            "predicted_value_after_ft": float(preds_orig_post[i]),
        }
        for i in range(len(test_targets_raw))
    ]
    json_path = os.path.join(output_dir, "predictions.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"\n[Saved] {json_path}")

    # CSV: metric comparison
    csv_rows = [
        {"stage": "before_ft", **metrics_pre},
        {"stage": "after_ft", **metrics_post},
        {
            "stage": "improvement",
            "R2": metrics_post["R2"] - metrics_pre["R2"],
            "RMSE": metrics_pre["RMSE"] - metrics_post["RMSE"],
            "MAE": metrics_pre["MAE"] - metrics_post["MAE"],
        },
    ]
    csv_path = os.path.join(output_dir, "metrics.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[Saved] {csv_path}")

    print(f"\n{'='*50}")
    print(f"  Continue-Train Results Summary ({dataset_name})")
    print(f"  Before FT: R2={metrics_pre['R2']:.4f}, RMSE={metrics_pre['RMSE']:.4f}, MAE={metrics_pre['MAE']:.4f}")
    print(f"  After  FT: R2={metrics_post['R2']:.4f}, RMSE={metrics_post['RMSE']:.4f}, MAE={metrics_post['MAE']:.4f}")
    print(f"  ΔR2={metrics_post['R2']-metrics_pre['R2']:+.4f}, "
          f"ΔRMSE={metrics_pre['RMSE']-metrics_post['RMSE']:+.4f}, "
          f"ΔMAE={metrics_pre['MAE']-metrics_post['MAE']:+.4f}")
    print(f"  Output: {output_dir}")
    print(f"  TensorBoard: {tb_log_dir}")
    print(f"{'='*50}")
    return output_dir


def run_from_saved_config(run_dir, weights_dir_override=None, output_dir=None, *, seed_override=None):
    """Replay one domain run using its saved architecture and run snapshot.

    Intended for separate worker processes. Global settings are restored even
    after an error; this function is not safe for concurrent Python threads.
    Missing settings in historical snapshots retain current config defaults
    and are explicitly reported (older snapshots did not save every option).
    seed_override pairs main-task weights_seedN.pth with domain training seed N;
    it leaves the saved split and all training hyperparameters unchanged.
    """
    run_dir = os.path.abspath(run_dir)
    with open(os.path.join(run_dir, "continue_train_config.json"), encoding="utf-8") as handle:
        saved = json.load(handle)
    required = ("weights_dir", "test_file", "dataset_name", "input_type", "seed", "continue_train_config")
    missing = [key for key in required if key not in saved]
    if missing:
        raise ValueError(f"Incomplete domain snapshot in {run_dir}: {missing}")

    def project_path(path):
        path = os.fspath(path).replace("\\", os.sep)
        return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)

    module = sys.modules[__name__]
    overrides = {
        "WEIGHTS_DIR": project_path(weights_dir_override or saved["weights_dir"]),
        "TEST_FILE": project_path(saved["test_file"]),
        "DATASET_NAME": saved["dataset_name"],
        "INPUT_TYPE": saved["input_type"],
        "SEED": int(saved["seed"] if seed_override is None else seed_override),
        "BATCH_SIZE": saved.get("inference_batch_size", BATCH_SIZE),
        "EMBEDDING_BATCH_SIZE": saved.get("embedding_batch_size", EMBEDDING_BATCH_SIZE),
    }
    old_globals = {key: getattr(module, key) for key in overrides}
    ct_overrides = saved["continue_train_config"]
    unknown = [key for key in ct_overrides if not hasattr(ct_cfg, key)]
    if unknown:
        raise ValueError(f"Unsupported saved continue-training options: {unknown}")
    old_ct = {key: getattr(ct_cfg, key) for key in ct_overrides}
    defaults = sorted(key for key in vars(ct_cfg) if key.isupper() and key not in ct_overrides)
    if defaults:
        print(f"[Replay] Historical snapshot omitted these options; retaining config defaults: {defaults}")
    try:
        for key, value in overrides.items():
            setattr(module, key, value)
        for key, value in ct_overrides.items():
            setattr(ct_cfg, key, value)
        return main(config_path=os.path.join(run_dir, "config.yaml"),
                    output_dir_override=os.path.abspath(output_dir or run_dir))
    finally:
        for key, value in old_globals.items():
            setattr(module, key, value)
        for key, value in old_ct.items():
            setattr(ct_cfg, key, value)


if __name__ == "__main__":
    main()
