"""
utils.py — General utility functions: datasets, embedding caching, evaluation metrics, visualization, etc.
"""
import os
import gc
import json
import pickle
import random
import tempfile
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from target_scaling import fit_target_scaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple, Union

from config.path import (
    DATASET_PATHS, get_split_pkl_path, EMBEDDING_CACHE_DIR,
    RESULT_DIR, ensure_dir, MODEL_PATHS,
)
from config.prompt import STRUCT_INPUT, build_prompt
from models.llm_tokenization import llm_cache_key


# ======================== Ratio Information Encoding ========================

def scan_and_save_ratio_encoding(datasets: List[str] = None,
                                  save_path: Optional[str] = None):
    """
    Scan all datasets, collect all ratio_unit values, build and save one-hot encoding mapping.
    save_path: save path, must be explicitly provided by caller (saved in the experiment subdirectory).
    """
    if save_path is None:
        raise ValueError("save_path must be explicitly provided (ratio_encoding.json should be saved in the experiment directory).")
    if datasets is None:
        datasets = list(DATASET_PATHS.keys())

    all_units = set()
    for ds_name in datasets:
        data = load_dataset(ds_name)
        for entry in data:
            chem = entry.get("chemical_composition", {})
            for m in chem.get("monomers", []):
                ru = m.get("ratio_unit")
                if ru and ru.strip():
                    all_units.add(ru.strip())
    # Ensure mol% always exists (default unit for homopolymers)
    all_units.add("mol%")
    all_units = sorted(all_units)

    unit_to_idx = {u: i for i, u in enumerate(all_units)}
    # Vector dimensions: ratio_value(1) + has_value(1) + unit_onehot(n_units)
    vector_dim = 2 + len(all_units)

    encoding = {
        "units": all_units,
        "unit_to_idx": unit_to_idx,
        "n_units": len(all_units),
        "vector_dim": vector_dim,
        "unit_to_onehot": unit_to_idx,  # Compatible with PolyBERT interface
        "ratio_dim": vector_dim,
    }

    ensure_dir(os.path.dirname(save_path))
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(encoding, f, ensure_ascii=False, indent=2)
    print(f"  Ratio encoding saved: {save_path} ({len(all_units)} units, dim={vector_dim})")
    return encoding


def load_ratio_encoding(load_path: str) -> Dict[str, Any]:
    """Load ratio information one-hot encoding mapping. load_path must be explicitly provided by caller."""
    with open(load_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _parse_ratio_value(rv) -> float:
    """Parse ratio_value to float; supports range strings like '97-99' (takes the mean)."""
    if isinstance(rv, (int, float)):
        return float(rv)
    s = str(rv).strip()
    # Handle 'a-b' range format (note: exclude negative sign; only treat '-' as range separator when not at first char)
    dash_pos = s.find('-', 1)
    if dash_pos > 0:
        try:
            lo = float(s[:dash_pos])
            hi = float(s[dash_pos + 1:])
            return (lo + hi) / 2.0
        except ValueError:
            pass
    return float(s)


def build_ratio_vector(monomer: Dict, is_homopolymer: bool,
                       ratio_encoding: Dict) -> List[float]:
    """
    Build ratio information vector for a single monomer: [ratio_value, has_value, ratio_unit_onehot...]
    """
    n_units = ratio_encoding["n_units"]
    unit_to_idx = ratio_encoding["unit_to_idx"]

    if is_homopolymer:
        ratio_value = 100.0
        has_value = 1.0
        unit_str = "mol%"
    else:
        rv = monomer.get("ratio_value")
        ru = monomer.get("ratio_unit")
        if rv is not None:
            ratio_value = _parse_ratio_value(rv)
            has_value = 1.0
        else:
            ratio_value = 0.0
            has_value = 0.0
        unit_str = ru.strip() if ru else ""

    unit_onehot = [0.0] * n_units
    if unit_str in unit_to_idx:
        unit_onehot[unit_to_idx[unit_str]] = 1.0

    return [ratio_value, has_value] + unit_onehot


# ======================== Article-Aware Utilities ========================

def extract_article_ids(data: List[Dict]) -> List[int]:
    """
    Extract article group IDs using DOI first, then title as fallback.
    If both are empty, treat the record as a standalone article.
    Returns an integer list of the same length as data, where entries from the same article share the same ID.
    """
    title_to_id = {}
    article_ids = []
    for entry in data:
        title = str(entry.get("doi") or "").strip().lower() or entry.get("title") or ""
        title = str(title).strip()
        if not title:
            # No title or doi, treat as standalone article
            title = f"__singleton_{len(title_to_id) + len(article_ids)}__"
        if title not in title_to_id:
            title_to_id[title] = len(title_to_id)
        article_ids.append(title_to_id[title])
    return article_ids


class ArticleGroupedBatchSampler:
    """
    Article-grouped batch sampler: groups samples from the same article into the same batch,
    so article-aware loss functions can effectively compute intra-article statistics.

    Each epoch:
    1. Shuffle article order
    2. Shuffle sample order within each article
    3. Fill batches sequentially following the shuffled article order
    4. Shuffle batch order (preserving intra-article grouping within each batch)
    """

    def __init__(self, article_ids_for_indices: List[int], batch_size: int,
                 drop_last: bool = False):
        """
        Args:
            article_ids_for_indices: article ID list of the same length as the dataset
                                     (corresponding to internal indices 0..len-1 of the dataset)
            batch_size: batch size
            drop_last: whether to drop the last incomplete batch
        """
        self.batch_size = batch_size
        self.drop_last = drop_last

        # Group dataset indices by article ID (note: these are internal dataset indices 0..N-1)
        self.article_groups = {}
        for ds_idx, aid in enumerate(article_ids_for_indices):
            self.article_groups.setdefault(aid, []).append(ds_idx)
        self.n_samples = len(article_ids_for_indices)

    def __iter__(self):
        # Shuffle article order
        articles = list(self.article_groups.keys())
        random.shuffle(articles)

        # Arrange all samples in article order (shuffle within each article as well)
        all_indices = []
        for aid in articles:
            indices = self.article_groups[aid].copy()
            random.shuffle(indices)
            all_indices.extend(indices)

        # Create batches
        batches = []
        for i in range(0, len(all_indices), self.batch_size):
            batch = all_indices[i:i + self.batch_size]
            if self.drop_last and len(batch) < self.batch_size:
                continue
            batches.append(batch)

        # Shuffle batch order
        random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        if self.drop_last:
            return self.n_samples // self.batch_size
        return (self.n_samples + self.batch_size - 1) // self.batch_size


def _is_article_aware(cfg: Dict[str, Any]) -> bool:
    """Check whether the config enables article-aware losses"""
    return (cfg.get("ARTICLE_CONSISTENCY_WEIGHT", 0) > 0 or
            cfg.get("ARTICLE_BIAS_WEIGHT", 0) > 0 or
            cfg.get("ARTICLE_RANKING_WEIGHT", 0) > 0)


# ======================== Embedding Cache (memmap format) ========================

class MemmapEmbeddingCache:
    """
    LLM embedding cache based on numpy memmap.
    Concatenates variable-length hidden_states sequences into a single .npy file accessed via memory mapping,
    significantly reducing peak memory (OS loads pages on demand, no need to load everything into RAM).
    """

    def __init__(self, npy_path: str, meta_path: str):
        self.hidden = np.load(npy_path, mmap_mode='r')  # (total_tokens, hidden_dim)
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        self.offsets = meta['offsets']       # List[(start, end)]
        self.tokens_list = meta.get('tokens', None)  # List[List[str]] or None

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        start, end = self.offsets[idx]
        hs = torch.from_numpy(self.hidden[start:end].copy()).to(torch.float16)
        seq_len = end - start
        mask = torch.ones(seq_len, dtype=torch.long)
        result = {
            "hidden_states": hs,
            "attention_mask": mask,
        }
        if self.tokens_list is not None and idx < len(self.tokens_list):
            result["tokens"] = self.tokens_list[idx]
        return result


class MemmapStructCache:
    """
    Structure encoder embedding cache based on numpy memmap.
    """

    def __init__(self, npy_path: str, meta_path: str):
        self.hidden = np.load(npy_path, mmap_mode='r')  # (total_smiles, hidden_dim)
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        self.offsets = meta['offsets']  # List[(start, end)]

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        start, end = self.offsets[idx]
        return torch.from_numpy(self.hidden[start:end].copy()).to(torch.float16)


def _save_embeddings_memmap(all_embeddings: List[Dict], npy_path: str, meta_path: str):
    """Save LLM embeddings list as numpy memmap + metadata format"""
    hidden_dim = all_embeddings[0]["hidden_states"].shape[-1]
    offsets = []
    total_tokens = 0
    for emb in all_embeddings:
        seq_len = emb["hidden_states"].shape[0]
        offsets.append((total_tokens, total_tokens + seq_len))
        total_tokens += seq_len

    hidden_array = np.empty((total_tokens, hidden_dim), dtype=np.float16)
    for emb, (start, end) in zip(all_embeddings, offsets):
        hs = emb["hidden_states"]
        if isinstance(hs, torch.Tensor):
            hs = hs.cpu().numpy()
        hidden_array[start:end] = hs.astype(np.float16)

    ensure_dir(os.path.dirname(npy_path))
    np.save(npy_path, hidden_array)

    tokens_list = [emb.get("tokens") for emb in all_embeddings]
    meta = {"offsets": offsets, "tokens": tokens_list}
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)


def _save_struct_embeddings_memmap(all_embeddings: List[torch.Tensor], npy_path: str, meta_path: str):
    """Save structure encoder embeddings list as numpy memmap + metadata format"""
    hidden_dim = all_embeddings[0].shape[-1]
    offsets = []
    total_smiles = 0
    for emb in all_embeddings:
        n = emb.shape[0]
        offsets.append((total_smiles, total_smiles + n))
        total_smiles += n

    hidden_array = np.empty((total_smiles, hidden_dim), dtype=np.float16)
    for emb, (start, end) in zip(all_embeddings, offsets):
        if isinstance(emb, torch.Tensor):
            emb = emb.cpu().numpy()
        hidden_array[start:end] = emb.astype(np.float16)

    ensure_dir(os.path.dirname(npy_path))
    np.save(npy_path, hidden_array)

    meta = {"offsets": offsets}
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)


# ======================== Data Loading ========================

def load_dataset(dataset_name: str) -> List[Dict[str, Any]]:
    path = DATASET_PATHS[dataset_name]
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_split_indices(dataset_name: str, pkl_filename: str) -> Dict[str, List[int]]:
    path = get_split_pkl_path(dataset_name, pkl_filename)
    with open(path, "rb") as f:
        return pickle.load(f)


def extract_targets(data: List[Dict], dataset_name: str) -> np.ndarray:
    """Extract the value (first element) of the corresponding property in properties"""
    values = []
    for entry in data:
        val = entry["properties"][dataset_name]["value"][0]
        values.append(float(val))
    return np.array(values, dtype=np.float64)


# ======================== ZScaler ========================

def fit_scaler(targets: np.ndarray, train_indices: List[int], *,
               cfg=None, dataset_name=""):
    return fit_target_scaler(targets, train_indices, cfg=cfg, property_name=dataset_name)


def transform_targets(scaler, targets: np.ndarray) -> np.ndarray:
    return scaler.transform(targets.reshape(-1, 1)).flatten()


def inverse_transform(scaler, values: np.ndarray) -> np.ndarray:
    return scaler.inverse_transform(values.reshape(-1, 1)).flatten()


# ======================== Embedding Cache ========================

def _llm_input_tag(input_type: Union[int, str]) -> str:
    return STRUCT_INPUT if input_type == STRUCT_INPUT else f"type{input_type}"


def get_cache_npy_filename(dataset_name: str, model_key: str, input_type: Union[int, str]) -> str:
    key = llm_cache_key(model_key)
    return f"{dataset_name}_{key}_{_llm_input_tag(input_type)}_hidden.npy"


def get_cache_meta_filename(dataset_name: str, model_key: str, input_type: Union[int, str]) -> str:
    key = llm_cache_key(model_key)
    return f"{dataset_name}_{key}_{_llm_input_tag(input_type)}_meta.pkl"


def get_struct_cache_npy_filename(dataset_name: str, encoder_name: str = "polybert") -> str:
    return f"{dataset_name}_{encoder_name}_struct_hidden.npy"


def get_struct_cache_meta_filename(dataset_name: str, encoder_name: str = "polybert") -> str:
    return f"{dataset_name}_{encoder_name}_struct_meta.pkl"


def normalize_input_types(input_types=None) -> List[Union[int, str]]:
    """Validate concrete prompt types, preserving their requested order."""
    if input_types is None:
        input_types = range(1, 10)
    result = []
    for value in input_types:
        if value == STRUCT_INPUT:
            if value not in result:
                result.append(value)
            continue
        if isinstance(value, str) and value.lower().startswith("type"):
            value = value[4:]
        if isinstance(value, str) and value.isdigit():
            value = int(value)
        if isinstance(value, bool) or not isinstance(value, int) or value not in range(1, 10):
            raise ValueError(f"Input must be 1-9, 'type1'-'type9', or 'struct', got {value!r}")
        if value not in result:
            result.append(value)
    if not result:
        raise ValueError("At least one input type is required.")
    return result


def required_llm_input_types(cfg: Dict[str, Any]) -> List[Union[int, str]]:
    """Standalone LLM uses struct; semantic/hybrid paths retain their types."""
    if not cfg.get("FROZEN_BACKBONE", True):
        return []
    if "STRUCT_ENCODER" in cfg and "ENABLE_HYBRID_ENCODING" not in cfg:
        return [STRUCT_INPUT] if cfg["STRUCT_ENCODER"] == "llm" else []
    types = list(cfg.get("INPUT_CONTENT", [1])) + list(cfg.get("EVAL_INPUT_CONTENT", [1]))
    if cfg.get("ENABLE_HYBRID_ENCODING", False) and cfg.get("STRUCT_ENCODER") == "llm":
        types.append(9)
    return normalize_input_types(types)


def cache_exists(dataset_name: str, model_key: str, input_types=None) -> bool:
    """Check only requested types; both the array and metadata must exist."""
    for t in normalize_input_types(input_types):
        npy_path = os.path.join(EMBEDDING_CACHE_DIR, get_cache_npy_filename(dataset_name, model_key, t))
        meta_path = os.path.join(EMBEDDING_CACHE_DIR, get_cache_meta_filename(dataset_name, model_key, t))
        if not (os.path.isfile(npy_path) and os.path.isfile(meta_path)):
            return False
    return True


def struct_cache_exists(dataset_name: str, encoder_name: str = "polybert") -> bool:
    npy_path = os.path.join(EMBEDDING_CACHE_DIR, get_struct_cache_npy_filename(dataset_name, encoder_name))
    return os.path.exists(npy_path)


def get_periogt_cache_filename(dataset_name: str) -> str:
    return f"{dataset_name}_periogt_data.pth"


def periogt_cache_exists(dataset_name: str) -> bool:
    path = os.path.join(EMBEDDING_CACHE_DIR, get_periogt_cache_filename(dataset_name))
    return os.path.exists(path)


def save_periogt_cache(dataset_name: str, periogt_data: List[Dict]):
    """Save precomputed PerioGT data as .pth cache file"""
    ensure_dir(EMBEDDING_CACHE_DIR)
    cache_path = os.path.join(EMBEDDING_CACHE_DIR, get_periogt_cache_filename(dataset_name))
    torch.save(periogt_data, cache_path)
    print(f"  Saved PerioGT cache: {cache_path} ({len(periogt_data)} entries)")


def load_periogt_cache(dataset_name: str) -> List[Dict]:
    """Load precomputed PerioGT data from .pth cache file"""
    cache_path = os.path.join(EMBEDDING_CACHE_DIR, get_periogt_cache_filename(dataset_name))
    periogt_data = torch.load(cache_path, map_location='cpu', weights_only=False)
    print(f"  Loaded PerioGT cache: {cache_path} ({len(periogt_data)} entries)")
    return periogt_data


def load_embedding_llm(model_key: str):
    """Load the raw-text wrapper used by training, with shared attention settings."""
    from models.base_model import LLM_REGISTRY
    llm = LLM_REGISTRY[model_key](MODEL_PATHS[model_key], device="cuda")
    print(f"[Input] {model_key}: raw text")
    llm.model.eval()
    return llm


def _write_llm_embedding_type(llm, data, dataset_name, input_type,
                              batch_size, npy_path, meta_path):
    """Two tokenizer passes, one model pass; keep at most one hidden batch in RAM."""
    offsets = []
    total_tokens = 0
    for i in range(0, len(data), batch_size):
        texts = [build_prompt(e, dataset_name, input_type) for e in data[i:i + batch_size]]
        inputs = llm.tokenize(texts)
        for length in inputs["attention_mask"].sum(dim=1).cpu().tolist():
            length = int(length)
            offsets.append((total_tokens, total_tokens + length))
            total_tokens += length
        del inputs

    temporary_paths = []
    hidden_array = None
    try:
        for destination in (npy_path, meta_path):
            fd, path = tempfile.mkstemp(prefix=os.path.basename(destination) + ".",
                                        suffix=".partial", dir=os.path.dirname(destination))
            os.close(fd)
            temporary_paths.append(path)
        hidden_array = np.lib.format.open_memmap(
            temporary_paths[0], mode="w+", dtype=np.float16,
            shape=(total_tokens, llm.hidden_dim),
        )
        tokens_list = []
        for i in range(0, len(data), batch_size):
            texts = [build_prompt(e, dataset_name, input_type) for e in data[i:i + batch_size]]
            inputs = llm.tokenize(texts)
            with torch.no_grad():
                hidden = llm.get_hidden_states(inputs)
            for j in range(hidden.size(0)):
                valid = inputs["attention_mask"][j].bool()
                start, end = offsets[i + j]
                if int(valid.sum()) != end - start:
                    raise ValueError("Token lengths changed between cache sizing and inference.")
                hidden_array[start:end] = hidden[j, valid].detach().cpu().half().numpy()
                token_ids = inputs["input_ids"][j, valid].cpu().tolist()
                tokens_list.append(llm.tokenizer.convert_ids_to_tokens(token_ids))
            hidden_array.flush()
            del hidden, inputs, valid
        del hidden_array
        hidden_array = None
        with open(temporary_paths[1], "wb") as f:
            pickle.dump({"offsets": offsets, "tokens": tokens_list}, f)
        os.replace(temporary_paths[0], npy_path)
        # Metadata is published last; an interrupted first write is not a cache hit.
        os.replace(temporary_paths[1], meta_path)
    finally:
        if hidden_array is not None:
            del hidden_array
        for path in temporary_paths:
            if os.path.exists(path):
                os.remove(path)


def precompute_llm_embeddings(dataset_name: str, model_key: str,
                              data: List[Dict], batch_size: int = 1,
                              input_types=None, llm=None):
    """Cache selected types in the existing .npy/.pkl format, one type at a time.

    Omitted input_types retains the legacy 1-9 API. A supplied wrapper is borrowed
    so a batch caller can reuse one model across datasets; otherwise load lazily.
    """
    types = normalize_input_types(input_types)
    if llm is not None and llm.MODEL_KEY != model_key:
        raise ValueError("Borrowed LLM model does not match the requested cache.")
    pending = [t for t in types if not cache_exists(dataset_name, model_key, [t])]
    if not pending:
        print(f"  Requested LLM caches already exist for {dataset_name}/{model_key}: {types}")
        return
    if not data or batch_size < 1:
        raise ValueError("Embedding requires nonempty data and batch_size >= 1.")
    ensure_dir(EMBEDDING_CACHE_DIR)
    owns_model = llm is None
    if owns_model:
        llm = load_embedding_llm(model_key)
    try:
        for input_type in pending:
            npy_path = os.path.join(EMBEDDING_CACHE_DIR,
                                   get_cache_npy_filename(dataset_name, model_key, input_type))
            meta_path = os.path.join(EMBEDDING_CACHE_DIR,
                                    get_cache_meta_filename(dataset_name, model_key, input_type))
            print(f"  Computing {dataset_name}/{model_key}/{_llm_input_tag(input_type)}...")
            _write_llm_embedding_type(llm, data, dataset_name, input_type,
                                      batch_size, npy_path, meta_path)
            gc.collect()
            torch.cuda.empty_cache()
            print(f"  Saved: {npy_path} ({len(data)} entries)")
    finally:
        if owns_model:
            del llm
            gc.collect()
            torch.cuda.empty_cache()


def precompute_struct_embeddings(dataset_name: str, data: List[Dict],
                                 encoder_name: str = "polybert",
                                 ratio_encoding_path: Optional[str] = None):
    """Precompute structure encoder embeddings and cache as numpy memmap format"""
    from models.base_model import STRUCT_ENCODER_REGISTRY

    ensure_dir(EMBEDDING_CACHE_DIR)
    npy_path = os.path.join(
        EMBEDDING_CACHE_DIR, get_struct_cache_npy_filename(dataset_name, encoder_name)
    )
    if os.path.exists(npy_path):
        print(f"  Struct cache already exists: {npy_path}, skipping.")
        return

    print(f"  Computing {encoder_name} struct embeddings...")
    encoder_cls = STRUCT_ENCODER_REGISTRY[encoder_name]["encoder_cls"]
    encoder = encoder_cls(MODEL_PATHS[encoder_name], device="cuda")

    # Load ratio encoding info
    ratio_info = None
    if ratio_encoding_path is not None and os.path.exists(ratio_encoding_path):
        ratio_info = load_ratio_encoding(ratio_encoding_path)

    all_embeddings = []
    for entry in data:
        emb = encoder.encode_entry(entry, ratio_info=ratio_info)  # (n_smiles, hidden_dim [+ ratio_dim])
        all_embeddings.append(emb.cpu().half())

    meta_path = os.path.join(
        EMBEDDING_CACHE_DIR, get_struct_cache_meta_filename(dataset_name, encoder_name)
    )
    _save_struct_embeddings_memmap(all_embeddings, npy_path, meta_path)
    print(f"  Saved: {npy_path} ({len(all_embeddings)} entries)")

    del encoder
    torch.cuda.empty_cache()


def load_cached_embeddings(dataset_name: str, model_key: str,
                           input_type: Union[int, str]):
    """Load cached embeddings (memmap format)"""
    npy_path = os.path.join(
        EMBEDDING_CACHE_DIR,
        get_cache_npy_filename(dataset_name, model_key, input_type)
    )
    meta_path = os.path.join(
        EMBEDDING_CACHE_DIR,
        get_cache_meta_filename(dataset_name, model_key, input_type)
    )
    return MemmapEmbeddingCache(npy_path, meta_path)


def load_cached_struct_embeddings(dataset_name: str, encoder_name: str = "polybert"):
    """Load cached structure encoder embeddings (memmap format)"""
    npy_path = os.path.join(
        EMBEDDING_CACHE_DIR, get_struct_cache_npy_filename(dataset_name, encoder_name)
    )
    meta_path = os.path.join(
        EMBEDDING_CACHE_DIR, get_struct_cache_meta_filename(dataset_name, encoder_name)
    )
    return MemmapStructCache(npy_path, meta_path)


# ======================== Dataset ========================

class CachedEmbeddingDataset(Dataset):
    """Dataset used with frozen backbone — loads embeddings from cache"""

    def __init__(self, indices: List[int], all_cached: Dict[int, List[Dict]],
                 targets: np.ndarray, input_content: List[int],
                 struct_embeddings: Optional[List[torch.Tensor]] = None,
                 article_ids: Optional[List[int]] = None):
        self.indices = indices
        self.all_cached = all_cached  # {input_type: [list of dicts]}
        self.targets = targets
        self.input_content = input_content
        self.struct_embeddings = struct_embeddings
        self.article_ids = article_ids

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        # Randomly sample an input type from the input_content list (single-element list is equivalent to fixed)
        input_type = random.choice(self.input_content)
        cached = self.all_cached[input_type][real_idx]

        result = {
            "text_hidden": cached["hidden_states"].float(),
            "text_mask": cached["attention_mask"],
            "target": torch.tensor(self.targets[real_idx], dtype=torch.float32),
        }

        if "tokens" in cached:
            result["tokens"] = cached["tokens"]

        if self.struct_embeddings is not None:
            result["struct_emb"] = self.struct_embeddings[real_idx].float()

        if self.article_ids is not None:
            result["article_id"] = self.article_ids[real_idx]

        return result


class LiveInferenceDataset(Dataset):
    """Dataset used with non-frozen backbone — returns text, inference occurs during model forward"""

    def __init__(self, indices: List[int], data: List[Dict],
                 targets: np.ndarray, dataset_name: str,
                 input_content: List[int],
                 struct_embeddings: Optional[List[torch.Tensor]] = None,
                 need_smiles: bool = False,
                 ratio_encoding: Optional[Dict] = None,
                 need_struct_text: bool = False,
                 article_ids: Optional[List[int]] = None):
        self.indices = indices
        self.data = data
        self.targets = targets
        self.dataset_name = dataset_name
        self.input_content = input_content
        self.struct_embeddings = struct_embeddings
        self.need_smiles = need_smiles
        self.ratio_encoding = ratio_encoding
        self.need_struct_text = need_struct_text  # Hybrid + non-frozen LLM struct stream: provide type 9 text
        self.article_ids = article_ids

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        # Randomly sample an input type from the input_content list (single-element list is equivalent to fixed)
        input_type = random.choice(self.input_content)
        text = build_prompt(self.data[real_idx], self.dataset_name, input_type)

        result = {
            "text": text,
            "target": torch.tensor(self.targets[real_idx], dtype=torch.float32),
        }

        if self.struct_embeddings is not None:
            result["struct_emb"] = self.struct_embeddings[real_idx].float()
        elif self.need_struct_text:
            # Hybrid + non-frozen LLM struct stream: provide type 9 text for struct LLM real-time inference
            struct_text = build_prompt(self.data[real_idx], self.dataset_name, 9)
            result["struct_texts"] = struct_text
        elif self.need_smiles:
            # Hybrid + non-frozen mode: provide SMILES list and ratio info for real-time encoding
            entry = self.data[real_idx]
            chem = entry.get("chemical_composition", {})
            monomers = chem.get("monomers", [])
            is_homo = chem.get("is_homopolymer", False)
            smiles_list = []
            ratio_vectors = []
            for m in monomers:
                smi = m.get("smiles")
                if smi and smi.strip():
                    smiles_list.append(smi.strip())
                    if self.ratio_encoding is not None:
                        rv = build_ratio_vector(m, is_homo, self.ratio_encoding)
                        ratio_vectors.append(torch.tensor(rv, dtype=torch.float32))
            result["smiles_list"] = smiles_list if smiles_list else []
            if ratio_vectors:
                result["ratio_vectors"] = ratio_vectors

        if self.article_ids is not None:
            result["article_id"] = self.article_ids[real_idx]

        return result


def cached_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for CachedEmbeddingDataset, handling variable-length sequences"""
    max_seq_len = max(item["text_hidden"].size(0) for item in batch)
    hidden_dim = batch[0]["text_hidden"].size(-1)
    bsz = len(batch)

    text_hidden = torch.zeros(bsz, max_seq_len, hidden_dim)
    text_mask = torch.zeros(bsz, max_seq_len, dtype=torch.long)
    targets = torch.stack([item["target"] for item in batch])

    for i, item in enumerate(batch):
        seq_len = item["text_hidden"].size(0)
        text_hidden[i, :seq_len] = item["text_hidden"]
        text_mask[i, :seq_len] = item["text_mask"]

    result = {
        "text_hidden": text_hidden,
        "text_mask": text_mask,
        "target": targets,
    }

    # Structure embeddings
    if "struct_emb" in batch[0]:
        max_smiles = max(item["struct_emb"].size(0) for item in batch)
        struct_dim = batch[0]["struct_emb"].size(-1)
        struct_emb = torch.zeros(bsz, max_smiles, struct_dim)
        struct_mask = torch.zeros(bsz, max_smiles, dtype=torch.long)
        for i, item in enumerate(batch):
            n = item["struct_emb"].size(0)
            struct_emb[i, :n] = item["struct_emb"]
            struct_mask[i, :n] = 1
        result["struct_emb"] = struct_emb
        result["struct_mask"] = struct_mask

    # Token strings (for pooling weight visualization)
    if "tokens" in batch[0]:
        result["tokens"] = [item["tokens"] for item in batch]

    # Article ID
    if "article_id" in batch[0]:
        result["article_ids"] = torch.tensor([item["article_id"] for item in batch], dtype=torch.long)

    return result


def live_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function for LiveInferenceDataset"""
    result = {
        "text": [item["text"] for item in batch],
        "target": torch.stack([item["target"] for item in batch]),
    }

    if "struct_emb" in batch[0]:
        max_smiles = max(item["struct_emb"].size(0) for item in batch)
        struct_dim = batch[0]["struct_emb"].size(-1)
        bsz = len(batch)
        struct_emb = torch.zeros(bsz, max_smiles, struct_dim)
        struct_mask = torch.zeros(bsz, max_smiles, dtype=torch.long)
        for i, item in enumerate(batch):
            n = item["struct_emb"].size(0)
            struct_emb[i, :n] = item["struct_emb"]
            struct_mask[i, :n] = 1
        result["struct_emb"] = struct_emb
        result["struct_mask"] = struct_mask

    if "smiles_list" in batch[0]:
        result["smiles_lists"] = [item["smiles_list"] for item in batch]
        if "ratio_vectors" in batch[0]:
            result["ratio_vectors_lists"] = [item.get("ratio_vectors", []) for item in batch]

    if "struct_texts" in batch[0]:
        result["struct_texts"] = [item["struct_texts"] for item in batch]

    if "article_id" in batch[0]:
        result["article_ids"] = torch.tensor([item["article_id"] for item in batch], dtype=torch.long)

    return result

def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    r2 = r2_score(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    mae = mean_absolute_error(targets, preds)
    return {"R2": r2, "RMSE": rmse, "MAE": mae}


# ======================== Visualization ========================

def plot_parity(preds: np.ndarray, targets: np.ndarray, rmse: float,
                title: str, save_path: str):
    """Generate true vs predicted scatter plot"""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(targets, preds, alpha=0.6, s=20)

    all_vals = np.concatenate([targets, preds])
    vmin, vmax = all_vals.min(), all_vals.max()
    margin = (vmax - vmin) * 0.05
    ax.plot([vmin - margin, vmax + margin], [vmin - margin, vmax + margin],
            "r--", linewidth=1)

    ax.set_xlabel("True Value")
    ax.set_ylabel("Predicted Value")
    ax.set_title(title)
    ax.text(0.05, 0.95, f"RMSE = {rmse:.4f}", transform=ax.transAxes,
            verticalalignment="top", fontsize=12)
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ======================== Config Saving ========================

def _make_yaml_serializable(obj):
    """Recursively convert non-YAML-serializable types"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _make_yaml_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_yaml_serializable(v) for v in obj]
    # Ray Tune Domain objects (grid_search / choice etc.)
    try:
        from ray.tune.search.sample import Domain
        if isinstance(obj, Domain):
            return repr(obj)
    except ImportError:
        pass
    # Other non-serializable types fall back to string
    try:
        yaml.dump(obj)
    except (yaml.YAMLError, TypeError):
        return str(obj)
    return obj


def save_config_yaml(cfg: Dict[str, Any], save_path: str):
    """Save the config dictionary as YAML"""
    ensure_dir(os.path.dirname(save_path))
    # Drop the obsolete switch from imported historical configs. This is fixed
    # provenance metadata, not an input-format option.
    current_cfg = {k: v for k, v in cfg.items() if k != "USE_CHAT_TEMPLATE"}
    if "MODEL_TYPE" in current_cfg:
        current_cfg["_LLM_INPUT_FORMAT"] = "raw_text"
    serializable = _make_yaml_serializable(current_cfg)
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(serializable, f, default_flow_style=False, allow_unicode=True)


def save_csv_summary(results: List[Dict], save_path: str):
    """Save CSV summary table"""
    ensure_dir(os.path.dirname(save_path))
    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")


# ======================== Test Set Evaluation ========================

def evaluate_test_set(best_model, test_loader, cfg, scaler,
                      struct_only: bool, frozen: bool,
                      experiment_dir: str, seed: int, dataset_name: str,
                      label_suffix: str = "", file_suffix: str = "",
                      save_visualizations: bool = True,
                      train_article_ranges: Optional[Dict[int, tuple]] = None):
    """
    Run inference on the specified dataset, compute metrics, and optionally collect
    visualization data and generate parity plots.

    Args:
        best_model: best model with loaded weights (eval mode, already on correct device)
        test_loader: DataLoader for evaluation (test or validation set)
        cfg: configuration dictionary
        scaler: StandardScaler for inverse normalization
        struct_only: whether this is structure-only control mode
        frozen: whether backbone is frozen
        experiment_dir: experiment output directory
        seed: current random seed
        dataset_name: dataset name (for labels and printing)
        label_suffix: parity plot title suffix (e.g. " (Transfer)")
        file_suffix: output filename suffix (e.g. "_Tg")
        save_visualizations: whether to save parity plot and visualization JSON (set to False for grid search mode)
        train_article_ranges: training set target value range per article {article_id: (min, max)}, used for Article Range Hit Rate

    Returns:
        (metrics, preds_orig, targets_orig)
    """
    device = best_model.device
    preds_list = []
    targets_list = []
    article_ids_list = []

    # Visualization flags (collected only when save_visualizations=True)
    need_pooling_viz = (save_visualizations and not struct_only and
                        cfg.get("POOLING_TYPE") in ("attention_pooling", "sigmoid_pooling"))
    need_gate_viz = (save_visualizations and not struct_only and cfg.get("ENABLE_HYBRID_ENCODING", False))
    pooling_records = []
    gate_records = []
    sample_idx = 0

    with torch.no_grad():
        for batch in test_loader:
            # Build PerioGT graph data (if needed)
            graph_data = None
            _has_periogt = (
                (struct_only and cfg.get("STRUCT_ENCODER") == "periogt") or
                (not struct_only and getattr(best_model, 'periogt', None) is not None)
            )
            if _has_periogt and 'graphs' in batch:
                graph_data = {k: batch[k].to(device) for k in
                              ['graphs', 'fp_1', 'md_1', 'fp_2', 'md_2',
                               'ratio_vec_1', 'ratio_vec_2', 'global_type']}

            batch_tokens = batch.get("tokens")  # List[List[str]] or None

            if struct_only:
                enc_name = cfg.get("STRUCT_ENCODER", "polybert")
                if enc_name == "periogt":
                    p = best_model(graph_data=graph_data)
                elif enc_name == "llm":
                    if best_model.frozen:
                        text_mask = batch.get("text_mask")
                        p = best_model(text_hidden=batch["text_hidden"].to(device),
                                       text_mask=text_mask.to(device) if text_mask is not None else None)
                    else:
                        inputs = best_model.llm.tokenize(batch["text"])
                        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                  for k, v in inputs.items()}
                        text_hidden = best_model.llm.get_hidden_states(inputs).float()
                        text_mask = inputs.get("attention_mask")
                        p = best_model(text_hidden=text_hidden, text_mask=text_mask)
                else:
                    if best_model.frozen:
                        struct_mask = batch.get("struct_mask")
                        p = best_model(struct_emb=batch["struct_emb"].to(device),
                                       struct_mask=struct_mask.to(device) if struct_mask is not None else None)
                    else:
                        struct_emb, struct_mask = best_model._encode_smiles_batch(
                            batch["smiles_lists"],
                            ratio_vectors_lists=batch.get("ratio_vectors_lists"))
                        p = best_model(struct_emb=struct_emb, struct_mask=struct_mask)
                aux = {}
            elif frozen:
                text_hidden = batch["text_hidden"].to(device)
                text_mask = batch["text_mask"].to(device)
                struct_emb = batch.get("struct_emb")
                struct_mask = batch.get("struct_mask")
                if struct_emb is not None:
                    struct_emb = struct_emb.to(device)
                    struct_mask = struct_mask.to(device)
                if need_pooling_viz or need_gate_viz:
                    p, aux = best_model(text_hidden, text_mask, struct_emb, struct_mask,
                                        graph_data=graph_data, return_aux=True)
                else:
                    p = best_model(text_hidden, text_mask, struct_emb, struct_mask,
                                   graph_data=graph_data)
                    aux = {}
            else:
                inputs = best_model.llm.tokenize(batch["text"])
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                          for k, v in inputs.items()}
                text_hidden = best_model.llm.get_hidden_states(inputs).float()
                text_mask = inputs.get("attention_mask")
                struct_emb = batch.get("struct_emb")
                struct_mask = batch.get("struct_mask")
                if struct_emb is not None:
                    struct_emb = struct_emb.to(device)
                    struct_mask = struct_mask.to(device)
                elif best_model.hybrid_gate is not None and not _has_periogt:
                    struct_texts = batch.get("struct_texts")
                    smiles_lists = batch.get("smiles_lists")
                    if struct_texts is not None:
                        type9_inputs = best_model.llm.tokenize(struct_texts)
                        type9_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                        for k, v in type9_inputs.items()}
                        struct_emb = best_model.llm.get_hidden_states(type9_inputs).float()
                        struct_mask = type9_inputs.get("attention_mask")
                    elif smiles_lists is not None:
                        struct_emb, struct_mask = best_model._encode_smiles_batch(
                            smiles_lists,
                            ratio_vectors_lists=batch.get("ratio_vectors_lists"))
                # In non-frozen mode, extract token strings for visualization
                if batch_tokens is None and need_pooling_viz:
                    batch_tokens = [
                        best_model.llm.tokenizer.convert_ids_to_tokens(
                            inputs["input_ids"][i].tolist()
                        )
                        for i in range(inputs["input_ids"].size(0))
                    ]
                if need_pooling_viz or need_gate_viz:
                    p, aux = best_model(text_hidden, text_mask, struct_emb, struct_mask,
                                        graph_data=graph_data, return_aux=True)
                else:
                    p = best_model(text_hidden, text_mask, struct_emb, struct_mask,
                                   graph_data=graph_data)
                    aux = {}

            preds_list.append(p.cpu().numpy())
            targets_list.append(batch["target"].numpy())
            if "article_ids" in batch:
                article_ids_list.append(batch["article_ids"].numpy())

            # Collect visualization data
            bsz = p.size(0)
            if need_pooling_viz and "pooling_weights" in aux:
                pw = aux["pooling_weights"].cpu()  # (B, seq_len)
                for i in range(bsz):
                    if text_mask is not None:
                        seq_len = int(text_mask[i].sum().item())
                    else:
                        seq_len = pw.size(1)
                    w = pw[i, :seq_len].tolist()
                    tok = batch_tokens[i][:seq_len] if batch_tokens else None
                    pooling_records.append({
                        "index": sample_idx + i,
                        "tokens": tok,
                        "weights": w,
                    })
            if need_gate_viz and "gate_beta" in aux:
                gb = aux["gate_beta"].cpu()  # (B,)
                for i in range(bsz):
                    gate_records.append({
                        "index": sample_idx + i,
                        "gate_beta": float(gb[i]),
                    })
            sample_idx += bsz

    preds_scaled = np.concatenate(preds_list)
    targets_scaled_test = np.concatenate(targets_list)

    # Inverse normalization
    preds_orig = inverse_transform(scaler, preds_scaled)
    targets_orig = inverse_transform(scaler, targets_scaled_test)

    metrics = compute_metrics(preds_orig, targets_orig)
    metrics["seed"] = seed

    print(f"  Seed {seed}: R2={metrics['R2']:.4f}, "
          f"RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}")

    # Article-aware metrics (Article Range Hit Rate, NARS)
    if article_ids_list:
        all_article_ids = np.concatenate(article_ids_list)
        parts = []
        # Article Range Hit Rate（需要训练集文章范围）
        if train_article_ranges:
            from models.base_model import _compute_article_range_hit_rate, _compute_article_nars
            hr = _compute_article_range_hit_rate(preds_orig, all_article_ids, train_article_ranges)
            if not np.isnan(hr):
                metrics["article_range_hit_rate"] = hr
                parts.append(f"RangeHR={hr:.4f}")
            nars = _compute_article_nars(preds_orig, all_article_ids, train_article_ranges)
            if not np.isnan(nars):
                metrics["article_nars"] = nars
                parts.append(f"NARS={nars:.4f}")
        if parts:
            print(f"  [Test] Article: {', '.join(parts)}")

    if save_visualizations:
        # Parity plot
        plot_parity(
            preds_orig, targets_orig, metrics["RMSE"],
            f"{dataset_name} Seed={seed}{label_suffix}",
            os.path.join(experiment_dir, f"parity{file_suffix}_seed{seed}.png"),
        )

        # Save visualization JSON
        if pooling_records:
            for rec in pooling_records:
                idx = rec["index"]
                rec["true_value"] = float(targets_orig[idx])
                rec["pred_value"] = float(preds_orig[idx])
            path = os.path.join(experiment_dir, f"pooling_weights{file_suffix}_seed{seed}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(pooling_records, f, ensure_ascii=False, indent=2)
            print(f"  Saved pooling weights -> {path}")

        if gate_records:
            for rec in gate_records:
                idx = rec["index"]
                rec["true_value"] = float(targets_orig[idx])
                rec["pred_value"] = float(preds_orig[idx])
            path = os.path.join(experiment_dir, f"gate_weights{file_suffix}_seed{seed}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(gate_records, f, ensure_ascii=False, indent=2)
            print(f"  Saved gate weights -> {path}")

    return metrics, preds_orig, targets_orig


# ======================== Seed Setting ========================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ======================== DataLoader Construction ========================

def build_dataloaders(
    data: List[Dict],
    dataset_name: str,
    split_indices: Dict[str, List[int]],
    targets_scaled: np.ndarray,
    cfg: Dict[str, Any],
    struct_embeddings: Optional[List[torch.Tensor]] = None,
    periogt_data: Optional[List[Dict]] = None,
    ratio_encoding_path: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders"""
    frozen = cfg["FROZEN_BACKBONE"]
    input_content = cfg["INPUT_CONTENT"]
    eval_input_content = cfg.get("EVAL_INPUT_CONTENT", input_content)
    batch_size = cfg["BATCH_SIZE"]
    num_workers = cfg["NUM_WORKERS"]

    # ---------- article-aware ----------
    article_aware = _is_article_aware(cfg)
    article_ids = extract_article_ids(data) if article_aware else None
    train_article_ids = article_ids if article_aware else None
    num_workers = cfg["NUM_WORKERS"]

    enable_hybrid = cfg.get("ENABLE_HYBRID_ENCODING", False)
    encoder_name = cfg.get("STRUCT_ENCODER", "polybert") if enable_hybrid else None

    if enable_hybrid and encoder_name == "periogt" and periogt_data is not None:
        # Hybrid stream + PerioGT: PerioGT always trains with full parameters, using precomputed graph data
        if frozen:
            model_key = cfg["MODEL_TYPE"]
            all_cached = {}
            all_types_needed = set(input_content) | set(eval_input_content)
            for t in all_types_needed:
                all_cached[t] = load_cached_embeddings(dataset_name, model_key, t)

            train_ds = HybridPerioGTCachedDataset(
                split_indices["train"], all_cached, targets_scaled,
                input_content, periogt_data, article_ids=train_article_ids)
            val_ds = HybridPerioGTCachedDataset(
                split_indices["val"], all_cached, targets_scaled,
                eval_input_content, periogt_data, article_ids=train_article_ids)
            test_ds = HybridPerioGTCachedDataset(
                split_indices["test"], all_cached, targets_scaled,
                eval_input_content, periogt_data, article_ids=train_article_ids)
            collate = hybrid_periogt_cached_collate_fn
        else:
            train_ds = HybridPerioGTLiveDataset(
                split_indices["train"], data, targets_scaled,
                dataset_name, input_content, periogt_data, article_ids=train_article_ids)
            val_ds = HybridPerioGTLiveDataset(
                split_indices["val"], data, targets_scaled,
                dataset_name, eval_input_content, periogt_data, article_ids=train_article_ids)
            test_ds = HybridPerioGTLiveDataset(
                split_indices["test"], data, targets_scaled,
                dataset_name, eval_input_content, periogt_data, article_ids=train_article_ids)
            collate = hybrid_periogt_live_collate_fn
    elif frozen:
        model_key = cfg["MODEL_TYPE"]
        # Load input types needed for the training set and validation/test set
        all_cached = {}
        all_types_needed = set(input_content) | set(eval_input_content)
        for t in all_types_needed:
            all_cached[t] = load_cached_embeddings(dataset_name, model_key, t)

        if enable_hybrid and encoder_name == "llm":
            # Hybrid stream LLM struct flow: type 9 hidden states as struct encoder input
            cached_struct = load_cached_embeddings(dataset_name, model_key, 9)
            train_ds = HybridLLMStructCachedDataset(
                split_indices["train"], all_cached, targets_scaled,
                input_content, cached_struct, article_ids=train_article_ids)
            val_ds = HybridLLMStructCachedDataset(
                split_indices["val"], all_cached, targets_scaled,
                eval_input_content, cached_struct, article_ids=train_article_ids)
            test_ds = HybridLLMStructCachedDataset(
                split_indices["test"], all_cached, targets_scaled,
                eval_input_content, cached_struct, article_ids=train_article_ids)
            collate = hybrid_llm_struct_cached_collate_fn
        else:
            train_ds = CachedEmbeddingDataset(
                split_indices["train"], all_cached, targets_scaled,
                input_content, struct_embeddings, article_ids=train_article_ids,
            )
            val_ds = CachedEmbeddingDataset(
                split_indices["val"], all_cached, targets_scaled,
                eval_input_content, struct_embeddings, article_ids=train_article_ids,
            )
            test_ds = CachedEmbeddingDataset(
                split_indices["test"], all_cached, targets_scaled,
                eval_input_content, struct_embeddings, article_ids=train_article_ids,
            )
            collate = cached_collate_fn
    else:
        # Hybrid stream + non-frozen mode needs SMILES lists or type9 text for real-time encoding
        _enc = cfg.get("STRUCT_ENCODER", "polybert") if encoder_name else None
        need_struct_text = (cfg.get("ENABLE_HYBRID_ENCODING", False)
                            and _enc == "llm"
                            and struct_embeddings is None)
        need_smiles = (cfg.get("ENABLE_HYBRID_ENCODING", False)
                       and struct_embeddings is None
                       and _enc not in ("llm", "periogt", None))
        ratio_enc = None
        if need_smiles and ratio_encoding_path is not None and os.path.exists(ratio_encoding_path):
            ratio_enc = load_ratio_encoding(ratio_encoding_path)
        train_ds = LiveInferenceDataset(
            split_indices["train"], data, targets_scaled,
            dataset_name, input_content, struct_embeddings,
            need_smiles=need_smiles, ratio_encoding=ratio_enc,
            need_struct_text=need_struct_text, article_ids=train_article_ids,
        )
        val_ds = LiveInferenceDataset(
            split_indices["val"], data, targets_scaled,
            dataset_name, eval_input_content, struct_embeddings,
            need_smiles=need_smiles, ratio_encoding=ratio_enc,
            need_struct_text=need_struct_text, article_ids=train_article_ids,
        )
        test_ds = LiveInferenceDataset(
            split_indices["test"], data, targets_scaled,
            dataset_name, eval_input_content, struct_embeddings,
            need_smiles=need_smiles, ratio_encoding=ratio_enc,
            need_struct_text=need_struct_text, article_ids=train_article_ids,
        )
        collate = live_collate_fn

    if article_aware:
        # Build article ID list corresponding to training set indices
        # (local indices 0..N-1 within the dataset)
        train_article_ids_local = [article_ids[idx] for idx in split_indices["train"]]
        train_sampler = ArticleGroupedBatchSampler(
            train_article_ids_local, batch_size)
        train_loader = DataLoader(
            train_ds, batch_sampler=train_sampler,
            num_workers=num_workers, collate_fn=collate, pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, collate_fn=collate, pin_memory=True,
        )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate, pin_memory=True,
    )
    return train_loader, val_loader, test_loader


# ======================== Structure-Only Dataset / DataLoader ========================

class StructOnlyDataset(Dataset):
    """Dataset for structure-only control mode — frozen mode, loads cached embeddings"""

    def __init__(self, indices: List[int],
                 struct_embeddings: List[torch.Tensor],
                 targets: np.ndarray,
                 article_ids: Optional[List[int]] = None):
        self.indices = indices
        self.struct_embeddings = struct_embeddings
        self.targets = targets
        self.article_ids = article_ids

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        result = {
            "struct_emb": self.struct_embeddings[real_idx].float(),
            "target": torch.tensor(self.targets[real_idx], dtype=torch.float32),
        }
        if self.article_ids is not None:
            result["article_id"] = self.article_ids[real_idx]
        return result


class LiveStructDataset(Dataset):
    """Dataset for structure-only control mode — non-frozen mode, returns SMILES lists for real-time encoding"""

    def __init__(self, indices: List[int], data: List[Dict],
                 targets: np.ndarray,
                 ratio_encoding: Optional[Dict] = None,
                 article_ids: Optional[List[int]] = None):
        self.indices = indices
        self.data = data
        self.targets = targets
        self.ratio_encoding = ratio_encoding
        self.article_ids = article_ids

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        entry = self.data[real_idx]
        chem = entry.get("chemical_composition", {})
        monomers = chem.get("monomers", [])
        is_homo = chem.get("is_homopolymer", False)
        smiles_list = []
        ratio_vectors = []
        for m in monomers:
            smi = m.get("smiles")
            if smi and smi.strip():
                smiles_list.append(smi.strip())
                if self.ratio_encoding is not None:
                    rv = build_ratio_vector(m, is_homo, self.ratio_encoding)
                    ratio_vectors.append(torch.tensor(rv, dtype=torch.float32))

        result = {
            "smiles_list": smiles_list if smiles_list else [],
            "target": torch.tensor(self.targets[real_idx], dtype=torch.float32),
        }
        if ratio_vectors:
            result["ratio_vectors"] = ratio_vectors
        if self.article_ids is not None:
            result["article_id"] = self.article_ids[real_idx]
        return result


class LLMStructCachedDataset(Dataset):
    """Frozen standalone LLM dataset — uses the reduced structural input."""

    def __init__(self, indices: List[int], cached_struct: List[Dict],
                 targets: np.ndarray,
                 article_ids: Optional[List[int]] = None):
        self.indices = indices
        self.cached_struct = cached_struct
        self.targets = targets
        self.article_ids = article_ids

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        cached = self.cached_struct[real_idx]
        result = {
            "text_hidden": cached["hidden_states"].float(),
            "text_mask": cached["attention_mask"],
            "target": torch.tensor(self.targets[real_idx], dtype=torch.float32),
        }
        if self.article_ids is not None:
            result["article_id"] = self.article_ids[real_idx]
        return result


class LLMStructLiveDataset(Dataset):
    """Non-frozen standalone LLM dataset — returns reduced structural text."""

    def __init__(self, indices: List[int], data: List[Dict],
                 targets: np.ndarray, dataset_name: str,
                 article_ids: Optional[List[int]] = None):
        self.indices = indices
        self.data = data
        self.targets = targets
        self.dataset_name = dataset_name
        self.article_ids = article_ids

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        text = build_prompt(self.data[real_idx], self.dataset_name, STRUCT_INPUT)
        result = {
            "text": text,
            "target": torch.tensor(self.targets[real_idx], dtype=torch.float32),
        }
        if self.article_ids is not None:
            result["article_id"] = self.article_ids[real_idx]
        return result


class PerioGTDataset(Dataset):
    """PerioGT struct flow dataset — uses precomputed graphs and features"""

    def __init__(self, indices: List[int], all_periogt_data: List[Dict],
                 targets: np.ndarray,
                 article_ids: Optional[List[int]] = None):
        self.indices = indices
        self.all_data = all_periogt_data
        self.targets = targets
        self.article_ids = article_ids

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        item = self.all_data[real_idx]
        result = {
            "graph": item["graph"],
            "fp_1": item["fp_1"],
            "md_1": item["md_1"],
            "fp_2": item["fp_2"],
            "md_2": item["md_2"],
            "ratio_vec_1": item["ratio_vec_1"],
            "ratio_vec_2": item["ratio_vec_2"],
            "global_type": item["global_type"],
            "target": torch.tensor(self.targets[real_idx], dtype=torch.float32),
        }
        if self.article_ids is not None:
            result["article_id"] = self.article_ids[real_idx]
        return result


def periogt_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function for PerioGT dataset"""
    from torch_geometric.data import Batch
    from models.PerioGT.graph_builder import preprocess_batch_light
    graphs = [item["graph"] for item in batch]
    batched_graph = Batch.from_data_list(graphs)

    # Path indices need to be corrected according to the node offset of each graph in the batch
    num_nodes_per_graph = torch.diff(batched_graph.ptr).numpy()
    batch_idx_per_edge = batched_graph.batch[batched_graph.edge_index[0]]
    num_edges_per_graph = batch_idx_per_edge.bincount(
        minlength=batched_graph.num_graphs).numpy()
    batched_graph.path = preprocess_batch_light(
        num_nodes_per_graph, num_edges_per_graph, batched_graph.path
    )

    result = {
        "graphs": batched_graph,
        "fp_1": torch.stack([item["fp_1"] for item in batch]),
        "md_1": torch.stack([item["md_1"] for item in batch]),
        "fp_2": torch.stack([item["fp_2"] for item in batch]),
        "md_2": torch.stack([item["md_2"] for item in batch]),
        "ratio_vec_1": torch.stack([item["ratio_vec_1"] for item in batch]),
        "ratio_vec_2": torch.stack([item["ratio_vec_2"] for item in batch]),
        "global_type": torch.stack([item["global_type"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
    }
    if "article_id" in batch[0]:
        result["article_ids"] = torch.tensor([item["article_id"] for item in batch], dtype=torch.long)
    return result


# ======================== Hybrid Stream + PerioGT Dataset / collate ========================

class HybridPerioGTCachedDataset(Dataset):
    """Frozen LLM + PerioGT hybrid stream dataset"""

    def __init__(self, indices: List[int], all_cached: Dict[int, List[Dict]],
                 targets: np.ndarray, input_content: List[int],
                 periogt_data: List[Dict],
                 article_ids: Optional[List[int]] = None):
        self.indices = indices
        self.all_cached = all_cached
        self.targets = targets
        self.input_content = input_content
        self.periogt_data = periogt_data
        self.article_ids = article_ids

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        input_type = random.choice(self.input_content)
        cached = self.all_cached[input_type][real_idx]
        item = self.periogt_data[real_idx]

        result = {
            "text_hidden": cached["hidden_states"].float(),
            "text_mask": cached["attention_mask"],
            "graph": item["graph"],
            "fp_1": item["fp_1"],
            "md_1": item["md_1"],
            "fp_2": item["fp_2"],
            "md_2": item["md_2"],
            "ratio_vec_1": item["ratio_vec_1"],
            "ratio_vec_2": item["ratio_vec_2"],
            "global_type": item["global_type"],
            "target": torch.tensor(self.targets[real_idx], dtype=torch.float32),
        }
        if "tokens" in cached:
            result["tokens"] = cached["tokens"]
        if self.article_ids is not None:
            result["article_id"] = self.article_ids[real_idx]
        return result


class HybridPerioGTLiveDataset(Dataset):
    """Non-frozen LLM + PerioGT hybrid stream dataset"""

    def __init__(self, indices: List[int], data: List[Dict],
                 targets: np.ndarray, dataset_name: str,
                 input_content: List[int], periogt_data: List[Dict],
                 article_ids: Optional[List[int]] = None):
        self.indices = indices
        self.data = data
        self.targets = targets
        self.dataset_name = dataset_name
        self.input_content = input_content
        self.periogt_data = periogt_data
        self.article_ids = article_ids

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        input_type = random.choice(self.input_content)
        text = build_prompt(self.data[real_idx], self.dataset_name, input_type)
        item = self.periogt_data[real_idx]

        result = {
            "text": text,
            "graph": item["graph"],
            "fp_1": item["fp_1"],
            "md_1": item["md_1"],
            "fp_2": item["fp_2"],
            "md_2": item["md_2"],
            "ratio_vec_1": item["ratio_vec_1"],
            "ratio_vec_2": item["ratio_vec_2"],
            "global_type": item["global_type"],
            "target": torch.tensor(self.targets[real_idx], dtype=torch.float32),
        }
        if self.article_ids is not None:
            result["article_id"] = self.article_ids[real_idx]
        return result


class HybridLLMStructCachedDataset(Dataset):
    """Frozen LLM + LLM struct flow hybrid mode dataset: both semantic and struct streams use LLM cached embeddings.
    Semantic stream: randomly sampled from all_cached (multiple input types); struct stream: fixed type 9 cache.
    """

    def __init__(self, indices: List[int], all_cached: Dict[int, List[Dict]],
                 targets: np.ndarray, input_content: List[int],
                 cached_struct: List[Dict],
                 article_ids: Optional[List[int]] = None):
        self.indices = indices
        self.all_cached = all_cached       # {input_type: List[Dict]} — semantic stream
        self.targets = targets
        self.input_content = input_content
        self.cached_struct = cached_struct  # List[Dict] — type 9 struct stream
        self.article_ids = article_ids

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        input_type = random.choice(self.input_content)
        cached = self.all_cached[input_type][real_idx]
        struct = self.cached_struct[real_idx]
        result = {
            "text_hidden": cached["hidden_states"].float(),
            "text_mask": cached["attention_mask"],
            "struct_emb": struct["hidden_states"].float(),
            "struct_mask": struct["attention_mask"],
            "target": torch.tensor(self.targets[real_idx], dtype=torch.float32),
        }
        if "tokens" in cached:
            result["tokens"] = cached["tokens"]
        if self.article_ids is not None:
            result["article_id"] = self.article_ids[real_idx]
        return result


def hybrid_periogt_cached_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function for frozen LLM + PerioGT hybrid stream"""
    from torch_geometric.data import Batch
    from models.PerioGT.graph_builder import preprocess_batch_light

    bsz = len(batch)
    max_seq_len = max(item["text_hidden"].size(0) for item in batch)
    hidden_dim = batch[0]["text_hidden"].size(-1)

    text_hidden = torch.zeros(bsz, max_seq_len, hidden_dim)
    text_mask = torch.zeros(bsz, max_seq_len, dtype=torch.long)
    for i, item in enumerate(batch):
        seq_len = item["text_hidden"].size(0)
        text_hidden[i, :seq_len] = item["text_hidden"]
        text_mask[i, :seq_len] = item["text_mask"]

    graphs = Batch.from_data_list([item["graph"] for item in batch])
    num_nodes_per_graph = torch.diff(graphs.ptr).numpy()
    batch_idx_per_edge = graphs.batch[graphs.edge_index[0]]
    num_edges_per_graph = batch_idx_per_edge.bincount(minlength=graphs.num_graphs).numpy()
    graphs.path = preprocess_batch_light(num_nodes_per_graph, num_edges_per_graph, graphs.path)

    result = {
        "text_hidden": text_hidden,
        "text_mask": text_mask,
        "graphs": graphs,
        "fp_1": torch.stack([item["fp_1"] for item in batch]),
        "md_1": torch.stack([item["md_1"] for item in batch]),
        "fp_2": torch.stack([item["fp_2"] for item in batch]),
        "md_2": torch.stack([item["md_2"] for item in batch]),
        "ratio_vec_1": torch.stack([item["ratio_vec_1"] for item in batch]),
        "ratio_vec_2": torch.stack([item["ratio_vec_2"] for item in batch]),
        "global_type": torch.stack([item["global_type"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
    }
    if "tokens" in batch[0]:
        result["tokens"] = [item["tokens"] for item in batch]
    if "article_id" in batch[0]:
        result["article_ids"] = torch.tensor([item["article_id"] for item in batch], dtype=torch.long)
    return result


def hybrid_periogt_live_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function for non-frozen LLM + PerioGT hybrid stream"""
    from torch_geometric.data import Batch
    from models.PerioGT.graph_builder import preprocess_batch_light

    graphs = Batch.from_data_list([item["graph"] for item in batch])
    num_nodes_per_graph = torch.diff(graphs.ptr).numpy()
    batch_idx_per_edge = graphs.batch[graphs.edge_index[0]]
    num_edges_per_graph = batch_idx_per_edge.bincount(minlength=graphs.num_graphs).numpy()
    graphs.path = preprocess_batch_light(num_nodes_per_graph, num_edges_per_graph, graphs.path)

    result = {
        "text": [item["text"] for item in batch],
        "graphs": graphs,
        "fp_1": torch.stack([item["fp_1"] for item in batch]),
        "md_1": torch.stack([item["md_1"] for item in batch]),
        "fp_2": torch.stack([item["fp_2"] for item in batch]),
        "md_2": torch.stack([item["md_2"] for item in batch]),
        "ratio_vec_1": torch.stack([item["ratio_vec_1"] for item in batch]),
        "ratio_vec_2": torch.stack([item["ratio_vec_2"] for item in batch]),
        "global_type": torch.stack([item["global_type"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
    }
    if "article_id" in batch[0]:
        result["article_ids"] = torch.tensor([item["article_id"] for item in batch], dtype=torch.long)
    return result


def hybrid_llm_struct_cached_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for frozen LLM + LLM struct hybrid mode
    Pad variable-length semantic and struct stream hidden states to their respective batch max lengths.
    """
    bsz = len(batch)
    max_text_len = max(item["text_hidden"].size(0) for item in batch)
    max_struct_len = max(item["struct_emb"].size(0) for item in batch)
    text_dim = batch[0]["text_hidden"].size(-1)
    struct_dim = batch[0]["struct_emb"].size(-1)

    text_hidden = torch.zeros(bsz, max_text_len, text_dim)
    text_mask = torch.zeros(bsz, max_text_len, dtype=torch.long)
    struct_emb = torch.zeros(bsz, max_struct_len, struct_dim)
    struct_mask_t = torch.zeros(bsz, max_struct_len, dtype=torch.long)

    for i, item in enumerate(batch):
        tl = item["text_hidden"].size(0)
        text_hidden[i, :tl] = item["text_hidden"]
        text_mask[i, :tl] = item["text_mask"]
        sl = item["struct_emb"].size(0)
        struct_emb[i, :sl] = item["struct_emb"]
        struct_mask_t[i, :sl] = item["struct_mask"]

    result = {
        "text_hidden": text_hidden,
        "text_mask": text_mask,
        "struct_emb": struct_emb,
        "struct_mask": struct_mask_t,
        "target": torch.stack([item["target"] for item in batch]),
    }
    if "tokens" in batch[0]:
        result["tokens"] = [item["tokens"] for item in batch]
    if "article_id" in batch[0]:
        result["article_ids"] = torch.tensor([item["article_id"] for item in batch], dtype=torch.long)
    return result


def struct_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for StructOnlyDataset (frozen mode)"""
    bsz = len(batch)
    max_smiles = max(item["struct_emb"].size(0) for item in batch)
    struct_dim = batch[0]["struct_emb"].size(-1)

    struct_emb = torch.zeros(bsz, max_smiles, struct_dim)
    struct_mask = torch.zeros(bsz, max_smiles, dtype=torch.long)
    targets = torch.stack([item["target"] for item in batch])

    for i, item in enumerate(batch):
        n = item["struct_emb"].size(0)
        struct_emb[i, :n] = item["struct_emb"]
        struct_mask[i, :n] = 1

    result = {
        "struct_emb": struct_emb,
        "struct_mask": struct_mask,
        "target": targets,
    }
    if "article_id" in batch[0]:
        result["article_ids"] = torch.tensor([item["article_id"] for item in batch], dtype=torch.long)
    return result


def live_struct_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function for LiveStructDataset (non-frozen mode)"""
    result = {
        "smiles_lists": [item["smiles_list"] for item in batch],
        "target": torch.stack([item["target"] for item in batch]),
    }
    if "ratio_vectors" in batch[0]:
        result["ratio_vectors_lists"] = [item.get("ratio_vectors", []) for item in batch]
    if "article_id" in batch[0]:
        result["article_ids"] = torch.tensor([item["article_id"] for item in batch], dtype=torch.long)
    return result


def build_struct_dataloaders(
    split_indices: Dict[str, List[int]],
    targets_scaled: np.ndarray,
    cfg: Dict[str, Any],
    struct_embeddings: Optional[List[torch.Tensor]] = None,
    data: Optional[List[Dict]] = None,
    periogt_data: Optional[List[Dict]] = None,
    dataset_name: str = "",
    ratio_encoding_path: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders for structure-only control group"""
    batch_size = cfg["BATCH_SIZE"]
    num_workers = cfg["NUM_WORKERS"]
    encoder_name = cfg.get("STRUCT_ENCODER", "polybert")
    frozen = cfg.get("FROZEN_BACKBONE", True)

    # ---------- article-aware ----------
    article_aware = _is_article_aware(cfg)
    article_ids = extract_article_ids(data) if (article_aware and data is not None) else None
    train_article_ids = article_ids if article_aware else None

    if encoder_name == "periogt":
        # PerioGT always trains with full parameters (uses precomputed graphs and features)
        train_ds = PerioGTDataset(split_indices["train"], periogt_data, targets_scaled, article_ids=train_article_ids)
        val_ds = PerioGTDataset(split_indices["val"], periogt_data, targets_scaled, article_ids=train_article_ids)
        test_ds = PerioGTDataset(split_indices["test"], periogt_data, targets_scaled, article_ids=train_article_ids)
        collate = periogt_collate_fn
    elif encoder_name == "llm":
        if frozen:
            # Standalone LLM: reduced structural fields, separate from type 9.
            model_key = cfg["MODEL_TYPE"]
            cached_struct = load_cached_embeddings(dataset_name, model_key, STRUCT_INPUT)
            train_ds = LLMStructCachedDataset(split_indices["train"], cached_struct, targets_scaled, article_ids=train_article_ids)
            val_ds = LLMStructCachedDataset(split_indices["val"], cached_struct, targets_scaled, article_ids=train_article_ids)
            test_ds = LLMStructCachedDataset(split_indices["test"], cached_struct, targets_scaled, article_ids=train_article_ids)
            collate = cached_collate_fn
        else:
            # Use the same reduced structural prompt in live mode.
            train_ds = LLMStructLiveDataset(split_indices["train"], data, targets_scaled, dataset_name, article_ids=train_article_ids)
            val_ds = LLMStructLiveDataset(split_indices["val"], data, targets_scaled, dataset_name, article_ids=train_article_ids)
            test_ds = LLMStructLiveDataset(split_indices["test"], data, targets_scaled, dataset_name, article_ids=train_article_ids)
            collate = live_collate_fn
    elif frozen:
        train_ds = StructOnlyDataset(split_indices["train"], struct_embeddings, targets_scaled, article_ids=train_article_ids)
        val_ds = StructOnlyDataset(split_indices["val"], struct_embeddings, targets_scaled, article_ids=train_article_ids)
        test_ds = StructOnlyDataset(split_indices["test"], struct_embeddings, targets_scaled, article_ids=train_article_ids)
        collate = struct_collate_fn
    else:
        ratio_enc = None
        if ratio_encoding_path is not None and os.path.exists(ratio_encoding_path):
            ratio_enc = load_ratio_encoding(ratio_encoding_path)
        train_ds = LiveStructDataset(split_indices["train"], data, targets_scaled, ratio_encoding=ratio_enc, article_ids=train_article_ids)
        val_ds = LiveStructDataset(split_indices["val"], data, targets_scaled, ratio_encoding=ratio_enc, article_ids=train_article_ids)
        test_ds = LiveStructDataset(split_indices["test"], data, targets_scaled, ratio_encoding=ratio_enc, article_ids=train_article_ids)
        collate = live_struct_collate_fn

    if article_aware:
        # Build article ID list corresponding to training set indices
        # (local indices 0..N-1 within the dataset)
        train_article_ids_local = [article_ids[idx] for idx in split_indices["train"]]
        train_sampler = ArticleGroupedBatchSampler(
            train_article_ids_local, batch_size)
        train_loader = DataLoader(
            train_ds, batch_sampler=train_sampler,
            num_workers=num_workers, collate_fn=collate, pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, collate_fn=collate, pin_memory=True,
        )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate, pin_memory=True,
    )
    return train_loader, val_loader, test_loader


# ======================== Model Weight Save / Load ========================

def save_trainable_weights(model, save_path: str):
    """Save only trainable parameters (for frozen/LoRA mode)"""
    ensure_dir(os.path.dirname(save_path))
    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    state = {k: v.cpu() for k, v in model.state_dict().items() if k in trainable_names}
    torch.save(state, save_path)


def load_trainable_weights(model, load_path: str, device: str = "cuda"):
    """Load trainable parameters"""
    state = torch.load(load_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)


# ======================== PerioGT Data Precomputation ========================

def precompute_periogt_data(dataset_name: str, data: List[Dict],
                            ratio_encoding: Dict,
                            device: str = "cuda",
                            use_prompt: bool = True,
                            use_cache: bool = False) -> List[Dict]:
    """
    Precompute graphs, features, and ratio information vectors for all data entries for PerioGT.
    Includes periodic prompt computation.

    Args:
        use_cache: If True, preferentially load from .pth cache file. Always compute and cache
                   with use_prompt=True (superset); the model's self.use_prompt flag determines
                   whether to use prompt data during forward pass.

    Returns:
        List[Dict], each Dict contains:
        - graph: PyG Data graph (with prompt attribute)
        - fp_1, md_1, fp_2, md_2: fingerprint and descriptor tensors
        - ratio_vec_1, ratio_vec_2: ratio information vectors
        - global_type: copolymer type
    """
    # Cache check: always cache with use_prompt=True result (superset),
    # the model's self.use_prompt flag determines whether to use prompt data during forward
    if use_cache and periogt_cache_exists(dataset_name):
        return load_periogt_cache(dataset_name)

    # When using cache mode, always compute with use_prompt=True (generate complete prompt data)
    if use_cache:
        use_prompt = True

    from rdkit import Chem
    from models.PerioGT.encoder import PERIOGT_DEFAULT_CONFIG, D_FP_FEATS, D_MD_FEATS
    from models.PerioGT.graph_builder import PolyGraphBuilder, preprocess_batch_light
    from models.PerioGT.features import (
        precompute_features, compute_single_smiles_features, safe_mol_from_smiles,
    )
    from models.PerioGT.aug import generate_multimer_smiles, periodicity_augment_traverse
    from models.PerioGT.light import LiGhTPredictor, init_params
    from models.PerioGT.vocab import Vocab
    from config.path import MODEL_PATHS, PERIOGT_CONFIG_PATH
    from sklearn.preprocessing import StandardScaler
    from tqdm import tqdm
    import yaml

    # Load configuration
    if os.path.exists(PERIOGT_CONFIG_PATH):
        with open(PERIOGT_CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f).get('base', PERIOGT_DEFAULT_CONFIG)
    else:
        config = PERIOGT_DEFAULT_CONFIG

    d_g_feats = config['d_g_feats']
    max_prompt = 10
    vocab = Vocab()

    # Load pretrained model for prompt generation (only needed when use_prompt=True)
    pretrained_path = MODEL_PATHS["periogt"]
    if use_prompt:
        pretrain_model = LiGhTPredictor(
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
            n_node_types=vocab.vocab_size,
        )
        if os.path.exists(pretrained_path):
            state_dict = torch.load(pretrained_path, map_location='cpu', weights_only=True)
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            pretrain_model.load_state_dict(state_dict, strict=False)
        pretrain_model.to(device).eval()

        # Prompt graph builder
        prompt_builder = PolyGraphBuilder(max_length=config['path_length'],
                                          n_local_nodes=2, n_global_nodes=0)
    else:
        pretrain_model = None
        prompt_builder = None

    # Copolymer graph builder (always needed)
    copolym_builder = PolyGraphBuilder(max_length=config['path_length'],
                                       n_local_nodes=3, n_global_nodes=1)

    # ---------- Step 1: Collect all unique SMILES and precompute fp/md ----------
    all_smiles_set = set()
    for entry in data:
        chem = entry.get("chemical_composition", {})
        monomers = chem.get("monomers", [])
        for m in monomers:
            smi = m.get("smiles", "")
            if smi and smi.strip():
                all_smiles_set.add(smi.strip())

    unique_smiles = list(all_smiles_set)
    valid_unique_smiles = []
    invalid_unique_smiles = []
    for smi in unique_smiles:
        if safe_mol_from_smiles(smi) is None:
            invalid_unique_smiles.append(smi)
        else:
            valid_unique_smiles.append(smi)

    print(
        f"    Precomputing features for {len(unique_smiles)} unique SMILES "
        f"({len(valid_unique_smiles)} valid)..."
    )
    if invalid_unique_smiles:
        print(
            f"    Found {len(invalid_unique_smiles)} invalid SMILES; "
            f"using placeholder graphs and zero fp/md features for those components."
        )

    valid_smiles_set = set(valid_unique_smiles)

    # Batch precompute multimer features (for prompt generation and fp/md input during fine-tuning)
    feat_cache = {}
    if valid_unique_smiles:
        feat_cache = precompute_features(valid_unique_smiles, units=(3, 6, 9),
                                         workers=max(1, os.cpu_count() - 1))

    # Also compute the fp/md for each SMILES itself (1-mer features)
    single_feat_cache = {}
    for smi in tqdm(valid_unique_smiles, desc="Single-molecule features", ncols=100):
        fp, md = compute_single_smiles_features(smi)
        if fp is not None:
            single_feat_cache[smi] = (fp, md)

    # ---------- Step 2: Fit descriptor scaler ----------
    md_scaler = None
    all_md = []
    for smi in valid_unique_smiles:
        for u in (3, 6, 9):
            result = feat_cache.get((smi, u))
            if result is not None and result[1] is not None:
                all_md.append(result[1])
    if all_md:
        all_md_array = np.array(all_md, dtype=np.float64)
        md_scaler = StandardScaler()
        md_scaler.fit(all_md_array)

    # ---------- Step 3: Generate prompt for each unique SMILES ----------
    prompt_dict = {}
    if use_prompt:
        print(f"    Generating prompts for {len(valid_unique_smiles)} valid unique SMILES...")
        for smi in valid_unique_smiles:
            try:
                prompt = _generate_prompt_for_smiles(
                    smi, feat_cache, md_scaler, pretrain_model,
                    prompt_builder, d_g_feats, max_prompt, device
                )
                prompt_dict[smi] = prompt
            except Exception:
                prompt_dict[smi] = None

    # ---------- Step 4: Build PerioGT data for each entry ----------
    all_periogt_data = []

    for i, entry in enumerate(data):
        chem = entry.get("chemical_composition", {})
        monomers = chem.get("monomers", [])
        is_homo = chem.get("is_homopolymer", True)

        monomer_slots = list(monomers[:2])
        if not monomer_slots:
            monomer_slots = [{}]

        # Always ensure 2 component slots to avoid inconsistent virtual node counts across batches
        if len(monomer_slots) == 1:
            monomer_slots = monomer_slots + monomer_slots

        slot_infos = []
        for monomer in monomer_slots:
            raw_smiles = monomer.get("smiles")
            if isinstance(raw_smiles, str):
                raw_smiles = raw_smiles.strip() or None
            else:
                raw_smiles = None

            is_valid = raw_smiles in valid_smiles_set
            slot_infos.append({
                "monomer": monomer,
                "raw_smiles": raw_smiles,
                "graph_smiles": raw_smiles if is_valid else "C",
                "is_valid": is_valid,
            })

        graph_smiles_list = [slot["graph_smiles"] for slot in slot_infos]

        # Build copolymer graph
        graph = copolym_builder.build_copolymer_graph(graph_smiles_list)

        # Get fp/md features (use 3-mer features, consistent with source code)
        fp_list, md_list = [], []
        for slot in slot_infos:
            smi = slot["raw_smiles"]
            result = feat_cache.get((smi, 3)) if slot["is_valid"] and smi is not None else None
            if result is not None and result[0] is not None:
                fp = np.array(result[0], dtype=np.float32)
                md = np.array(result[1], dtype=np.float32)
                # Normalize descriptors
                if md_scaler is not None:
                    md = md_scaler.transform(md.reshape(1, -1)).flatten().astype(np.float32)
                fp_list.append(torch.FloatTensor(fp))
                md_list.append(torch.FloatTensor(md))
            else:
                fp_list.append(torch.zeros(D_FP_FEATS))
                md_list.append(torch.zeros(D_MD_FEATS))

        # Pad to 2 components
        while len(fp_list) < 2:
            fp_list.append(torch.zeros(D_FP_FEATS))
            md_list.append(torch.zeros(D_MD_FEATS))

        # Build ratio vectors
        ratio_vecs = []
        for slot in slot_infos[:2]:
            rv = build_ratio_vector(slot["monomer"], is_homo, ratio_encoding)
            ratio_vecs.append(torch.FloatTensor(rv))
        while len(ratio_vecs) < 2:
            ratio_vecs.append(torch.zeros(ratio_encoding["vector_dim"]))

        # Assemble prompt into copolymer graph
        n_nodes = graph.num_nodes
        prompt_feats = torch.zeros(n_nodes, max_prompt, d_g_feats)

        if use_prompt:
            # Place each component's prompt at the corresponding position in the copolymer graph
            # Component 1 triplet nodes are arranged first, component 2 follows
            node_offset = 0
            for slot in slot_infos:
                prompt = prompt_dict.get(slot["raw_smiles"]) if slot["is_valid"] else None
                if prompt is not None:
                    n_prompt_nodes = prompt.size(0)
                    n_fill = min(n_prompt_nodes, n_nodes - node_offset)
                    if n_fill > 0:
                        prompt_feats[node_offset:node_offset + n_fill] = prompt[:n_fill]
                # Count triplet + local virtual nodes for current component
                try:
                    mol = safe_mol_from_smiles(slot["graph_smiles"])
                    if mol is not None:
                        n_bonds = mol.GetNumBonds()
                        n_unbonded = sum(1 for a in mol.GetAtoms()
                                         if a.GetDegree() == 0)
                        # triplet nodes + unbonded atoms + local virtual nodes (3)
                        node_offset += n_bonds + n_unbonded + 3
                    else:
                        node_offset += 4  # fallback
                except Exception:
                    node_offset += 4

        graph.prompt = prompt_feats

        # Global type: homopolymer or single-component input is 0, otherwise 1.
        nonempty_slot_count = sum(1 for slot in slot_infos if slot["raw_smiles"] is not None)
        global_type = torch.tensor(0 if is_homo or nonempty_slot_count <= 1 else 1,
                       dtype=torch.long)

        all_periogt_data.append({
            'graph': graph,
            'fp_1': fp_list[0], 'md_1': md_list[0],
            'fp_2': fp_list[1], 'md_2': md_list[1],
            'ratio_vec_1': ratio_vecs[0], 'ratio_vec_2': ratio_vecs[1],
            'global_type': global_type,
        })

        if (i + 1) % 100 == 0:
            print(f"    PerioGT preprocessed {i + 1}/{len(data)} entries")

    if pretrain_model is not None:
        del pretrain_model
        torch.cuda.empty_cache()
    print(f"  PerioGT data precomputed: {len(all_periogt_data)} entries")

    # Cache to disk
    if use_cache:
        save_periogt_cache(dataset_name, all_periogt_data)

    return all_periogt_data


def _generate_prompt_for_smiles(base_smiles: str, feat_cache: Dict,
                                md_scaler, pretrain_model,
                                prompt_builder, d_g_feats: int,
                                max_prompt: int, device: str):
    """
    Generate periodic prompt features for a single SMILES.
    Periodicity augment and multimer generate augmented SMILES,
    use pretrained model to obtain node embeddings, and map back to base graph nodes.

    Returns:
        prompt_feats: (n_base_nodes, max_prompt, d_g_feats)
    """
    from models.PerioGT.aug import generate_multimer_smiles, periodicity_augment_traverse
    from models.PerioGT.features import compute_single_smiles_features

    # Build base prompt graph (to get base node count and bid mapping)
    base_graph = prompt_builder.build_prompt_graph(base_smiles)
    if base_graph is None:
        return None
    n_base_nodes = base_graph.num_nodes
    base_bids = base_graph.bid.numpy()

    # Generate augmented SMILES
    try:
        pa_smiles = periodicity_augment_traverse(base_smiles)
    except Exception:
        pa_smiles = [base_smiles]

    # Generate multimer (1x, 2x, 3x) for each periodicity variant
    augmented_smiles = []
    augmented_n_units = []
    for pa in pa_smiles:
        for n_units in [1, 2, 3]:
            try:
                ms = generate_multimer_smiles(num_repeat_units=n_units, smiles=pa)
                if ms is not None:
                    augmented_smiles.append(ms)
                    augmented_n_units.append(n_units)
            except Exception:
                continue

    # Randomly select at most max_prompt entries (keep the first — base itself)
    if len(augmented_smiles) > max_prompt:
        kept = [0]  # Keep base
        others = list(range(1, len(augmented_smiles)))
        import random as _rand
        _rand.shuffle(others)
        kept.extend(others[:max_prompt - 1])
        kept.sort()
        augmented_smiles = [augmented_smiles[k] for k in kept]
        augmented_n_units = [augmented_n_units[k] for k in kept]

    # Build graph and get embeddings for each augmented SMILES
    prompt_feats = torch.zeros(n_base_nodes, max_prompt, d_g_feats)
    prompt_idx = 0

    for aug_smi, n_units in zip(augmented_smiles, augmented_n_units):
        if prompt_idx >= max_prompt:
            break
        try:
            aug_graph = prompt_builder.build_prompt_graph(aug_smi)
            if aug_graph is None:
                continue

            # Get fp/md from feat_cache (using corresponding n_units cache)
            # Original author mapping: n_units {1→3, 2→6, 3→9}
            cache_units = {1: 3, 2: 6, 3: 9}.get(n_units, 3)
            cache_key = (base_smiles, cache_units)
            cache_result = feat_cache.get(cache_key)
            if cache_result is None or cache_result[0] is None:
                # Fallback: compute directly
                fp, md = compute_single_smiles_features(aug_smi)
                if fp is None:
                    continue
            else:
                fp, md = cache_result
                fp = np.array(fp, dtype=np.float32)
                md = np.array(md, dtype=np.float32)

            # Normalize descriptors
            if md_scaler is not None:
                md = md_scaler.transform(md.reshape(1, -1)).flatten().astype(np.float32)

            aug_fp = torch.FloatTensor(fp).unsqueeze(0).to(device)
            aug_md = torch.FloatTensor(md).unsqueeze(0).to(device)
            aug_graph = aug_graph.to(device)

            with torch.no_grad():
                embs, bids = pretrain_model.generate_node_emb(
                    aug_graph, aug_fp, aug_md
                )  # embs: (N, d_g_feats), bids: (N,)

            # Map augmented graph node embeddings back to base graph
            # Mapping strategy: match by bid, ignore virtual nodes with bid=-1
            aug_bids = bids.cpu().numpy()
            embs_cpu = embs.cpu()

            # Assign values for each bid-corresponding node in the base graph
            aug_bid_to_emb = {}
            for node_idx in range(len(aug_bids)):
                bid = int(aug_bids[node_idx])
                if bid >= 0:
                    aug_bid_to_emb[bid] = embs_cpu[node_idx]

            for node_idx in range(n_base_nodes):
                bid = int(base_bids[node_idx])
                if bid >= 0 and bid in aug_bid_to_emb:
                    prompt_feats[node_idx, prompt_idx] = aug_bid_to_emb[bid]

            prompt_idx += 1

        except Exception:
            continue

    return prompt_feats
