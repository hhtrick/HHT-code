"""Disk-backed LLM caches shared by domain training and precomputation.

Each complete source JSON has its own cache in embedding_cache. Training and
validation use index views of the training JSON; test data has a separate cache.
The token writer and on-disk format are the same as main-dataset training.
"""
import gc
import json
import os
import pickle
from pathlib import Path

import torch

from config.path import CONTINUE_TRAIN_DATASET_DIR, VAL_DATASET_DIR, EMBEDDING_CACHE_DIR
from utils import (
    MemmapEmbeddingCache, _write_llm_embedding_type, cache_exists,
    get_cache_meta_filename, get_cache_npy_filename, load_embedding_llm,
    normalize_input_types,
)


def domain_sources(domain_name, test_file=None):
    """Return (cache namespace, JSON path) pairs, distinct from main data."""
    training = Path(CONTINUE_TRAIN_DATASET_DIR) / domain_name / f"{domain_name}.json"
    test = Path(test_file) if test_file is not None else Path(VAL_DATASET_DIR) / f"val_{domain_name}_1.json"
    if not training.is_file() or not test.is_file():
        raise FileNotFoundError(f"Domain data files are required: {training}; {test}")
    # Filename scope is unambiguous for published val_dataset files. Do not
    # silently collide if a caller supplies another directory with that name.
    if test.resolve().parent != Path(VAL_DATASET_DIR).resolve():
        raise ValueError(f"Domain test file must be under {VAL_DATASET_DIR}: {test}")
    return [(f"domain_{domain_name}_trainval", training),
            (f"domain_{test.stem}_test", test)]


def _cache_paths(namespace, model_key, input_type):
    return (
        os.path.join(EMBEDDING_CACHE_DIR, get_cache_npy_filename(namespace, model_key, input_type)),
        os.path.join(EMBEDDING_CACHE_DIR, get_cache_meta_filename(namespace, model_key, input_type)),
    )


def domain_cache_exists(domain_name, model_key, input_types, test_file=None):
    types = normalize_input_types(input_types)
    return all(cache_exists(namespace, model_key, types)
               for namespace, _ in domain_sources(domain_name, test_file))


def precompute_domain_embeddings(domain_name, model_key, input_types, batch_size=1,
                                 llm=None, test_file=None):
    """Write missing types only, keeping at most one hidden-state batch in RAM.

    A borrowed LLM remains loaded for the caller to reuse across datasets. A
    cache hit never loads the backbone. Input names and prompt construction
    (including ``struct``) are shared with main training.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    types = normalize_input_types(input_types)
    sources = domain_sources(domain_name, test_file)
    pending = [(namespace, path, t) for namespace, path in sources for t in types
               if not cache_exists(namespace, model_key, [t])]
    if not pending:
        print(f"[Domain cache] {domain_name}/{model_key}: requested types already cached.")
        return
    if llm is not None and llm.MODEL_KEY != model_key:
        raise ValueError("Borrowed LLM model does not match the requested domain cache")
    os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)
    owns_model = llm is None
    if owns_model:
        llm = load_embedding_llm(model_key)
    try:
        for namespace, path in sources:
            missing = [t for t in types if not cache_exists(namespace, model_key, [t])]
            if not missing:
                continue
            with path.open(encoding="utf-8") as handle:
                data = json.load(handle)
            if not data:
                raise ValueError(f"Cannot embed an empty domain source: {path}")
            properties = data[0].get("properties", {})
            prop_name = domain_name if domain_name in properties else next(
                (key for key, value in properties.items() if isinstance(value, dict) and "value" in value), None)
            if prop_name is None:
                raise ValueError(f"Cannot infer target property from {path}")
            for t in missing:
                npy_path, meta_path = _cache_paths(namespace, model_key, t)
                print(f"[Domain cache] Computing {namespace}/{model_key}/{t} ({len(data)} entries)...")
                _write_llm_embedding_type(llm, data, prop_name, t, batch_size, npy_path, meta_path)
                gc.collect()
                torch.cuda.empty_cache()
            del data
    finally:
        if owns_model:
            del llm
            gc.collect()
            torch.cuda.empty_cache()


class DomainCacheView:
    """Lazy row-index view exposing the names expected by domain training."""
    def __init__(self, cache, indices=None):
        self.cache = cache
        self.indices = list(range(len(cache))) if indices is None else list(indices)
        if any(i < 0 or i >= len(cache) for i in self.indices):
            raise IndexError("Domain split contains an index outside its embedding cache")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = self.cache[self.indices[index]]
        return {"hidden": row["hidden_states"], "mask": row["attention_mask"]}


def load_domain_cached_splits(domain_name, model_key, input_type, test_file=None,
                              split_pkl_name="split_continue_train.pkl"):
    """Return train/validation/test views in precisely the published row order."""
    input_type = normalize_input_types([input_type])[0]
    sources = domain_sources(domain_name, test_file)
    caches = []
    for namespace, path in sources:
        cached = MemmapEmbeddingCache(*_cache_paths(namespace, model_key, input_type))
        with path.open(encoding="utf-8") as handle:
            n_entries = len(json.load(handle))
        if len(cached) != n_entries:
            raise ValueError(f"Cache row count differs from source {path}; recompute its cache")
        caches.append(cached)
    split_path = Path(CONTINUE_TRAIN_DATASET_DIR) / domain_name / split_pkl_name
    with split_path.open("rb") as handle:
        split = pickle.load(handle)
    return (DomainCacheView(caches[0], split.get("train", [])),
            DomainCacheView(caches[0], split.get("val", [])),
            DomainCacheView(caches[1]))
