"""Precompute selected main and domain inputs using the training cache format."""

# Run models -> datasets -> input types, preserving these list orders.
DATASETS = ["Tg", "Tm", "E", "UTS", "eps", "n"]
# Domain train/validation and test JSONs are cached separately from main data.
# Set DATASETS=[] for domain-only work; E_article uses its separate article split.
DOMAIN_DATASETS = []  # e.g. ["E", "Tm", "E_article"]
LLM_MODELS = ['chemdfm_v1_5_8b','qwen3_4b_instruct_2507','qwen3_4b_thinking_2507','qwen3_4b_base','qwen3_8b_base','qwen3_0_6b_base']
INPUT_TYPES = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # also accepts ["type1", "type2", ...]
# For config_stru_article.py with STRUCT_ENCODER="llm", set INPUT_TYPES = ["struct"].
# Mixed lists such as [2, 9, "struct"] reuse one LLM; "struct" has its own cache.
BATCH_SIZE = 3  # embedding forward-pass batch size; increase to fit available GPU memory

import gc
import torch
from config.path import DATASET_PATHS, MODEL_PATHS
from utils import (
    cache_exists, load_dataset, load_embedding_llm,
    normalize_input_types, precompute_llm_embeddings,
)
from domain_embedding_cache import (
    domain_cache_exists, domain_sources, precompute_domain_embeddings,
)


def main():
    types = normalize_input_types(INPUT_TYPES)
    if isinstance(BATCH_SIZE, bool) or not isinstance(BATCH_SIZE, int) or BATCH_SIZE < 1:
        raise ValueError("BATCH_SIZE must be a positive integer.")
    datasets = list(dict.fromkeys(DATASETS))
    domain_datasets = list(dict.fromkeys(DOMAIN_DATASETS))
    models = list(dict.fromkeys(LLM_MODELS))
    if not (datasets or domain_datasets) or not models:
        raise ValueError("Select DATASETS and/or DOMAIN_DATASETS, and at least one LLM_MODELS entry.")
    for name in datasets:
        if name not in DATASET_PATHS:
            raise ValueError(f"Unknown dataset: {name!r}; choose from {list(DATASET_PATHS)}")
    for key in models:
        if key not in MODEL_PATHS or key in ("polybert", "periogt"):
            raise ValueError(f"Unknown LLM model: {key!r}")
    for name in domain_datasets:
        domain_sources(name)  # Validate both training and test sources before loading a model.

    for model_key in models:
        llm = None
        try:
            for dataset_name in datasets:
                if cache_exists(dataset_name, model_key, types):
                    print(f"[Skip] {model_key}/{dataset_name}: requested types already cached.")
                    continue
                data = load_dataset(dataset_name)
                if llm is None:
                    llm = load_embedding_llm(model_key)
                precompute_llm_embeddings(
                    dataset_name, model_key, data, batch_size=BATCH_SIZE,
                    input_types=types, llm=llm,
                )
                del data
                gc.collect()
            for domain_name in domain_datasets:
                if domain_cache_exists(domain_name, model_key, types):
                    print(f"[Skip] {model_key}/domain:{domain_name}: requested types already cached.")
                    continue
                if llm is None:
                    llm = load_embedding_llm(model_key)
                precompute_domain_embeddings(
                    domain_name, model_key, types, batch_size=BATCH_SIZE, llm=llm,
                )
                gc.collect()
        finally:
            del llm
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
