"""
path.py — Centralized path management (relative paths, project-root based)
"""
import os

# Project root (this file is under config/, so root is parent of parent)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ======================== Datasets ========================
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")

DATASET_PATHS = {
    "Tg": os.path.join(DATASET_DIR, "Tg", "Tg.json"),
    "Tm": os.path.join(DATASET_DIR, "Tm", "Tm.json"),
    "n":  os.path.join(DATASET_DIR, "n", "n.json"),
    "eps": os.path.join(DATASET_DIR, "eps", "eps.json"),
    "E":  os.path.join(DATASET_DIR, "E", "E.json"),
    "UTS": os.path.join(DATASET_DIR, "UTS", "UTS.json"),
}

# Split index pkl directory (same level as corresponding JSON)
def get_split_pkl_path(dataset_name: str, pkl_filename: str) -> str:
    return os.path.join(DATASET_DIR, dataset_name, pkl_filename)

# ======================== Validation Datasets ========================
VAL_DATASET_DIR = os.path.join(PROJECT_ROOT, "val_dataset")

# ======================== Continue Training Datasets ========================
CONTINUE_TRAIN_DATASET_DIR = os.path.join(PROJECT_ROOT, "continue_train_dataset")

# ======================== autodl-tmp ========================
AUTODL_TMP_DIR = os.path.join(PROJECT_ROOT, "autodl-tmp")

# Model weights storage directory
MODEL_WEIGHTS_DIR = os.path.join(AUTODL_TMP_DIR, "model_weights")

# Model paths
MODEL_PATHS = {
    "qwen3_4b_instruct_2507": os.path.join(MODEL_WEIGHTS_DIR, "Qwen","Qwen3-4B-Instruct-2507"),
    "qwen3_4b_thinking_2507": os.path.join(MODEL_WEIGHTS_DIR, "Qwen","Qwen3-4B-Thinking-2507"),
    "qwen3_4b_base":          os.path.join(MODEL_WEIGHTS_DIR, "Qwen","Qwen3-4B-Base"),
    "qwen3_8b_base":          os.path.join(MODEL_WEIGHTS_DIR, "Qwen","Qwen3-8B-Base"),
    "qwen3_0_6b_base":        os.path.join(MODEL_WEIGHTS_DIR, "Qwen","Qwen3-0.6B-Base"),
    "chemdfm_v1_5_8b":        os.path.join(MODEL_WEIGHTS_DIR, "OpenDFM","ChemDFM-v1.5-8B"),
    "qwen3_4b_base_cpt_1":    os.path.join(MODEL_WEIGHTS_DIR, "CPT","qwen3_4b_base_cpt_r4"),
    "qwen3_4b_base_cpt_2":    os.path.join(MODEL_WEIGHTS_DIR, "CPT","qwen3_4b_base_cpt_r8"),
    "qwen3_4b_base_cpt_3":    os.path.join(MODEL_WEIGHTS_DIR, "CPT","qwen3_4b_base_cpt_r16"),
    "polybert":                os.path.join(MODEL_WEIGHTS_DIR, "PolyBERT"),
    "periogt":                 os.path.join(MODEL_WEIGHTS_DIR, "PerioGT", "base.pth"),
}

# PerioGT config path (uses config.yaml in code directory)
PERIOGT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "models", "PerioGT", "config.yaml")

# Embedding cache directory
EMBEDDING_CACHE_DIR = os.path.join(AUTODL_TMP_DIR, "embedding_cache")

# Training results output directory
RESULT_DIR = os.path.join(AUTODL_TMP_DIR, "results")

# ======================== TensorBoard Logs ========================
TB_LOG_DIR = os.path.join(PROJECT_ROOT, "tf-logs")


def ensure_dir(path: str):
    """Create directory if it does not exist"""
    os.makedirs(path, exist_ok=True)
