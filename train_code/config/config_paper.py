"""Reported model configurations for the existing train.py entry point.

Select this module with CONFIG_MODULE = "config.config_paper" in train.py.
These are fixed experimental settings, not a new hyperparameter search.
"""
from importlib import import_module as _import_module

# Models: HHT, HHT-C, LLM-struct, PolyBERT, PerioGT,
#         HHT+PolyBERT, HHT+PerioGT (Supplementary Note 15).
PAPER_MODEL = "HHT"
# Protocols: random, article, article_holdout (Supplementary Note 14).
PAPER_PROTOCOL = "article_holdout"


def _settings(model, protocol):
    splits = {"random": "split_random.pkl", "article": "split_article.pkl",
              "article_holdout": "split_holdout_whole.pkl"}
    structural = {"PolyBERT": "polybert", "PerioGT": "periogt", "LLM-struct": "llm"}
    semantic = {"HHT", "HHT-C", "HHT+PolyBERT", "HHT+PerioGT"}
    if model not in structural and model not in semantic:
        raise ValueError(f"Unknown PAPER_MODEL: {model}")
    if protocol not in splits:
        raise ValueError(f"Unknown PAPER_PROTOCOL: {protocol}")
    if model.startswith("HHT+") and protocol != "random":
        raise ValueError("The reported hybrid experiments use the random protocol.")
    base = _import_module("config.config_stru_article" if model in structural
                          else "config.config_article")
    cfg = {k: v for k, v in vars(base).items() if k.isupper()}
    cfg.update(NUM_GPUS=1, SEEDS=[42, 62, 82], LOG_TARGET_DATASETS=[],
               SWEEP_TRAIN_FRACTIONS=False, SPLIT_PKL=splits[protocol])
    if model in structural:
        cfg["STRUCT_ENCODER"] = structural[model]
        if model == "LLM-struct":
            cfg.update(MODEL_TYPE="chemdfm_v1_5_8b", MLP_HIDDEN_DIM=[128, 64],
                       STRUCT_POOLING_TYPE="sigmoid_pooling")
    elif model == "HHT-C":
        cfg.update(INPUT_CONTENT=[9], EVAL_INPUT_CONTENT=[9])
    elif model.startswith("HHT+"):
        cfg.update(ENABLE_HYBRID_ENCODING=True,
                   STRUCT_ENCODER="polybert" if model == "HHT+PolyBERT" else "periogt",
                   HYBRID_PROJ_DIM=512, HYBRID_GATE_DIM=256, MLP_HIDDEN_DIM=[128, 64])
    return cfg


globals().update(_settings(PAPER_MODEL, PAPER_PROTOCOL))
