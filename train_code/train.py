"""
train.py — Main training script: supports single-config training and YAML grid search
"""
import os
import sys
import copy
import json
import importlib
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from typing import Dict, Any, List

from config.path import (
    RESULT_DIR, TB_LOG_DIR,
    ensure_dir,
)
from utils import (
    load_dataset, load_split_indices, extract_targets,
    fit_scaler, transform_targets, inverse_transform,
    precompute_llm_embeddings, precompute_struct_embeddings,
    load_cached_struct_embeddings,
    build_dataloaders, build_struct_dataloaders, set_seed,
    compute_metrics, plot_parity,
    save_config_yaml, save_csv_summary,
    save_trainable_weights,
    scan_and_save_ratio_encoding, load_ratio_encoding,
    precompute_periogt_data, load_periogt_cache,
    evaluate_test_set,
    extract_article_ids, _is_article_aware,
)
from models.base_model import PropertyPredictionModel, StructureOnlyModel

# ======================== Training Parameter Settings ========================
# Datasets to train on (list)
TRAIN_DATASETS = ["Tg","Tm","E","UTS","eps","n"]

# Single-config training module: "config.config_article" | "config.config_stru_article"
CONFIG_MODULE = "config.config_article"

# YAML grid search config file path (set to enable YAML grid search)
# Set to None to use CONFIG_MODULE for single-config training
# Example: "config/grid/exp1_semantic_frozen.yaml"
GRID_YAML = None
# ==============================================================


def load_config_as_dict(module_name: str) -> Dict[str, Any]:
    """Load public variables from a config module as a dictionary"""
    mod = importlib.import_module(module_name)
    cfg = {}
    for key in dir(mod):
        if key.startswith("_"):
            continue
        if not key.isupper() and not (key[0].isupper() and "_" in key):
            # Only keep all-uppercase or uppercase+underscore variable names (e.g., BATCH_SIZE, LR), exclude class names and functions
            continue
        val = getattr(mod, key)
        # Exclude modules, classes, functions and other non-serializable objects
        if isinstance(val, type) or callable(val) or isinstance(val, type(sys)):
            continue
        cfg[key] = val
    return cfg


def _is_struct_only(cfg: Dict[str, Any]) -> bool:
    """Check if using structural stream-only configuration (no semantic LLM stream)"""
    return "STRUCT_ENCODER" in cfg and "ENABLE_HYBRID_ENCODING" not in cfg


def _resolve_visible_gpu_count(cfg: Dict[str, Any], context: str = "training") -> int:
    """Resolve the actual number of GPUs to use based on visible devices in the current process."""
    requested_gpus = max(int(cfg.get("NUM_GPUS", 0)), 0)
    visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    effective_gpus = min(requested_gpus, visible_gpus)

    if requested_gpus > visible_gpus:
        print(
            f"[GPU] Requested {requested_gpus} GPU(s) for {context}, "
            f"but only {visible_gpus} visible in this process. Using {effective_gpus}."
        )

    return effective_gpus


def _build_trainer_device_kwargs(cfg: Dict[str, Any], context: str = "training") -> Dict[str, Any]:
    """Generate Lightning Trainer device kwargs matching the GPUs visible to the current process."""
    num_gpus = _resolve_visible_gpu_count(cfg, context=context)
    return {
        "accelerator": "gpu" if num_gpus > 0 else "cpu",
        "devices": num_gpus if num_gpus > 0 else "auto",
        "strategy": "ddp" if num_gpus > 1 else "auto",
        "num_gpus": num_gpus,
    }


def _extract_grid_values(search_space: Dict, param_name: str,
                         linked_unpacking: Dict, default) -> List:
    """Extract all candidate values for a parameter from the search space (including linked_grid values)"""
    values = set()
    # Directly in grid
    if param_name in search_space:
        domain = search_space[param_name]
        if hasattr(domain, 'categories'):
            for v in domain.categories:
                values.add(v)
    # In linked_grid
    for link_key, param_names in linked_unpacking.items():
        if param_name in param_names:
            idx = param_names.index(param_name)
            if link_key in search_space:
                domain = search_space[link_key]
                if hasattr(domain, 'categories'):
                    for combo in domain.categories:
                        values.add(combo[idx])
    if not values:
        values.add(default)
    return list(values)


def load_grid_yaml(yaml_path: str):
    """
    Load a YAML grid search configuration file.
    The YAML contains only three fields: base_config, grid, and linked_grid.
    Other parameters (including Ray parallelism settings) are set in the config module
    specified by base_config.
    Returns: (base_cfg, search_space, grid_param_names, linked_unpacking)
    - base_cfg: base configuration (without tune objects)
    - search_space: {param_name: tune.grid_search(...)} dict
    - grid_param_names: list of all parameter names participating in grid search (including linked param names after expansion)
    - linked_unpacking: {link_key: [param_names]} used to expand linked params in each trial
    """
    import yaml as yaml_mod
    from ray import tune

    with open(yaml_path, 'r', encoding='utf-8') as f:
        yaml_cfg = yaml_mod.safe_load(f)

    # 1. Load base config module
    base_module = yaml_cfg.get('base_config', 'config.config')
    base_cfg = load_config_as_dict(base_module)

    # 2. Build grid search space
    grid = yaml_cfg.get('grid', {})
    search_space = {}
    grid_param_names = []

    for key, spec in grid.items():
        if isinstance(spec, dict) and '_generate' in spec:
            gen_type = spec['_generate']
            if gen_type == 'pairs_d1_gt_d2':
                values = []
                for d1 in spec['d1']:
                    for d2 in spec['d2']:
                        if d1 > d2:
                            values.append([d1, d2])
                search_space[key] = tune.grid_search(values)
                grid_param_names.append(key)
            elif gen_type == 'pairs_d1_eq_2d2':
                values = []
                for d1 in spec['d1']:
                    for d2 in spec['d2']:
                        if d1 == 2 * d2:
                            values.append([d1, d2])
                search_space[key] = tune.grid_search(values)
                grid_param_names.append(key)
            else:
                raise ValueError(f"Unknown grid generator: {gen_type}")
        elif isinstance(spec, list):
            search_space[key] = tune.grid_search(spec)
            grid_param_names.append(key)
        else:
            # Single value, treat as fixed parameter
            base_cfg[key] = spec

    # 3. Handle linked parameter groups (linked_grid)
    linked_grid = yaml_cfg.get('linked_grid', {})
    linked_unpacking = {}
    for link_key, link_spec in linked_grid.items():
        params = link_spec['params']
        values = [tuple(v) for v in link_spec['values']]
        search_space[link_key] = tune.grid_search(values)
        linked_unpacking[link_key] = params
        grid_param_names.extend(params)

    return base_cfg, search_space, grid_param_names, linked_unpacking


def train_single_config(cfg: Dict[str, Any], dataset_name: str,
                         experiment_dir: str, skip_precompute: bool = False,
                         skip_test: bool = False):
    """
    Run multi-seed training on a dataset with a fixed config.
    Supports semantic/hybrid stream mode and structural stream control mode.

    Args:
        skip_test: If True, skip test set evaluation and only evaluate on validation set
                   (used for grid search to avoid data leakage).
    """
    torch.set_float32_matmul_precision(cfg["FLOAT32_MATMUL_PRECISION"])

    struct_only = _is_struct_only(cfg)

    # ---- Load data ----
    data = load_dataset(dataset_name)
    split_pkl = cfg["SPLIT_PKL"]
    split_indices = load_split_indices(dataset_name, split_pkl)
    targets_raw = extract_targets(data, dataset_name)
    scaler = fit_scaler(targets_raw, split_indices["train"])
    targets_scaled = transform_targets(scaler, targets_raw)

    # Save scaler
    import pickle as pkl
    scaler_path = os.path.join(experiment_dir, "scaler.pkl")
    ensure_dir(experiment_dir)
    # ratio_encoding.json is saved in the experiment directory (alongside model weights, results, etc.)
    # In grid search mode, _RATIO_ENCODING_PATH is pre-injected by train_ray_tune
    ratio_path = cfg.get("_RATIO_ENCODING_PATH") or os.path.join(experiment_dir, "ratio_encoding.json")
    cfg["_RATIO_ENCODING_PATH"] = ratio_path
    with open(scaler_path, "wb") as f:
        pkl.dump(scaler, f)

    frozen = cfg.get("FROZEN_BACKBONE", True)

    if struct_only:
        # ---- Structural stream control group mode ----
        encoder_name = cfg.get("STRUCT_ENCODER", "polybert")
        frozen_struct = cfg.get("FROZEN_BACKBONE", True)
        model_key = f"struct_{encoder_name}"

        # Ensure ratio encoding exists
        if not skip_precompute:
            if not os.path.exists(ratio_path):
                print(f"\n[Ratio Encoding] Scanning datasets...")
                scan_and_save_ratio_encoding([dataset_name], save_path=ratio_path)

        periogt_data = None
        if encoder_name == "periogt":
            # PerioGT always trains with full parameters
            frozen_struct = False
            if not os.path.exists(ratio_path):
                scan_and_save_ratio_encoding([dataset_name], save_path=ratio_path)
            ratio_encoding = load_ratio_encoding(ratio_path)
            if skip_precompute:
                # Ray trial: load from cache (precomputed in train_ray_tune)
                periogt_data = load_periogt_cache(dataset_name)
            else:
                print(f"\n[PerioGT] Precomputing graph data...")
                periogt_data = precompute_periogt_data(
                    dataset_name, data, ratio_encoding,
                    use_prompt=cfg.get("USE_PERIOGT_PROMPT", True)
                )
            struct_embeddings = None
        elif encoder_name == "llm":
            # LLM structural stream
            llm_model_key = cfg.get("MODEL_TYPE", "qwen3_4b_base")
            model_key = f"struct_llm_{llm_model_key}"
            if frozen_struct:
                if not skip_precompute:
                    print(f"\n[Embedding] Precomputing LLM embeddings for type 9 ({llm_model_key})...")
                    precompute_llm_embeddings(
                        dataset_name, llm_model_key, data,
                        batch_size=1
                    )
                struct_embeddings = None
            else:
                struct_embeddings = None
        else:
            # polybert
            if frozen_struct:
                if not skip_precompute:
                    print(f"\n[Embedding] Precomputing struct embeddings ({encoder_name})...")
                    precompute_struct_embeddings(
                        dataset_name, data,
                        encoder_name=encoder_name,
                        ratio_encoding_path=ratio_path,
                    )
                struct_embeddings = load_cached_struct_embeddings(dataset_name, encoder_name)
            else:
                struct_embeddings = None
    else:
        # ---- Semantic / Hybrid stream mode ----
        model_key = cfg["MODEL_TYPE"]
        enable_hybrid = cfg["ENABLE_HYBRID_ENCODING"]

        # Ensure ratio encoding exists (hybrid mode may need it)
        if enable_hybrid and not skip_precompute:
            if not os.path.exists(ratio_path):
                print(f"\n[Ratio Encoding] Scanning datasets...")
                scan_and_save_ratio_encoding([dataset_name], save_path=ratio_path)

        periogt_data = None
        if frozen:
            if not skip_precompute:
                print(f"\n[Embedding] Precomputing LLM embeddings ({model_key})...")
                precompute_llm_embeddings(
                    dataset_name, model_key, data,
                    batch_size=1
                )
                if enable_hybrid:
                    encoder_name = cfg.get("STRUCT_ENCODER", "polybert")
                    if encoder_name not in ("llm", "periogt"):
                        print(f"[Embedding] Precomputing struct embeddings ({encoder_name})...")
                        precompute_struct_embeddings(
                            dataset_name, data,
                            encoder_name=encoder_name,
                            ratio_encoding_path=ratio_path,
                        )

        # PerioGT hybrid stream precompute graph data
        if enable_hybrid:
            encoder_name = cfg.get("STRUCT_ENCODER", "polybert")
            if encoder_name == "periogt":
                if not os.path.exists(ratio_path):
                    scan_and_save_ratio_encoding([dataset_name], save_path=ratio_path)
                ratio_encoding = load_ratio_encoding(ratio_path)
                if skip_precompute:
                    # Ray trial: load from cache
                    periogt_data = load_periogt_cache(dataset_name)
                else:
                    print(f"\n[PerioGT] Precomputing graph data for hybrid mode...")
                    periogt_data = precompute_periogt_data(
                        dataset_name, data, ratio_encoding,
                        use_prompt=cfg.get("USE_PERIOGT_PROMPT", True)
                    )

        struct_embeddings = None
        if enable_hybrid:
            encoder_name = cfg.get("STRUCT_ENCODER", "polybert")
            if encoder_name == "periogt":
                pass  # PerioGT data already precomputed above
            elif encoder_name == "llm":
                # LLM structural stream frozen mode: build_dataloaders internally loads type 9 cache as struct embeddings, no need to pass separately
                pass
            elif frozen:
                struct_embeddings = load_cached_struct_embeddings(
                    dataset_name, cfg.get("STRUCT_ENCODER", "polybert")
                )
            # Non-frozen hybrid mode: no cache, real-time encoding (with gradient)

    # Inject scaler params into cfg for TensorBoard logging at original scale
    cfg["_SCALER_MEAN"] = float(scaler.mean_[0])
    cfg["_SCALER_STD"] = float(scaler.scale_[0])

    # Save full config immediately (including scaler info), record before training starts to prevent loss on interruption
    save_config_yaml(cfg, os.path.join(experiment_dir, "config.yaml"))

    # ---- Compute training set article ranges (for Article Range Hit Rate) ----
    train_article_ranges = {}
    if _is_article_aware(cfg):
        article_ids_all = extract_article_ids(data)
        for idx in split_indices["train"]:
            aid = article_ids_all[idx]
            val = float(targets_raw[idx])
            if aid not in train_article_ranges:
                train_article_ranges[aid] = (val, val)
            else:
                lo, hi = train_article_ranges[aid]
                train_article_ranges[aid] = (min(lo, val), max(hi, val))

    # ---- Multi-seed training ----
    seeds = cfg["SEEDS"]
    all_test_results = []
    all_preds = []
    all_targets = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"  Seed: {seed} | Dataset: {dataset_name}")
        print(f"{'='*60}")
        set_seed(seed)

        # Build DataLoader
        if struct_only:
            train_loader, val_loader, test_loader = build_struct_dataloaders(
                split_indices, targets_scaled, cfg,
                struct_embeddings=struct_embeddings, data=data,
                periogt_data=periogt_data, dataset_name=dataset_name,
                ratio_encoding_path=ratio_path,
            )
        else:
            train_loader, val_loader, test_loader = build_dataloaders(
                data, dataset_name, split_indices, targets_scaled,
                cfg, struct_embeddings, periogt_data=periogt_data,
                ratio_encoding_path=ratio_path,
            )

        # Build model
        if struct_only:
            model = StructureOnlyModel(cfg)
        else:
            model = PropertyPredictionModel(cfg)

        # Callbacks
        ckpt_path = os.path.join(experiment_dir, f"best_seed{seed}.ckpt")
        checkpoint_cb = ModelCheckpoint(
            dirpath=experiment_dir,
            filename=f"best_seed{seed}",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
        )
        early_stop_cb = EarlyStopping(
            monitor="val/loss",
            patience=cfg["PATIENCE"],
            mode="min",
        )

        # Logger
        tb_logger = TensorBoardLogger(
            save_dir=TB_LOG_DIR,
            name=f"{dataset_name}_{model_key}",
            version=f"seed{seed}",
        )

        # Trainer
        trainer_device_kwargs = _build_trainer_device_kwargs(
            cfg, context=f"dataset {dataset_name}"
        )
        trainer = pl.Trainer(
            max_epochs=cfg["EPOCHS"],
            accelerator=trainer_device_kwargs["accelerator"],
            devices=trainer_device_kwargs["devices"],
            strategy=trainer_device_kwargs["strategy"],
            gradient_clip_val=cfg["GRADIENT_CLIP_VAL"],
            accumulate_grad_batches=cfg.get("GRADIENT_ACCUMULATION_STEPS", 1),
            callbacks=[checkpoint_cb, early_stop_cb],
            logger=tb_logger,
            enable_progress_bar=True,
        )

        # Training
        trainer.fit(model, train_loader, val_loader)

        # Release GPU memory from training model to avoid OOM when load_from_checkpoint creates a second model
        del model, trainer
        torch.cuda.empty_cache()

        # Load best model weights for testing
        if struct_only:
            best_model = StructureOnlyModel.load_from_checkpoint(
                checkpoint_cb.best_model_path, cfg=cfg
            )
        else:
            best_model = PropertyPredictionModel.load_from_checkpoint(
                checkpoint_cb.best_model_path, cfg=cfg
            )

        # Save only trainable parameter weights
        weights_path = os.path.join(experiment_dir, f"weights_seed{seed}.pth")
        save_trainable_weights(best_model, weights_path)

        # Collect predictions (evaluate val set when skip_test=True, otherwise evaluate test set)
        best_model.eval()
        best_model.to("cuda" if trainer_device_kwargs["num_gpus"] > 0 else "cpu")

        eval_loader = val_loader if skip_test else test_loader
        metrics, preds_orig, targets_orig = evaluate_test_set(
            best_model, eval_loader, cfg, scaler,
            struct_only, frozen, experiment_dir, seed, dataset_name,
            save_visualizations=(not skip_test),
            train_article_ranges=train_article_ranges,
        )
        all_test_results.append(metrics)
        all_preds.append(preds_orig)
        all_targets.append(targets_orig)

        # Delete .ckpt files (only trainable-weight .pth is kept)
        if os.path.exists(checkpoint_cb.best_model_path):
            os.remove(checkpoint_cb.best_model_path)

        # Release GPU memory to avoid OOM during multi-seed training
        del best_model
        torch.cuda.empty_cache()

    # ---- Summary ----
    r2_vals = [m["R2"] for m in all_test_results]
    rmse_vals = [m["RMSE"] for m in all_test_results]
    mae_vals = [m["MAE"] for m in all_test_results]

    summary = {
        "dataset": dataset_name,
        "Mean_R2": f"{np.mean(r2_vals):.4f} ± {np.std(r2_vals):.4f}",
        "Mean_RMSE": f"{np.mean(rmse_vals):.4f} ± {np.std(rmse_vals):.4f}",
        "Mean_MAE": f"{np.mean(mae_vals):.4f} ± {np.std(mae_vals):.4f}",
    }

    # CSV
    csv_data = all_test_results.copy()
    summary_row = {
        "seed": "Mean±Std",
        "R2": summary["Mean_R2"],
        "RMSE": summary["Mean_RMSE"],
        "MAE": summary["Mean_MAE"],
    }
    # Article-aware metric summary
    for article_metric in ["article_range_hit_rate", "article_nars"]:
        vals = [m[article_metric] for m in all_test_results if article_metric in m]
        if vals:
            summary_row[article_metric] = f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"
    csv_data.append(summary_row)
    save_csv_summary(csv_data, os.path.join(experiment_dir, "results.csv"))

    print(f"\n[Summary] {dataset_name}: {summary}")
    return summary


# ======================== Ray Tune Grid Search ========================

def train_ray_tune(fixed_cfg: Dict[str, Any], search_space: Dict,
                   dataset_name: str, grid_experiment_dir: str,
                   grid_param_names: List[str] = None,
                   linked_unpacking: Dict = None,
                   yaml_path: str = None):
    """
    Ray Tune grid search training.
    Supports YAML configuration (including linked_grid linked parameters).
    GPU fraction and max concurrency are read from fixed_cfg (RAY_GPU_FRACTION / MAX_CONCURRENT_TRIALS).
    """
    from ray import tune

    if grid_param_names is None:
        grid_param_names = list(search_space.keys())
    if linked_unpacking is None:
        linked_unpacking = {}

    gpu_fraction = fixed_cfg.get("RAY_GPU_FRACTION", 1.0)
    max_concurrent = fixed_cfg.get("MAX_CONCURRENT_TRIALS", 4)
    cpu_per_trial = fixed_cfg.get("RAY_CPU_PER_TRIAL", 2)
    max_failures = fixed_cfg.get("RAY_MAX_FAILURES", 3)

    # ---- Save grid config copy immediately (record at experiment start, don't wait for training to finish) ----
    ensure_dir(grid_experiment_dir)
    if yaml_path and os.path.exists(yaml_path):
        import shutil as _shutil
        _shutil.copy2(yaml_path, os.path.join(grid_experiment_dir, "grid_config.yaml"))
    else:
        save_config_yaml(
            {k: v for k, v in fixed_cfg.items() if not k.startswith("_")},
            os.path.join(grid_experiment_dir, "grid_config.yaml")
        )

    # ---- Precompute embeddings (before all trials) ----
    struct_only = _is_struct_only(fixed_cfg)
    data = load_dataset(dataset_name)

    # Ensure ratio encoding exists (saved in grid experiment directory)
    ratio_path = os.path.join(grid_experiment_dir, "ratio_encoding.json")
    if not os.path.exists(ratio_path):
        print(f"\n[Ratio Encoding] Scanning datasets...")
        scan_and_save_ratio_encoding([dataset_name], save_path=ratio_path)
    fixed_cfg["_RATIO_ENCODING_PATH"] = ratio_path

    # Determine if precomputation might be needed (considering FROZEN_BACKBONE changes via linked_grid)
    might_be_frozen = fixed_cfg.get("FROZEN_BACKBONE", True)
    if not might_be_frozen:
        for link_key, params in linked_unpacking.items():
            if "FROZEN_BACKBONE" in params:
                might_be_frozen = True
                break

    # Extract all possible MODEL_TYPE and STRUCT_ENCODER values from search space
    all_model_types = _extract_grid_values(
        search_space, "MODEL_TYPE", linked_unpacking,
        fixed_cfg.get("MODEL_TYPE", "qwen3_4b_base")
    )
    all_struct_encoders = _extract_grid_values(
        search_space, "STRUCT_ENCODER", linked_unpacking,
        fixed_cfg.get("STRUCT_ENCODER", "polybert")
    )

    if struct_only:
        for encoder_name in all_struct_encoders:
            if encoder_name == "periogt":
                # Precompute PerioGT graph data and cache (with prompt, superset)
                ratio_encoding = load_ratio_encoding(ratio_path)
                print(f"\n[Pre-compute] PerioGT graph data for {dataset_name}")
                precompute_periogt_data(
                    dataset_name, data, ratio_encoding,
                    use_prompt=True, use_cache=True,
                )
            elif encoder_name == "llm":
                if might_be_frozen:
                    for llm_model_key in all_model_types:
                        print(f"\n[Pre-compute] LLM type 9 embeddings for {dataset_name} ({llm_model_key})")
                        precompute_llm_embeddings(
                            dataset_name, llm_model_key, data,
                            batch_size=1
                        )
            elif might_be_frozen:
                print(f"\n[Pre-compute] Struct embeddings for {dataset_name} ({encoder_name})")
                precompute_struct_embeddings(
                    dataset_name, data,
                    encoder_name=encoder_name,
                    ratio_encoding_path=ratio_path,
                )
    elif might_be_frozen:
        for model_key in all_model_types:
            print(f"\n[Pre-compute] LLM embeddings for {dataset_name} ({model_key})")
            precompute_llm_embeddings(
                dataset_name, model_key, data,
                batch_size=1
            )
        if fixed_cfg.get("ENABLE_HYBRID_ENCODING", False):
            for encoder_name in all_struct_encoders:
                if encoder_name == "periogt":
                    ratio_encoding = load_ratio_encoding(ratio_path)
                    print(f"\n[Pre-compute] PerioGT graph data for {dataset_name} (hybrid)")
                    precompute_periogt_data(
                        dataset_name, data, ratio_encoding,
                        use_prompt=True, use_cache=True,
                    )
                elif encoder_name != "llm":
                    print(f"[Pre-compute] {encoder_name} embeddings for {dataset_name}")
                    precompute_struct_embeddings(
                        dataset_name, data,
                        encoder_name=encoder_name,
                        ratio_encoding_path=ratio_path,
                    )

    trial_counter_file = os.path.join(grid_experiment_dir, ".trial_counter")

    def _get_next_trial_id():
        """Thread-safe retrieval of the next trial ID"""
        import filelock
        lock = filelock.FileLock(trial_counter_file + ".lock")
        with lock:
            if os.path.exists(trial_counter_file):
                with open(trial_counter_file, "r") as f:
                    counter = int(f.read().strip())
            else:
                counter = 0
            counter += 1
            with open(trial_counter_file, "w") as f:
                f.write(str(counter))
        return counter

    def trial_train_fn(trial_config):
        """Execute training inside a Ray Tune trial"""
        # Limit CPU threads per trial to prevent resource contention between processes
        os.environ["OMP_NUM_THREADS"] = str(cpu_per_trial)
        os.environ["MKL_NUM_THREADS"] = str(cpu_per_trial)
        os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_per_trial)
        torch.set_num_threads(cpu_per_trial)

        # Merge config
        merged_cfg = {**fixed_cfg}
        merged_cfg.update(trial_config)

        # Expand linked parameter groups
        for link_key, param_names in linked_unpacking.items():
            if link_key in merged_cfg:
                values = merged_cfg.pop(link_key)
                for param_name, val in zip(param_names, values):
                    merged_cfg[param_name] = val

        trial_id = _get_next_trial_id()
        trial_dir = os.path.join(grid_experiment_dir, str(trial_id))
        ensure_dir(trial_dir)

        summary = train_single_config(merged_cfg, dataset_name, trial_dir, skip_precompute=True, skip_test=True)

        # Record trial results
        tune.report({
            "mean_r2": float(summary["Mean_R2"].split(" \u00b1")[0]),
            "mean_rmse": float(summary["Mean_RMSE"].split(" \u00b1")[0]),
            "mean_mae": float(summary["Mean_MAE"].split(" \u00b1")[0]),
            "trial_id": trial_id,
        })

    tuner = tune.Tuner(
        tune.with_resources(
            trial_train_fn,
            resources={"gpu": gpu_fraction, "cpu": cpu_per_trial},
        ),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=1,
            max_concurrent_trials=max_concurrent,
        ),
        run_config=tune.RunConfig(
            storage_path=os.path.join(grid_experiment_dir, "ray_results"),
            name=f"grid_{dataset_name}",
            failure_config=tune.FailureConfig(max_failures=max_failures),
        ),
    )
    results = tuner.fit()

    # ---- Clean up incomplete directories from failed trials ----
    trial_dirs_all = [
        d for d in os.listdir(grid_experiment_dir)
        if d.isdigit() and os.path.isdir(os.path.join(grid_experiment_dir, d))
    ]
    for td in trial_dirs_all:
        td_path = os.path.join(grid_experiment_dir, td)
        csv_path = os.path.join(td_path, "results.csv")
        if not os.path.exists(csv_path):
            import shutil
            print(f"[Cleanup] Removing incomplete trial directory: {td_path}")
            shutil.rmtree(td_path, ignore_errors=True)

    # ---- Summarize all trials (dynamic column names) ----
    all_trial_results = []
    trial_dirs = sorted(
        [d for d in os.listdir(grid_experiment_dir)
         if d.isdigit() and os.path.isdir(os.path.join(grid_experiment_dir, d))],
        key=int
    )
    for td in trial_dirs:
        td_path = os.path.join(grid_experiment_dir, td)
        csv_path = os.path.join(td_path, "results.csv")
        yaml_cfg_path = os.path.join(td_path, "config.yaml")
        if os.path.exists(csv_path) and os.path.exists(yaml_cfg_path):
            import yaml as yaml_mod
            with open(yaml_cfg_path, "r", encoding="utf-8") as f:
                trial_cfg = yaml_mod.safe_load(f)
            df = pd.read_csv(csv_path)
            last_row = df.iloc[-1]

            row = {"Trial": int(td)}
            for param_name in grid_param_names:
                row[param_name] = str(trial_cfg.get(param_name, ""))

            # Parse "mean ± std" into separate columns
            for metric in ["R2", "RMSE", "MAE"]:
                raw = str(last_row.get(metric, ""))
                parts = raw.split("±")
                if len(parts) == 2:
                    row[f"Mean_{metric}"] = parts[0].strip()
                    row[f"Std_{metric}"] = parts[1].strip()
                else:
                    row[f"Mean_{metric}"] = raw.strip()
                    row[f"Std_{metric}"] = ""
            # Article-aware metrics
            for article_metric in ["article_range_hit_rate", "article_nars"]:
                raw = str(last_row.get(article_metric, ""))
                if raw and raw != "nan":
                    parts = raw.split("±")
                    if len(parts) == 2:
                        row[f"Mean_{article_metric}"] = parts[0].strip()
                        row[f"Std_{article_metric}"] = parts[1].strip()
                    else:
                        row[f"Mean_{article_metric}"] = raw.strip()
                        row[f"Std_{article_metric}"] = ""
            all_trial_results.append(row)

    summary_df = pd.DataFrame(all_trial_results)
    summary_df.to_csv(
        os.path.join(grid_experiment_dir, "grid_summary.csv"),
        index=False, encoding="utf-8-sig"
    )

    print(f"\n[Grid Search Complete] Results saved to {grid_experiment_dir}")


# ======================== Main Entry ========================

def main():
    from datetime import datetime

    if GRID_YAML:
        # ---- YAML Grid Search Mode ----
        yaml_full_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), GRID_YAML
        )
        base_cfg, search_space, grid_param_names, linked_unpacking = \
            load_grid_yaml(yaml_full_path)

        if not search_space:
            # No grid params in YAML, treat as single-config training
            for dataset_name in TRAIN_DATASETS:
                print(f"\n{'#'*70}")
                print(f"  Dataset: {dataset_name}")
                print(f"{'#'*70}")
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                if _is_struct_only(base_cfg):
                    model_label = f"struct_{base_cfg.get('STRUCT_ENCODER', 'polybert')}"
                else:
                    model_label = base_cfg.get('MODEL_TYPE', 'model')
                exp_dir = os.path.join(RESULT_DIR, f"{dataset_name}_{model_label}_{ts}")
                ensure_dir(exp_dir)
                train_single_config(base_cfg, dataset_name, exp_dir, skip_test=True)
        else:
            # YAML Grid Search
            for dataset_name in TRAIN_DATASETS:
                print(f"\n{'#'*70}")
                print(f"  Dataset: {dataset_name} (YAML Grid Search)")
                print(f"{'#'*70}")
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                grid_dir = os.path.join(RESULT_DIR, f"{dataset_name}_grid_{ts}")
                ensure_dir(grid_dir)
                train_ray_tune(
                    copy.deepcopy(base_cfg), search_space, dataset_name, grid_dir,
                    grid_param_names=grid_param_names,
                    linked_unpacking=linked_unpacking,
                    yaml_path=yaml_full_path,
                )
    else:
        # ---- Single Config Module Mode ----
        cfg = load_config_as_dict(CONFIG_MODULE)

        for dataset_name in TRAIN_DATASETS:
            print(f"\n{'#'*70}")
            print(f"  Dataset: {dataset_name}")
            print(f"{'#'*70}")

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            if _is_struct_only(cfg):
                model_label = f"struct_{cfg.get('STRUCT_ENCODER', 'polybert')}"
            else:
                model_label = cfg.get('MODEL_TYPE', 'model')
            exp_dir = os.path.join(
                RESULT_DIR,
                f"{dataset_name}_{model_label}_{ts}"
            )
            ensure_dir(exp_dir)
            train_single_config(cfg, dataset_name, exp_dir)


if __name__ == "__main__":
    main()
