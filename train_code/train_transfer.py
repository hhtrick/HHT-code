"""
train_transfer.py — Joint training script: merge train/val sets from multiple datasets,
early stopping on the merged validation set,
then compute test metrics separately for each dataset.
Parameters and functionality are largely the same as train.py, except that multiple datasets
in TRAIN_DATASETS are merged for joint training.
"""
import os
import copy
import json
import importlib
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import ConcatDataset, DataLoader
from typing import Dict, Any, List

from config.path import (
    RESULT_DIR, TB_LOG_DIR,
    ensure_dir,
)
from utils import (
    load_dataset, load_split_indices, extract_targets,
    fit_scaler, transform_targets, inverse_transform,
    precompute_llm_embeddings, precompute_struct_embeddings,
    load_cached_struct_embeddings, load_cached_embeddings,
    set_seed, compute_metrics, plot_parity,
    save_config_yaml, save_csv_summary,
    save_trainable_weights,
    scan_and_save_ratio_encoding, load_ratio_encoding,
    precompute_periogt_data, load_periogt_cache, periogt_cache_exists,
    evaluate_test_set,
    extract_article_ids, _is_article_aware,
    CachedEmbeddingDataset, LiveInferenceDataset,
    StructOnlyDataset, LiveStructDataset,
    LLMStructCachedDataset, LLMStructLiveDataset,
    PerioGTDataset,
    HybridPerioGTCachedDataset, HybridPerioGTLiveDataset,
    HybridLLMStructCachedDataset,
    cached_collate_fn, live_collate_fn,
    struct_collate_fn, live_struct_collate_fn,
    periogt_collate_fn,
    hybrid_periogt_cached_collate_fn, hybrid_periogt_live_collate_fn,
    hybrid_llm_struct_cached_collate_fn,
)
from models.base_model import PropertyPredictionModel, StructureOnlyModel
from train import (
    load_config_as_dict,
    _is_struct_only,
    _extract_grid_values,
    load_grid_yaml,
    _build_trainer_device_kwargs,
)

# ======================== Training Parameter Settings ========================
# Joint training dataset list
TRAIN_DATASETS = ["Tg","Tm","E","UTS","eps","n"]

# Single-config training module: "config.config_article" | "config.config_stru_article"
CONFIG_MODULE = "config.config_article"

# YAML grid search config file path (set to enable YAML grid search)
# Set to None to use CONFIG_MODULE for single-config training
# Example: "config/grid/exp7_transfer.yaml"
GRID_YAML = None
# ==============================================================


def train_transfer(cfg: Dict[str, Any], dataset_names: List[str],
                   experiment_dir: str, skip_test: bool = False):
    """
    Joint training: merge train/val sets from multiple datasets,
    early stopping based on merged validation set RMSE,
    then evaluate metrics on each dataset separately.

    Args:
        skip_test: If True, skip test set evaluation, only evaluate on validation set
                   (used for grid search to avoid data leakage).
    """
    torch.set_float32_matmul_precision(cfg["FLOAT32_MATMUL_PRECISION"])
    struct_only = _is_struct_only(cfg)
    ensure_dir(experiment_dir)
    # ratio_encoding.json is saved in the experiment directory (alongside model weights, results, etc.)
    ratio_path = os.path.join(experiment_dir, "ratio_encoding.json")
    cfg["_RATIO_ENCODING_PATH"] = ratio_path

    # ---- Load all datasets ----
    all_data = {}
    all_splits = {}
    all_targets_raw = {}
    all_scalers = {}
    all_targets_scaled = {}

    split_pkl = cfg["SPLIT_PKL"]

    for ds_name in dataset_names:
        data = load_dataset(ds_name)
        split = load_split_indices(ds_name, split_pkl)
        targets_raw = extract_targets(data, ds_name)
        all_data[ds_name] = data
        all_splits[ds_name] = split
        all_targets_raw[ds_name] = targets_raw

    # Fit a separate scaler for each dataset
    for ds_name in dataset_names:
        scaler = fit_scaler(all_targets_raw[ds_name], all_splits[ds_name]["train"])
        all_scalers[ds_name] = scaler
        all_targets_scaled[ds_name] = transform_targets(scaler, all_targets_raw[ds_name])

    # Save scalers
    import pickle as pkl
    for ds_name in dataset_names:
        scaler_path = os.path.join(experiment_dir, f"scaler_{ds_name}.pkl")
        with open(scaler_path, "wb") as f:
            pkl.dump(all_scalers[ds_name], f)

    # ---- Compute training set article ranges (for Article Range Hit Rate) ----
    all_train_article_ranges = {}
    if _is_article_aware(cfg):
        for ds_name in dataset_names:
            article_ids_all = extract_article_ids(all_data[ds_name])
            ranges = {}
            for idx in all_splits[ds_name]["train"]:
                aid = article_ids_all[idx]
                val = float(all_targets_raw[ds_name][idx])
                if aid not in ranges:
                    ranges[aid] = (val, val)
                else:
                    lo, hi = ranges[aid]
                    ranges[aid] = (min(lo, val), max(hi, val))
            all_train_article_ranges[ds_name] = ranges

    # ---- Precompute embeddings ----
    if not os.path.exists(ratio_path):
        print(f"\n[Ratio Encoding] Scanning datasets...")
        scan_and_save_ratio_encoding(dataset_names, save_path=ratio_path)

    all_periogt_data = {}
    all_struct_embeddings = {}

    for ds_name in dataset_names:
        data = all_data[ds_name]

        if struct_only:
            encoder_name = cfg.get("STRUCT_ENCODER", "polybert")
            frozen_struct = cfg.get("FROZEN_BACKBONE", True)

            if encoder_name == "periogt":
                ratio_encoding = load_ratio_encoding(ratio_path)
                if periogt_cache_exists(ds_name):
                    all_periogt_data[ds_name] = load_periogt_cache(ds_name)
                else:
                    print(f"\n[PerioGT] Precomputing graph data for {ds_name}...")
                    all_periogt_data[ds_name] = precompute_periogt_data(
                        ds_name, data, ratio_encoding,
                        use_prompt=cfg.get("USE_PERIOGT_PROMPT", True)
                    )
            elif encoder_name == "llm":
                if frozen_struct:
                    llm_key = cfg.get("MODEL_TYPE", "qwen3_4b_base")
                    precompute_llm_embeddings(ds_name, llm_key, data,
                                              batch_size=1)
            else:
                if frozen_struct:
                    precompute_struct_embeddings(ds_name, data, encoder_name=encoder_name,
                                                 ratio_encoding_path=ratio_path)
                    all_struct_embeddings[ds_name] = load_cached_struct_embeddings(ds_name, encoder_name)
        else:
            frozen = cfg["FROZEN_BACKBONE"]
            model_key = cfg["MODEL_TYPE"]
            if frozen:
                precompute_llm_embeddings(ds_name, model_key, data,
                                          batch_size=1)
                if cfg.get("ENABLE_HYBRID_ENCODING", False):
                    enc_name = cfg.get("STRUCT_ENCODER", "polybert")
                    if enc_name not in ("llm", "periogt"):
                        precompute_struct_embeddings(ds_name, data, encoder_name=enc_name,
                                                     ratio_encoding_path=ratio_path)
            # PerioGT hybrid stream: try loading from cache first
            if cfg.get("ENABLE_HYBRID_ENCODING", False) and cfg.get("STRUCT_ENCODER") == "periogt":
                ratio_encoding = load_ratio_encoding(ratio_path)
                if periogt_cache_exists(ds_name):
                    all_periogt_data[ds_name] = load_periogt_cache(ds_name)
                else:
                    print(f"\n[PerioGT] Precomputing graph data for {ds_name} (hybrid)...")
                    all_periogt_data[ds_name] = precompute_periogt_data(
                        ds_name, data, ratio_encoding,
                        use_prompt=cfg.get("USE_PERIOGT_PROMPT", True)
                    )

    # During joint training, each dataset has its own scaler, making it infeasible to restore true-scale per-sample during train/val;
    # Setting to 0/1 makes TensorBoard display normalized-space RMSE, maintaining internal consistency.
    # Test set metrics for each dataset are restored separately via all_scalers[ds_name] after training (see test loop below).
    cfg["_SCALER_MEAN"] = 0.0
    cfg["_SCALER_STD"] = 1.0
    # ---- Multi-seed training ----
    seeds = cfg["SEEDS"]
    all_results = {ds_name: [] for ds_name in dataset_names}

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"  Seed: {seed} | Transfer Training: {dataset_names}")
        print(f"{'='*60}")
        set_seed(seed)

        # Build merged train/val DataLoaders, plus a separate eval DataLoader for each dataset
        # When skip_test=True, use validation set for evaluation; otherwise use test set
        train_datasets_list = []
        val_datasets_list = []
        eval_loaders = {}  # Eval DataLoader for each dataset (test set or validation set)

        batch_size = cfg["BATCH_SIZE"]
        num_workers = cfg["NUM_WORKERS"]

        for ds_name in dataset_names:
            split = all_splits[ds_name]
            targets_sc = all_targets_scaled[ds_name]
            data = all_data[ds_name]

            if struct_only:
                encoder_name = cfg.get("STRUCT_ENCODER", "polybert")
                frozen_struct = cfg.get("FROZEN_BACKBONE", True)

                if encoder_name == "periogt":
                    pgt_data = all_periogt_data[ds_name]
                    train_ds = PerioGTDataset(split["train"], pgt_data, targets_sc)
                    val_ds = PerioGTDataset(split["val"], pgt_data, targets_sc)
                    test_ds = PerioGTDataset(split["test"], pgt_data, targets_sc)
                    collate = periogt_collate_fn
                elif encoder_name == "llm" and frozen_struct:
                    llm_key = cfg.get("MODEL_TYPE", "qwen3_4b_base")
                    cached_type9 = load_cached_embeddings(ds_name, llm_key, 9)
                    train_ds = LLMStructCachedDataset(split["train"], cached_type9, targets_sc)
                    val_ds = LLMStructCachedDataset(split["val"], cached_type9, targets_sc)
                    test_ds = LLMStructCachedDataset(split["test"], cached_type9, targets_sc)
                    collate = cached_collate_fn
                elif encoder_name == "llm" and not frozen_struct:
                    train_ds = LLMStructLiveDataset(split["train"], data, targets_sc, ds_name)
                    val_ds = LLMStructLiveDataset(split["val"], data, targets_sc, ds_name)
                    test_ds = LLMStructLiveDataset(split["test"], data, targets_sc, ds_name)
                    collate = live_collate_fn
                elif frozen_struct:
                    se = all_struct_embeddings[ds_name]
                    train_ds = StructOnlyDataset(split["train"], se, targets_sc)
                    val_ds = StructOnlyDataset(split["val"], se, targets_sc)
                    test_ds = StructOnlyDataset(split["test"], se, targets_sc)
                    collate = struct_collate_fn
                else:
                    ratio_enc = None
                    if os.path.exists(ratio_path):
                        ratio_enc = load_ratio_encoding(ratio_path)
                    train_ds = LiveStructDataset(split["train"], data, targets_sc, ratio_encoding=ratio_enc)
                    val_ds = LiveStructDataset(split["val"], data, targets_sc, ratio_encoding=ratio_enc)
                    test_ds = LiveStructDataset(split["test"], data, targets_sc, ratio_encoding=ratio_enc)
                    collate = live_struct_collate_fn
            else:
                frozen = cfg["FROZEN_BACKBONE"]
                model_key = cfg["MODEL_TYPE"]
                input_content = cfg["INPUT_CONTENT"]
                eval_input_content = cfg.get("EVAL_INPUT_CONTENT", input_content)
                enable_hybrid = cfg.get("ENABLE_HYBRID_ENCODING", False)
                enc_name = cfg.get("STRUCT_ENCODER", "polybert") if enable_hybrid else None

                if enable_hybrid and enc_name == "periogt" and ds_name in all_periogt_data:
                    # Hybrid stream + PerioGT
                    pgt_data = all_periogt_data[ds_name]
                    if frozen:
                        all_cached = {}
                        all_types_needed = set(input_content) | set(eval_input_content)
                        for t in all_types_needed:
                            all_cached[t] = load_cached_embeddings(ds_name, model_key, t)
                        train_ds = HybridPerioGTCachedDataset(
                            split["train"], all_cached, targets_sc, input_content, pgt_data)
                        val_ds = HybridPerioGTCachedDataset(
                            split["val"], all_cached, targets_sc, eval_input_content, pgt_data)
                        test_ds = HybridPerioGTCachedDataset(
                            split["test"], all_cached, targets_sc, eval_input_content, pgt_data)
                        collate = hybrid_periogt_cached_collate_fn
                    else:
                        train_ds = HybridPerioGTLiveDataset(
                            split["train"], data, targets_sc, ds_name, input_content, pgt_data)
                        val_ds = HybridPerioGTLiveDataset(
                            split["val"], data, targets_sc, ds_name, eval_input_content, pgt_data)
                        test_ds = HybridPerioGTLiveDataset(
                            split["test"], data, targets_sc, ds_name, eval_input_content, pgt_data)
                        collate = hybrid_periogt_live_collate_fn
                elif frozen:
                    all_cached = {}
                    all_types_needed = set(input_content) | set(eval_input_content)
                    for t in all_types_needed:
                        all_cached[t] = load_cached_embeddings(ds_name, model_key, t)
                    if enable_hybrid and enc_name == "llm":
                        # Hybrid stream LLM struct flow: use type 9 hidden states as struct encoder input
                        cached_struct = load_cached_embeddings(ds_name, model_key, 9)
                        train_ds = HybridLLMStructCachedDataset(split["train"], all_cached,
                                                                targets_sc, input_content, cached_struct)
                        val_ds = HybridLLMStructCachedDataset(split["val"], all_cached,
                                                              targets_sc, eval_input_content, cached_struct)
                        test_ds = HybridLLMStructCachedDataset(split["test"], all_cached,
                                                               targets_sc, eval_input_content, cached_struct)
                        collate = hybrid_llm_struct_cached_collate_fn
                    else:
                        se = None
                        if enable_hybrid and enc_name is not None:
                            se = load_cached_struct_embeddings(ds_name, enc_name)
                        train_ds = CachedEmbeddingDataset(split["train"], all_cached,
                                                          targets_sc, input_content, se)
                        val_ds = CachedEmbeddingDataset(split["val"], all_cached,
                                                        targets_sc, eval_input_content, se)
                        test_ds = CachedEmbeddingDataset(split["test"], all_cached,
                                                         targets_sc, eval_input_content, se)
                        collate = cached_collate_fn
                else:
                    _enc = cfg.get("STRUCT_ENCODER", "polybert") if enable_hybrid else None
                    need_struct_text = (enable_hybrid and _enc == "llm")
                    need_smiles = (enable_hybrid and _enc not in ("llm", "periogt", None))
                    ratio_enc = None
                    if need_smiles and os.path.exists(ratio_path):
                        ratio_enc = load_ratio_encoding(ratio_path)
                    train_ds = LiveInferenceDataset(split["train"], data, targets_sc,
                                                    ds_name, input_content,
                                                    need_smiles=need_smiles,
                                                    ratio_encoding=ratio_enc,
                                                    need_struct_text=need_struct_text)
                    val_ds = LiveInferenceDataset(split["val"], data, targets_sc,
                                                  ds_name, eval_input_content,
                                                  need_smiles=need_smiles,
                                                  ratio_encoding=ratio_enc,
                                                  need_struct_text=need_struct_text)
                    test_ds = LiveInferenceDataset(split["test"], data, targets_sc,
                                                   ds_name, eval_input_content,
                                                   need_smiles=need_smiles,
                                                   ratio_encoding=ratio_enc,
                                                   need_struct_text=need_struct_text)
                    collate = live_collate_fn

            train_datasets_list.append(train_ds)
            val_datasets_list.append(val_ds)
            # skip_test=True: evaluate on val set; skip_test=False: evaluate on test set
            eval_ds = val_ds if skip_test else test_ds
            eval_loaders[ds_name] = DataLoader(
                eval_ds, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, collate_fn=collate, pin_memory=True,
            )

        # Merge train/val
        combined_train = ConcatDataset(train_datasets_list)
        combined_val = ConcatDataset(val_datasets_list)

        # All datasets share the same collate function within the same mode
        train_loader = DataLoader(
            combined_train, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, collate_fn=collate, pin_memory=True,
        )
        val_loader = DataLoader(
            combined_val, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=collate, pin_memory=True,
        )

        # Build model
        if struct_only:
            model = StructureOnlyModel(cfg)
        else:
            model = PropertyPredictionModel(cfg)

        # Callbacks
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

        tb_logger = TensorBoardLogger(
            save_dir=TB_LOG_DIR,
            name=f"transfer_{'_'.join(dataset_names)}",
            version=f"seed{seed}",
        )

        trainer_device_kwargs = _build_trainer_device_kwargs(
            cfg, context=f"transfer datasets {','.join(dataset_names)}"
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

        trainer.fit(model, train_loader, val_loader)

        # Release GPU memory from training model to avoid OOM when load_from_checkpoint creates a second model
        del model, trainer
        torch.cuda.empty_cache()

        # Load best model
        if struct_only:
            best_model = StructureOnlyModel.load_from_checkpoint(
                checkpoint_cb.best_model_path, cfg=cfg
            )
        else:
            best_model = PropertyPredictionModel.load_from_checkpoint(
                checkpoint_cb.best_model_path, cfg=cfg
            )

        weights_path = os.path.join(experiment_dir, f"weights_seed{seed}.pth")
        save_trainable_weights(best_model, weights_path)

        # Evaluate each dataset separately
        best_model.eval()
        best_model.to("cuda" if trainer_device_kwargs["num_gpus"] > 0 else "cpu")
        frozen = cfg.get("FROZEN_BACKBONE", True)

        for ds_name in dataset_names:
            scaler = all_scalers[ds_name]
            metrics, preds_orig, targets_orig = evaluate_test_set(
                best_model, eval_loaders[ds_name], cfg, scaler,
                struct_only, frozen, experiment_dir, seed, ds_name,
                label_suffix=" (Transfer)", file_suffix=f"_{ds_name}",
                save_visualizations=(not skip_test),
                train_article_ranges=all_train_article_ranges.get(ds_name, {}),
            )
            metrics["dataset"] = ds_name
            all_results[ds_name].append(metrics)
            print(f"  [{ds_name}]", end="")

        if os.path.exists(checkpoint_cb.best_model_path):
            os.remove(checkpoint_cb.best_model_path)

        del best_model
        torch.cuda.empty_cache()

    # ---- Summary ----
    for ds_name in dataset_names:
        results = all_results[ds_name]
        r2_vals = [m["R2"] for m in results]
        rmse_vals = [m["RMSE"] for m in results]
        mae_vals = [m["MAE"] for m in results]
        summary = {
            "dataset": ds_name,
            "Mean_R2": f"{np.mean(r2_vals):.4f} ± {np.std(r2_vals):.4f}",
            "Mean_RMSE": f"{np.mean(rmse_vals):.4f} ± {np.std(rmse_vals):.4f}",
            "Mean_MAE": f"{np.mean(mae_vals):.4f} ± {np.std(mae_vals):.4f}",
        }
        csv_data = results.copy()
        summary_row = {
            "seed": "Mean±Std", "dataset": ds_name,
            "R2": summary["Mean_R2"],
            "RMSE": summary["Mean_RMSE"],
            "MAE": summary["Mean_MAE"],
        }
        # Article-aware metric summary
        for article_metric in ["article_range_hit_rate", "article_nars"]:
            vals = [m[article_metric] for m in results if article_metric in m]
            if vals:
                summary_row[article_metric] = f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"
        csv_data.append(summary_row)
        save_csv_summary(csv_data, os.path.join(experiment_dir, f"results_{ds_name}.csv"))
        print(f"\n[Summary] {ds_name}: {summary}")

    save_config_yaml(cfg, os.path.join(experiment_dir, "config.yaml"))


def train_transfer_ray_tune(fixed_cfg: Dict[str, Any], search_space: Dict,
                            dataset_names: List[str], grid_experiment_dir: str,
                            grid_param_names: List[str] = None,
                            linked_unpacking: Dict = None,
                            yaml_path: str = None):
    """
    Ray Tune grid search + joint training.
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

    struct_only = _is_struct_only(fixed_cfg)

    # ratio_encoding saved in grid experiment directory
    ratio_path = os.path.join(grid_experiment_dir, "ratio_encoding.json")
    if not os.path.exists(ratio_path):
        print(f"\n[Ratio Encoding] Scanning datasets...")
        scan_and_save_ratio_encoding(dataset_names, save_path=ratio_path)
    fixed_cfg["_RATIO_ENCODING_PATH"] = ratio_path

    # Determine if precomputation might be needed
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

    # Precompute embeddings for all datasets (before all trials, to avoid repeated large model loading)
    for ds_name in dataset_names:
        data = load_dataset(ds_name)
        if struct_only:
            for encoder_name in all_struct_encoders:
                if encoder_name == "periogt":
                    ratio_encoding = load_ratio_encoding(ratio_path)
                    print(f"\n[Pre-compute] PerioGT graph data for {ds_name}")
                    precompute_periogt_data(
                        ds_name, data, ratio_encoding,
                        use_prompt=True, use_cache=True,
                    )
                elif encoder_name == "llm":
                    if might_be_frozen:
                        for llm_model_key in all_model_types:
                            print(f"\n[Pre-compute] LLM type 9 embeddings for {ds_name} ({llm_model_key})")
                            precompute_llm_embeddings(
                                ds_name, llm_model_key, data,
                                batch_size=1
                            )
                elif might_be_frozen:
                    print(f"\n[Pre-compute] Struct embeddings for {ds_name} ({encoder_name})")
                    precompute_struct_embeddings(
                        ds_name, data, encoder_name=encoder_name,
                        ratio_encoding_path=ratio_path,
                    )
        elif might_be_frozen:
            for model_key in all_model_types:
                print(f"\n[Pre-compute] LLM embeddings for {ds_name} ({model_key})")
                precompute_llm_embeddings(
                    ds_name, model_key, data,
                    batch_size=1
                )
            if fixed_cfg.get("ENABLE_HYBRID_ENCODING", False):
                for encoder_name in all_struct_encoders:
                    if encoder_name == "periogt":
                        ratio_encoding = load_ratio_encoding(ratio_path)
                        print(f"\n[Pre-compute] PerioGT graph data for {ds_name} (hybrid)")
                        precompute_periogt_data(
                            ds_name, data, ratio_encoding,
                            use_prompt=True, use_cache=True,
                        )
                    elif encoder_name != "llm":
                        print(f"[Pre-compute] {encoder_name} embeddings for {ds_name}")
                        precompute_struct_embeddings(
                            ds_name, data, encoder_name=encoder_name,
                            ratio_encoding_path=ratio_path,
                        )

    trial_counter_file = os.path.join(grid_experiment_dir, ".trial_counter")
    ensure_dir(grid_experiment_dir)

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
        """Execute joint training inside a Ray Tune trial"""
        # Limit CPU threads per trial to prevent resource contention between processes
        os.environ["OMP_NUM_THREADS"] = str(cpu_per_trial)
        os.environ["MKL_NUM_THREADS"] = str(cpu_per_trial)
        os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_per_trial)
        torch.set_num_threads(cpu_per_trial)

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

        # Joint training (merge all datasets), skip test set evaluation in grid search mode
        train_transfer(merged_cfg, dataset_names, trial_dir, skip_test=True)

        # Read CSV results from each dataset and report to Ray Tune
        all_rmse, all_r2, all_mae = [], [], []
        for ds_name in dataset_names:
            csv_path = os.path.join(trial_dir, f"results_{ds_name}.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                last_row = df.iloc[-1]
                try:
                    all_rmse.append(float(str(last_row.get("RMSE", "0")).split("\u00b1")[0].strip()))
                    all_r2.append(float(str(last_row.get("R2", "0")).split("\u00b1")[0].strip()))
                    all_mae.append(float(str(last_row.get("MAE", "0")).split("\u00b1")[0].strip()))
                except (ValueError, AttributeError):
                    pass

        tune.report({
            "mean_r2": float(np.mean(all_r2)) if all_r2 else 0.0,
            "mean_rmse": float(np.mean(all_rmse)) if all_rmse else 0.0,
            "mean_mae": float(np.mean(all_mae)) if all_mae else 0.0,
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
            name=f"grid_transfer_{'_'.join(dataset_names)}",
            failure_config=tune.FailureConfig(max_failures=max_failures),
        ),
    )
    tuner.fit()

    # ---- Clean up incomplete directories from failed trials ----
    trial_dirs_all = [
        d for d in os.listdir(grid_experiment_dir)
        if d.isdigit() and os.path.isdir(os.path.join(grid_experiment_dir, d))
    ]
    for td in trial_dirs_all:
        td_path = os.path.join(grid_experiment_dir, td)
        # Joint training: check if all dataset results CSVs exist
        has_all = all(
            os.path.exists(os.path.join(td_path, f"results_{ds}.csv"))
            for ds in dataset_names
        )
        if not has_all:
            import shutil
            print(f"[Cleanup] Removing incomplete trial directory: {td_path}")
            shutil.rmtree(td_path, ignore_errors=True)

    # ---- Summarize all trial results ----
    all_trial_results = []
    trial_dirs = sorted(
        [d for d in os.listdir(grid_experiment_dir)
         if d.isdigit() and os.path.isdir(os.path.join(grid_experiment_dir, d))],
        key=int
    )
    for td in trial_dirs:
        td_path = os.path.join(grid_experiment_dir, td)
        yaml_cfg_path = os.path.join(td_path, "config.yaml")
        if not os.path.exists(yaml_cfg_path):
            continue
        import yaml as yaml_mod
        with open(yaml_cfg_path, "r", encoding="utf-8") as f:
            trial_cfg = yaml_mod.safe_load(f)

        row = {"Trial": int(td)}
        for param_name in grid_param_names:
            row[param_name] = str(trial_cfg.get(param_name, ""))
        for ds_name in dataset_names:
            csv_path = os.path.join(td_path, f"results_{ds_name}.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                last_row = df.iloc[-1]
                for metric in ["R2", "RMSE", "MAE"]:
                    raw = str(last_row.get(metric, ""))
                    parts = raw.split("\u00b1")
                    if len(parts) == 2:
                        row[f"Mean_{metric}_{ds_name}"] = parts[0].strip()
                        row[f"Std_{metric}_{ds_name}"] = parts[1].strip()
                    else:
                        row[f"Mean_{metric}_{ds_name}"] = raw.strip()
                        row[f"Std_{metric}_{ds_name}"] = ""
                # Article-aware metrics
                for article_metric in ["article_range_hit_rate", "article_nars"]:
                    raw = str(last_row.get(article_metric, ""))
                    if raw and raw != "nan":
                        parts = raw.split("\u00b1")
                        if len(parts) == 2:
                            row[f"Mean_{article_metric}_{ds_name}"] = parts[0].strip()
                            row[f"Std_{article_metric}_{ds_name}"] = parts[1].strip()
                        else:
                            row[f"Mean_{article_metric}_{ds_name}"] = raw.strip()
                            row[f"Std_{article_metric}_{ds_name}"] = ""
        all_trial_results.append(row)

    summary_df = pd.DataFrame(all_trial_results)
    summary_df.to_csv(
        os.path.join(grid_experiment_dir, "grid_summary.csv"),
        index=False, encoding="utf-8-sig"
    )

    # Save grid config
    if yaml_path and os.path.exists(yaml_path):
        import shutil
        shutil.copy2(yaml_path, os.path.join(grid_experiment_dir, "grid_config.yaml"))
    else:
        save_config_yaml(
            {k: v for k, v in fixed_cfg.items() if not k.startswith("_")},
            os.path.join(grid_experiment_dir, "grid_config.yaml")
        )

    print(f"\n[Grid Search Complete] Results saved to {grid_experiment_dir}")


def main():
    import copy
    from datetime import datetime

    ds_label = "_".join(TRAIN_DATASETS)

    print(f"\n{'#'*70}")
    print(f"  Transfer Training: {TRAIN_DATASETS}")
    print(f"{'#'*70}")

    if GRID_YAML:
        # ---- YAML Grid Search Mode ----
        yaml_full_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), GRID_YAML
        )
        base_cfg, search_space, grid_param_names, linked_unpacking = \
            load_grid_yaml(yaml_full_path)

        if not search_space:
            # No grid params in YAML, treat as single-config training
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            if _is_struct_only(base_cfg):
                model_label = f"struct_{base_cfg.get('STRUCT_ENCODER', 'polybert')}"
            else:
                model_label = base_cfg.get('MODEL_TYPE', 'model')
            exp_dir = os.path.join(RESULT_DIR, f"transfer_{ds_label}_{model_label}_{ts}")
            ensure_dir(exp_dir)
            train_transfer(base_cfg, TRAIN_DATASETS, exp_dir, skip_test=True)
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            grid_dir = os.path.join(RESULT_DIR, f"transfer_{ds_label}_grid_{ts}")
            ensure_dir(grid_dir)
            train_transfer_ray_tune(
                copy.deepcopy(base_cfg), search_space, TRAIN_DATASETS, grid_dir,
                grid_param_names=grid_param_names,
                linked_unpacking=linked_unpacking,
                yaml_path=yaml_full_path,
            )
    else:
        # ---- Single Config Module Mode ----
        cfg = load_config_as_dict(CONFIG_MODULE)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if _is_struct_only(cfg):
            model_label = f"struct_{cfg.get('STRUCT_ENCODER', 'polybert')}"
        else:
            model_label = cfg.get('MODEL_TYPE', 'model')

        exp_dir = os.path.join(RESULT_DIR, f"transfer_{ds_label}_{model_label}_{ts}")
        ensure_dir(exp_dir)
        train_transfer(cfg, TRAIN_DATASETS, exp_dir)


if __name__ == "__main__":
    main()
