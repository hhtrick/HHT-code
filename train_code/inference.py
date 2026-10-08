"""
inference.py — General inference script

Supports two types of weight directories:
    1) Directories produced by train.py: contain config.yaml + scaler.pkl + weights_seed*.pth
    2) Directories produced by continue_train_inference.py: contain
         continue_train_config.json + weights_seed{SEED}.pth + config.yaml/scaler.pkl
         The "weights_dir" field in that JSON points to the original train.py output directory,
         from which config.yaml / ratio_encoding.json / scaler.pkl are read to instantiate the model.

Inference outputs (written to OUTPUT_DIR):
    - predictions.json: true and predicted values for each entry (CT comparison mode has before/after fields)
    - metrics.csv: R2 / RMSE / MAE metrics
    - When POOLING_TYPE in {attention_pooling, sigmoid_pooling}:
            pooling_weights[_before_ct|_after_ct].json
            records the token list and corresponding pooling weights for downstream inspection.
"""
import os
import sys
import json
import pickle
import re
import numpy as np
import pandas as pd
import torch
import yaml
from datetime import datetime
from typing import Dict, Any, List, Optional

from config.path import MODEL_PATHS, ensure_dir
from config.prompt import STRUCT_INPUT, build_prompt
from target_scaling import validate_scaler_config, scaler_transform, validate_targets
from utils import (
    compute_metrics, inverse_transform, load_ratio_encoding,
    build_ratio_vector, load_trainable_weights,
    periogt_collate_fn, precompute_periogt_data,
)


# ======================== Inference Parameter Settings ========================
# Model weights folder (output directory of train.py or continue_train_inference.py)
WEIGHTS_DIR = "autodl-tmp/results/semantic_dynamic_prompt_1357/Tm_qwen3_4b_base_20260409_172026"

# Data file for inference (recommended to put under val_dataset/)
DATA_FILE = "val_dataset/val_Tm_1.json"

# Input type (1: type_1, 2: type_2, 9: chemical_composition only, etc.)
INPUT_TYPE = 2

# Single seed (continue-train directories usually contain only one seed)
SEED = 42

# Inference batch size
BATCH_SIZE = 2

# Output directory (None defaults to WEIGHTS_DIR/inference_<timestamp>)
OUTPUT_DIR = None

# When WEIGHTS_DIR is a continue-train directory, whether to also infer with "pre-CT" weights and compare
COMPARE_BEFORE_AFTER_CT = True
# ==============================================================


# ======================== Helper Functions ========================

def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _is_struct_only(cfg: Dict[str, Any]) -> bool:
    return "STRUCT_ENCODER" in cfg and "ENABLE_HYBRID_ENCODING" not in cfg


def _infer_dataset_name(data_file: str) -> str:
    """Infer property name from filename: val_Tg_1.json -> Tg; Tg.json -> Tg"""
    base = os.path.splitext(os.path.basename(data_file))[0]
    m = re.match(r"^val_(.+?)(?:_\d+)?$", base)
    return m.group(1) if m else base


def _resolve_ratio_path(cfg: Dict[str, Any], train_weights_dir: str) -> Optional[str]:
    """Same lookup logic as continue_train_inference.py: cfg original path -> train directory -> parent directory."""
    candidates = [
        cfg.get("_RATIO_ENCODING_PATH"),
        os.path.join(train_weights_dir, "ratio_encoding.json"),
    ]
    parent = train_weights_dir
    for _ in range(3):
        parent = os.path.dirname(parent)
        if parent and parent != os.path.dirname(parent):
            candidates.append(os.path.join(parent, "ratio_encoding.json"))
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


# ======================== Weight Directory Resolution ========================

def resolve_weight_sources(weights_dir: str, seed: int, compare_ct: bool,
                           property_name: str = "") -> Dict[str, Any]:
    """
    Returns a dictionary:
      {
        "is_ct": bool,
                "train_weights_dir": original train.py output directory,
        "config_yaml": str, "cfg": dict,
        "scaler_path": str,
        "ratio_path": Optional[str],
        "stages": [(stage_name, weight_path), ...]
          - Standard directory: [("inference", path)]
                    - CT directory (no comparison): [("after_ct", path)]
          - CT directory (comparison):   [("before_ct", path), ("after_ct", path)]
      }
    """
    ct_cfg_path = os.path.join(weights_dir, "continue_train_config.json")
    is_ct = os.path.exists(ct_cfg_path)

    if is_ct:
        ct_meta = _load_json(ct_cfg_path)
        train_weights_dir = ct_meta["weights_dir"]
        if not os.path.isabs(train_weights_dir):
            train_weights_dir = os.path.normpath(train_weights_dir)
        if not os.path.isdir(train_weights_dir):
            print(f"[Warning] Original train.py weights_dir not found: {train_weights_dir}")

        config_yaml = os.path.join(weights_dir, "config.yaml")
        if not os.path.exists(config_yaml):
            config_yaml = os.path.join(train_weights_dir, "config.yaml")
        ct_weight = os.path.join(weights_dir, f"weights_seed{seed}.pth")
        if not os.path.exists(ct_weight):
            print(f"[Error] Continue-train weight not found: {ct_weight}")
            sys.exit(1)
        pre_weight = os.path.join(train_weights_dir, f"weights_seed{seed}.pth")

        # New CT runs store their exact scaler; legacy runs fall back to the original directory.
        scaler_path = os.path.join(weights_dir, "scaler.pkl")
        if not os.path.exists(scaler_path):
            scaler_path = os.path.join(train_weights_dir, "scaler.pkl")

        stages = [("after_ct", ct_weight)]
        if compare_ct:
            if os.path.exists(pre_weight):
                stages.insert(0, ("before_ct", pre_weight))
            else:
                print(f"[Warning] Pre-CT weight not found: {pre_weight}; skipping comparison.")
    else:
        train_weights_dir = weights_dir
        config_yaml = os.path.join(weights_dir, "config.yaml")
        scaler_path = os.path.join(weights_dir, "scaler.pkl")
        weight = os.path.join(weights_dir, f"weights_seed{seed}.pth")
        if not os.path.exists(weight):
            print(f"[Error] Weight not found: {weight}")
            sys.exit(1)
        stages = [("inference", weight)]

    if not os.path.exists(scaler_path) and property_name:
        # Joint transfer training stores one scaler per property.
        scaler_path = os.path.join(train_weights_dir, f"scaler_{property_name}.pkl")

    if not os.path.exists(config_yaml):
        print(f"[Error] config.yaml not found: {config_yaml}")
        sys.exit(1)
    if not os.path.exists(scaler_path):
        print(f"[Error] scaler.pkl not found: {scaler_path}")
        sys.exit(1)

    cfg = _load_yaml(config_yaml)
    ratio_path = _resolve_ratio_path(cfg, train_weights_dir)
    if ratio_path:
        cfg["_RATIO_ENCODING_PATH"] = ratio_path

    stage_scaler_paths = {stage: scaler_path for stage, _ in stages}
    stage_config_paths = {stage: config_yaml for stage, _ in stages}
    if "before_ct" in stage_scaler_paths:
        before_scaler = os.path.join(train_weights_dir, "scaler.pkl")
        if not os.path.exists(before_scaler) and property_name:
            before_scaler = os.path.join(train_weights_dir, f"scaler_{property_name}.pkl")
        if not os.path.exists(before_scaler):
            raise FileNotFoundError(f"Pre-CT weights require their original scaler: {before_scaler}")
        stage_scaler_paths["before_ct"] = before_scaler
        stage_config_paths["before_ct"] = os.path.join(train_weights_dir, "config.yaml")

    return {
        "is_ct": is_ct,
        "train_weights_dir": train_weights_dir,
        "config_yaml": config_yaml,
        "cfg": cfg,
        "scaler_path": scaler_path,
        "ratio_path": ratio_path,
        "stages": stages,
        "stage_scaler_paths": stage_scaler_paths,
        "stage_config_paths": stage_config_paths,
    }


# ======================== External Model Loading (for frozen mode) ========================

def _load_external_llm(model_key: str, device: str):
    from models.base_model import LLM_REGISTRY
    llm_cls = LLM_REGISTRY[model_key]
    llm = llm_cls(MODEL_PATHS[model_key], device=device)
    llm.model.eval()
    return llm


def _load_external_struct_encoder(encoder_name: str, device: str):
    from models.base_model import STRUCT_ENCODER_REGISTRY
    encoder_cls = STRUCT_ENCODER_REGISTRY[encoder_name]["encoder_cls"]
    return encoder_cls(MODEL_PATHS[encoder_name], device=device)


def _encode_struct_batch(encoder, entries: List[Dict],
                         ratio_encoding: Optional[Dict], device: str):
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


def _extract_smiles_data(entries: List[Dict], ratio_encoding: Optional[Dict]):
    smiles_lists, ratio_vectors_lists = [], []
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


# ======================== Inference Core Loop ========================

def run_inference(model, data: List[Dict], dataset_name: str,
                  cfg: Dict[str, Any], device: str,
                  llm=None, struct_encoder=None,
                  ratio_encoding: Optional[Dict] = None,
                  periogt_data: Optional[List[Dict]] = None,
                  collect_pooling: bool = False,
                  tokenizer=None):
    """
        Returns:
      preds_scaled: ndarray(N,)
            pooling_records: list[dict] or None; each entry contains tokens / token_ids / weights
    """
    struct_only = _is_struct_only(cfg)
    frozen = cfg.get("FROZEN_BACKBONE", True)
    enable_hybrid = cfg.get("ENABLE_HYBRID_ENCODING", False)
    encoder_name = cfg.get("STRUCT_ENCODER", "polybert")

    n = len(data)
    all_preds = []
    pooling_records: List[Dict[str, Any]] = [] if collect_pooling else None

    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        batch = data[start:end]

        with torch.no_grad():
            # ---- Semantic stream ----
            text_hidden, text_mask, batch_input_ids = None, None, None
            if struct_only and encoder_name != "llm":
                pass
            else:
                use_type = STRUCT_INPUT if (struct_only and encoder_name == "llm") else INPUT_TYPE
                texts = [build_prompt(e, dataset_name, use_type) for e in batch]
                if frozen and llm is not None:
                    inputs = llm.tokenize(texts)
                    text_hidden = llm.get_hidden_states(inputs).float().to(device)
                    text_mask = inputs["attention_mask"].to(device)
                    batch_input_ids = inputs["input_ids"]
                elif model.llm is not None:
                    inputs = model.llm.tokenize(texts)
                    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                              for k, v in inputs.items()}
                    text_hidden = model.llm.get_hidden_states(inputs).float()
                    text_mask = inputs.get("attention_mask")
                    batch_input_ids = inputs["input_ids"]

            # ---- Structural stream ----
            struct_emb, struct_mask, graph_data = None, None, None
            need_struct = (
                (struct_only and encoder_name != "llm") or
                (not struct_only and enable_hybrid)
            )
            if need_struct:
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
                        struct_emb, struct_mask = _encode_struct_batch(
                            struct_encoder, batch, ratio_encoding, device)
                    else:
                        sl, rvl = _extract_smiles_data(batch, ratio_encoding)
                        struct_emb, struct_mask = model._encode_smiles_batch(sl, rvl)
                elif encoder_name == "llm":
                    type9_texts = [build_prompt(e, dataset_name, 9) for e in batch]
                    if frozen and llm is not None:
                        t9_inputs = llm.tokenize(type9_texts)
                        struct_emb = llm.get_hidden_states(t9_inputs).float().to(device)
                        struct_mask = t9_inputs["attention_mask"].to(device)
                    elif model.llm is not None:
                        t9_inputs = model.llm.tokenize(type9_texts)
                        t9_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                     for k, v in t9_inputs.items()}
                        struct_emb = model.llm.get_hidden_states(t9_inputs).float()
                        struct_mask = t9_inputs.get("attention_mask")

            # ---- Forward ----
            if struct_only:
                # StructureOnlyModel does not support return_aux
                if encoder_name == "periogt":
                    preds = model(graph_data=graph_data)
                elif encoder_name == "llm":
                    preds = model(text_hidden=text_hidden, text_mask=text_mask)
                else:
                    preds = model(struct_emb=struct_emb, struct_mask=struct_mask)
                aux = None
            else:
                if collect_pooling:
                    preds, aux = model(text_hidden, text_mask, struct_emb, struct_mask,
                                       graph_data=graph_data, return_aux=True)
                else:
                    preds = model(text_hidden, text_mask, struct_emb, struct_mask,
                                  graph_data=graph_data)
                    aux = None

        all_preds.append(preds.cpu().numpy())

        # ---- Collect pooling weights ----
        if collect_pooling and aux is not None and "pooling_weights" in aux:
            pw = aux["pooling_weights"].detach().cpu()  # (b, seq_len)
            mask_cpu = text_mask.detach().cpu() if text_mask is not None else None
            ids_cpu = batch_input_ids.detach().cpu() if batch_input_ids is not None else None
            for j in range(pw.size(0)):
                seq_len = (int(mask_cpu[j].sum().item())
                           if mask_cpu is not None else pw.size(1))
                weights_j = pw[j, :seq_len].tolist()
                tokens_j, ids_j = [], []
                if ids_cpu is not None:
                    ids_j = ids_cpu[j, :seq_len].tolist()
                    if tokenizer is not None:
                        tokens_j = tokenizer.convert_ids_to_tokens(ids_j)
                pooling_records.append({
                    "tokens": tokens_j,
                    "token_ids": ids_j,
                    "weights": weights_j,
                })

    return np.concatenate(all_preds), pooling_records


# ======================== Main Flow ========================

def main():
    weights_dir = WEIGHTS_DIR
    if not os.path.isdir(weights_dir):
        print(f"[Error] WEIGHTS_DIR not found: {weights_dir}")
        sys.exit(1)
    if not os.path.isfile(DATA_FILE):
        print(f"[Error] DATA_FILE not found: {DATA_FILE}")
        sys.exit(1)

    dataset_name = _infer_dataset_name(DATA_FILE)
    data = _load_json(DATA_FILE)
    if data and dataset_name not in data[0].get("properties", {}):
        candidates = [key for key, value in data[0].get("properties", {}).items()
                      if isinstance(value, dict) and "value" in value]
        if len(candidates) != 1:
            raise ValueError(f"Cannot infer one target property from {DATA_FILE}: {candidates}")
        dataset_name = candidates[0]
    src = resolve_weight_sources(weights_dir, SEED, COMPARE_BEFORE_AFTER_CT, dataset_name)
    cfg = src["cfg"]
    targets_raw = np.array(
        [float(e["properties"][dataset_name]["value"][0]) for e in data],
        dtype=np.float64,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_DIR or os.path.join(weights_dir, f"inference_{timestamp}")
    ensure_dir(output_dir)

    print(f"[Mode] {'Continue-Train' if src['is_ct'] else 'Standard'}")
    print(f"[Train cfg] {src['config_yaml']}")
    print(f"[Scaler] {src['scaler_path']}")
    print(f"[Data] {DATA_FILE} ({len(data)} entries, dataset={dataset_name})")
    print(f"[Stages] {[s[0] for s in src['stages']]}")
    print(f"[Output] {output_dir}")

    struct_only = _is_struct_only(cfg)
    frozen = cfg.get("FROZEN_BACKBONE", True)
    enable_hybrid = cfg.get("ENABLE_HYBRID_ENCODING", False)
    encoder_name = cfg.get("STRUCT_ENCODER", "polybert")
    pool_type = cfg.get("POOLING_TYPE", "")

    collect_pooling = (
        not struct_only and
        pool_type in ("attention_pooling", "sigmoid_pooling")
    )
    print(f"[Pooling] type={pool_type}, collect_weights={collect_pooling}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    ratio_encoding = (load_ratio_encoding(src["ratio_path"])
                      if src["ratio_path"] else None)

    llm = None
    need_external_llm = frozen and (
        (not struct_only) or (struct_only and encoder_name == "llm")
    )
    if need_external_llm:
        model_key = cfg["MODEL_TYPE"]
        print(f"[LLM] Loading {model_key} ...")
        llm = _load_external_llm(model_key, device=device)

    struct_encoder_ext = None
    need_ext_struct = frozen and (
        (enable_hybrid and encoder_name == "polybert") or
        (struct_only and encoder_name == "polybert")
    )
    if need_ext_struct:
        print(f"[Struct] Loading {encoder_name} ...")
        struct_encoder_ext = _load_external_struct_encoder(encoder_name, device=device)

    periogt_data = None
    if (struct_only and encoder_name == "periogt") or (enable_hybrid and encoder_name == "periogt"):
        print("[PerioGT] Preparing graph data ...")
        periogt_data = precompute_periogt_data(
            dataset_name, data, ratio_encoding,
            use_prompt=cfg.get("USE_PERIOGT_PROMPT", True),
        )

    from models.base_model import PropertyPredictionModel, StructureOnlyModel
    print("[Model] Building ...")
    model = StructureOnlyModel(cfg) if struct_only else PropertyPredictionModel(cfg)
    model.to(device)

    tokenizer = None
    if collect_pooling:
        if llm is not None:
            tokenizer = llm.tokenizer
        elif getattr(model, "llm", None) is not None:
            tokenizer = model.llm.tokenizer

    stage_results: Dict[str, Dict[str, Any]] = {}
    for stage_name, weight_path in src["stages"]:
        print(f"\n{'='*60}\n  Stage: {stage_name}  ({os.path.basename(weight_path)})\n{'='*60}")
        scaler = _load_pickle(src["stage_scaler_paths"][stage_name])
        scale_cfg = _load_yaml(src["stage_config_paths"][stage_name])
        validate_scaler_config(scaler, scale_cfg, dataset_name, context=stage_name)
        validate_targets(targets_raw, scaler_transform(scaler))
        load_trainable_weights(model, weight_path, device=device)
        model.eval()

        preds_scaled, pooling_records = run_inference(
            model, data, dataset_name, cfg, device,
            llm=llm, struct_encoder=struct_encoder_ext,
            ratio_encoding=ratio_encoding, periogt_data=periogt_data,
            collect_pooling=collect_pooling, tokenizer=tokenizer,
        )
        preds_orig = inverse_transform(scaler, preds_scaled)
        metrics = compute_metrics(preds_orig, targets_raw)
        print(f"  R2={metrics['R2']:.4f}  RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}")

        stage_results[stage_name] = {
            "preds_orig": preds_orig,
            "metrics": metrics,
            "pooling_records": pooling_records,
        }

    # Free up GPU memory
    if llm is not None:
        del llm
    if struct_encoder_ext is not None:
        del struct_encoder_ext
    torch.cuda.empty_cache()

    # ---- predictions.json ----
    predictions = []
    for i in range(len(targets_raw)):
        rec = {"index": i, "true_value": float(targets_raw[i])}
        for stage_name, res in stage_results.items():
            key = ("predicted_value" if stage_name == "inference"
                   else f"predicted_value_{stage_name}")
            rec[key] = float(res["preds_orig"][i])
        predictions.append(rec)
    pred_path = os.path.join(output_dir, "predictions.json")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"\n[Saved] {pred_path}")

    # ---- metrics.csv ----
    csv_rows = []
    for stage_name, res in stage_results.items():
        csv_rows.append({"stage": stage_name, **res["metrics"]})
    if "before_ct" in stage_results and "after_ct" in stage_results:
        m_pre = stage_results["before_ct"]["metrics"]
        m_post = stage_results["after_ct"]["metrics"]
        csv_rows.append({
            "stage": "improvement",
            "R2": m_post["R2"] - m_pre["R2"],
            "RMSE": m_pre["RMSE"] - m_post["RMSE"],
            "MAE": m_pre["MAE"] - m_post["MAE"],
        })
    csv_path = os.path.join(output_dir, "metrics.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[Saved] {csv_path}")

    # ---- pooling_weights JSON ----
    if collect_pooling:
        for stage_name, res in stage_results.items():
            recs = res["pooling_records"]
            if not recs:
                continue
            suffix = "" if stage_name == "inference" else f"_{stage_name}"
            out_records = []
            for i, r in enumerate(recs):
                out_records.append({
                    "index": i,
                    "true_value": float(targets_raw[i]),
                    "pred_value": float(res["preds_orig"][i]),
                    "tokens": r["tokens"],
                    "token_ids": r["token_ids"],
                    "weights": r["weights"],
                    "pooling_type": pool_type,
                    "model_type": cfg.get("MODEL_TYPE"),
                })
            pw_path = os.path.join(output_dir, f"pooling_weights{suffix}.json")
            with open(pw_path, "w", encoding="utf-8") as f:
                json.dump(out_records, f, ensure_ascii=False, indent=2)
            print(f"[Saved] {pw_path}")

    # ---- inference_meta.json ----
    meta = {
        "weights_dir": weights_dir,
        "is_continue_train": src["is_ct"],
        "train_weights_dir": src["train_weights_dir"],
        "data_file": DATA_FILE,
        "dataset_name": dataset_name,
        "input_type": STRUCT_INPUT if (struct_only and encoder_name == "llm") else INPUT_TYPE,
        "seed": SEED,
        "stages": [{"name": n, "weight": w} for n, w in src["stages"]],
        "input_format": "raw_text",
        "pooling_type": pool_type,
        "collect_pooling_weights": collect_pooling,
        "compare_before_after_ct": COMPARE_BEFORE_AFTER_CT,
    }
    meta_path = os.path.join(output_dir, "inference_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {meta_path}")

    print(f"\n{'='*60}")
    print(f"  Done - output at {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
