"""
compute_article_baseline.py — Article-memorization hypothesis baseline computation

For randomly split datasets (split_random.pkl and split_random_*pct.pkl at various training set ratios),
as well as continue-training scenarios (train=subset specified by split_continue_train.pkl["train"]
from continue_train_dataset/{name}/{name}.json; test=val_dataset/val_{name}_*.json),
construct a null model:
  The model is assumed to only remember article-level statistics; the prediction rule for each test sample is:
  - If the sample's article appeared in the training set → predict that article's mean (or median) in training
  - If the sample's article did not appear in training → predict the global mean (or median) of the training set

Compute R², MAE, RMSE from the predicted vs. true values of all test samples
(three metrics output for each (dataset, split) combination).

Outputs three CSVs to autodl-tmp/baseline_results/:
  1. baseline_mean.csv     — mean strategy
  2. baseline_median.csv   — median strategy
  3. baseline_best.csv     — for each (dataset, split), takes the strategy with higher R² and labels it
"""
import os
import re
import glob
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ======================== Parameter Settings ========================
DATASETS = ["Tg", "Tm", "E", "UTS", "eps", "n"]

# Split pkl filename patterns to scan under the main dataset directory
# Includes split_random.pkl and split_random_20pct.pkl / 40pct / 60pct / 80pct etc.
SPLIT_PKL_PATTERNS = ["split_random.pkl", "split_random_*pct.pkl"]

# Split pkl filename under the continue-training dataset directory
CONTINUE_TRAIN_SPLIT_PKL = "split_continue_train.pkl"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")
CONTINUE_TRAIN_DATASET_DIR = os.path.join(SCRIPT_DIR, "continue_train_dataset")
VAL_DATASET_DIR = os.path.join(SCRIPT_DIR, "val_dataset")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "autodl-tmp", "baseline_results")
# ========================================================


def _compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    """Compute R², RMSE, MAE; returns NaN if sample count is insufficient."""
    if len(targets) < 2:
        return {"R2": float("nan"), "RMSE": float("nan"), "MAE": float("nan")}
    return {
        "R2": float(r2_score(targets, preds)),
        "RMSE": float(np.sqrt(mean_squared_error(targets, preds))),
        "MAE": float(mean_absolute_error(targets, preds)),
    }


def _get_title(entry: dict) -> str:
    return entry.get("title", "") or ""


def _get_value(entry: dict, prop_name: str) -> float:
    return float(entry["properties"][prop_name]["value"][0])


def _infer_prop_name(data: list, dataset_name: str) -> str:
    if not data:
        return dataset_name
    props = data[0].get("properties", {})
    if dataset_name in props:
        return dataset_name
    for key in props:
        if "value" in props[key]:
            return key
    return dataset_name


def _baseline_from_train_test(
    train_entries: list, test_entries: list,
    prop_name: str, strategy: str,
) -> dict:
    """
    Given train/test entry lists, compute baseline metrics based on the article-memorization hypothesis.
    """
    agg_fn = np.mean if strategy == "mean" else np.median

    train_values = np.array([_get_value(e, prop_name) for e in train_entries])
    if len(train_values) == 0:
        return {"R2": float("nan"), "RMSE": float("nan"), "MAE": float("nan")}
    global_agg = float(agg_fn(train_values))

    article_train_values: dict = {}
    for e in train_entries:
        article_train_values.setdefault(_get_title(e), []).append(_get_value(e, prop_name))
    article_agg = {t: float(agg_fn(v)) for t, v in article_train_values.items()}

    test_targets = np.array([_get_value(e, prop_name) for e in test_entries])
    test_preds = np.array([
        article_agg.get(_get_title(e), global_agg) for e in test_entries
    ])
    return _compute_metrics(test_preds, test_targets)


# ======================== split_random* Scenario ========================

def compute_baseline_for_random_split(
    dataset_name: str, pkl_path: str, strategy: str,
) -> dict:
    """Based on a split pkl, compute train → test baseline metrics."""
    json_path = os.path.join(DATASET_DIR, dataset_name, f"{dataset_name}.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(pkl_path, "rb") as f:
        split = pickle.load(f)

    prop_name = _infer_prop_name(data, dataset_name)
    train_entries = [data[i] for i in split["train"]]
    test_entries = [data[i] for i in split["test"]]

    metrics = _baseline_from_train_test(train_entries, test_entries, prop_name, strategy)

    return {
        "Dataset": dataset_name,
        "Split": os.path.splitext(os.path.basename(pkl_path))[0],
        "Strategy": strategy,
        "N_Train": len(train_entries),
        "N_Test": len(test_entries),
        **metrics,
    }


# ======================== continue_train Scenario ========================

def compute_baseline_for_continue_train(
    dataset_name: str, strategy: str,
) -> list:
    """
    Training set: continue_train_dataset/{name}/{name}.json via split_continue_train.pkl["train"]
    Test set:     each file under val_dataset/val_{name}_*.json (one result per file)

    Returns [] if relevant files are missing.
    """
    ct_json = os.path.join(CONTINUE_TRAIN_DATASET_DIR, dataset_name, f"{dataset_name}.json")
    ct_pkl = os.path.join(CONTINUE_TRAIN_DATASET_DIR, dataset_name, CONTINUE_TRAIN_SPLIT_PKL)
    if not (os.path.exists(ct_json) and os.path.exists(ct_pkl)):
        return []

    val_files = sorted(glob.glob(os.path.join(VAL_DATASET_DIR, f"val_{dataset_name}_*.json")))
    if not val_files:
        return []

    with open(ct_json, "r", encoding="utf-8") as f:
        ct_data = json.load(f)
    with open(ct_pkl, "rb") as f:
        ct_split = pickle.load(f)

    prop_name = _infer_prop_name(ct_data, dataset_name)
    train_entries = [ct_data[i] for i in ct_split.get("train", [])]

    rows = []
    for vf in val_files:
        with open(vf, "r", encoding="utf-8") as f:
            test_entries = json.load(f)
        metrics = _baseline_from_train_test(train_entries, test_entries, prop_name, strategy)
        split_tag = f"continue_train__{os.path.splitext(os.path.basename(vf))[0]}"
        rows.append({
            "Dataset": dataset_name,
            "Split": split_tag,
            "Strategy": strategy,
            "N_Train": len(train_entries),
            "N_Test": len(test_entries),
            **metrics,
        })
    return rows


# ======================== Sort Key ========================

_PCT_RE = re.compile(r"split_random_(\d+)pct")


def _split_sort_key(split_name: str):
    """Sort splits semantically: 20pct < 40pct < 60pct < 80pct < 100% (split_random) < continue_train."""
    m = _PCT_RE.match(split_name)
    if m:
        return (0, int(m.group(1)))
    if split_name == "split_random":
        return (0, 100)
    if split_name.startswith("continue_train"):
        return (1, split_name)
    return (2, split_name)


# ======================== Main Flow ========================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    mean_rows = []
    median_rows = []

    for ds in DATASETS:
        ds_dir = os.path.join(DATASET_DIR, ds)
        # ---- Collect all split_random*.pkl for this dataset ----
        pkl_paths: list = []
        for pat in SPLIT_PKL_PATTERNS:
            pkl_paths.extend(glob.glob(os.path.join(ds_dir, pat)))
        # Deduplicate and sort semantically
        pkl_paths = sorted(set(pkl_paths),
                           key=lambda p: _split_sort_key(os.path.splitext(os.path.basename(p))[0]))

        if not pkl_paths:
            print(f"[WARN] No split_random*.pkl found in {ds_dir}, skipping random splits for {ds}")

        for pkl_path in pkl_paths:
            mean_row = compute_baseline_for_random_split(ds, pkl_path, "mean")
            median_row = compute_baseline_for_random_split(ds, pkl_path, "median")
            mean_rows.append(mean_row)
            median_rows.append(median_row)
            print(f"\n[{ds} | {mean_row['Split']}] "
                  f"Train={mean_row['N_Train']}, Test={mean_row['N_Test']}")
            print(f"  Mean   — R2={mean_row['R2']:.4f}, "
                  f"RMSE={mean_row['RMSE']:.4f}, MAE={mean_row['MAE']:.4f}")
            print(f"  Median — R2={median_row['R2']:.4f}, "
                  f"RMSE={median_row['RMSE']:.4f}, MAE={median_row['MAE']:.4f}")

        # ---- Continue-training scenario ----
        ct_mean = compute_baseline_for_continue_train(ds, "mean")
        ct_median = compute_baseline_for_continue_train(ds, "median")
        for mean_row, median_row in zip(ct_mean, ct_median):
            mean_rows.append(mean_row)
            median_rows.append(median_row)
            print(f"\n[{ds} | {mean_row['Split']}] "
                  f"Train={mean_row['N_Train']}, Test={mean_row['N_Test']}")
            print(f"  Mean   — R2={mean_row['R2']:.4f}, "
                  f"RMSE={mean_row['RMSE']:.4f}, MAE={mean_row['MAE']:.4f}")
            print(f"  Median — R2={median_row['R2']:.4f}, "
                  f"RMSE={median_row['RMSE']:.4f}, MAE={median_row['MAE']:.4f}")

    if not mean_rows:
        print("[Error] No baseline computed. Check dataset / split pkl paths.")
        return

    df_mean = pd.DataFrame(mean_rows)
    df_mean.to_csv(os.path.join(OUTPUT_DIR, "baseline_mean.csv"),
                   index=False, float_format="%.6f")
    print(f"\nSaved: {os.path.join(OUTPUT_DIR, 'baseline_mean.csv')}")

    df_median = pd.DataFrame(median_rows)
    df_median.to_csv(os.path.join(OUTPUT_DIR, "baseline_median.csv"),
                     index=False, float_format="%.6f")
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'baseline_median.csv')}")

    # ---- Best strategy CSV ----
    best_rows = []
    for mean_row, median_row in zip(mean_rows, median_rows):
        mean_r2 = mean_row["R2"] if not np.isnan(mean_row["R2"]) else -np.inf
        median_r2 = median_row["R2"] if not np.isnan(median_row["R2"]) else -np.inf
        best_row = dict(median_row if median_r2 > mean_r2 else mean_row)
        best_row["Best_Strategy"] = "median" if median_r2 > mean_r2 else "mean"
        best_rows.append(best_row)

    df_best = pd.DataFrame(best_rows)
    cols = ["Dataset", "Split", "Best_Strategy"] + [
        c for c in df_best.columns if c not in ("Dataset", "Split", "Best_Strategy")
    ]
    df_best = df_best[cols]
    df_best.to_csv(os.path.join(OUTPUT_DIR, "baseline_best.csv"),
                   index=False, float_format="%.6f")
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'baseline_best.csv')}")


if __name__ == "__main__":
    main()

