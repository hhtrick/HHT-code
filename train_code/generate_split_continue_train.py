"""
generate_split_continue_train.py — Generate train/val split indices for the continue-training dataset

Keeps the same logic as generate_split_random.py, but operates on
continue_train_dataset/{name}/{name}.json.

Notes:
  - The continue-training phase has no test set (test data comes from val_dataset/),
    so this script only splits train/val;
  - If VAL_RATIO is 0, all samples are used as the training set, and the val list is empty (no validation set);
  - The output pkl is saved in continue_train_dataset/{name}/, next to the JSON file;
  - Split ratios are configured directly at the top of this file, no longer maintained in config_continue_train.py.
"""
import os
import json
import pickle
import numpy as np

# ======================== Parameters ========================
RANDOM_SEED = 42

# Training / Validation ratio (sum should be <= 1; the remainder is discarded, but usually they add up to 1.0)
# If VAL_RATIO = 0.0, the validation set is empty (no validation set)
TRAIN_RATIO = 1
VAL_RATIO = 0

# Continue-training datasets to process
DATASETS = ["Tm"]

# Output pkl filename (without suffix; .pkl will be appended automatically)
OUTPUT_NAME = "split_continue_train"

# Continue-training dataset root directory (relative to this script's location)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONTINUE_TRAIN_DIR = os.path.join(SCRIPT_DIR, "continue_train_dataset")
# ==========================================================


def generate_splits(dataset_name: str):
    json_path = os.path.join(CONTINUE_TRAIN_DIR, dataset_name, f"{dataset_name}.json")
    output_dir = os.path.join(CONTINUE_TRAIN_DIR, dataset_name)

    if not os.path.exists(json_path):
        print(f"[WARN] {json_path} not found, skipping {dataset_name}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n = len(data)
    indices = np.arange(n)

    rng = np.random.RandomState(RANDOM_SEED)
    rng.shuffle(indices)

    if not (0.0 <= VAL_RATIO < 1.0) or not (0.0 < TRAIN_RATIO <= 1.0):
        raise ValueError(
            f"Invalid ratios: TRAIN_RATIO={TRAIN_RATIO}, VAL_RATIO={VAL_RATIO}"
        )
    if TRAIN_RATIO + VAL_RATIO > 1.0 + 1e-9:
        raise ValueError(
            f"TRAIN_RATIO + VAL_RATIO = {TRAIN_RATIO + VAL_RATIO} > 1.0"
        )

    n_val = int(n * VAL_RATIO)
    n_train = n - n_val if (TRAIN_RATIO + VAL_RATIO >= 1.0 - 1e-9) else int(n * TRAIN_RATIO)

    train_idx = indices[:n_train].tolist()
    val_idx = indices[n_train:n_train + n_val].tolist()

    pkl_path = os.path.join(output_dir, f"{OUTPUT_NAME}.pkl")
    split = {
        "train": train_idx,
        "val": val_idx,
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(split, f)
    print(f"[{dataset_name}] split saved: {pkl_path}")
    print(f"  total={n}, train={len(train_idx)}, val={len(val_idx)}")


if __name__ == "__main__":
    for ds in DATASETS:
        print(f"\n{'='*50}")
        print(f"Processing continue-train dataset: {ds}")
        print(f"{'='*50}")
        generate_splits(ds)
    print("\nDone.")
