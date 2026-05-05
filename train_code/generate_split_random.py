"""
generate_split_random.py — Generate train/val/test split index pkl files
"""
import os
import json
import pickle
import numpy as np

# ======================== Parameters ========================
RANDOM_SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

# List of datasets to process
DATASETS = ["Tg","Tm","E","UTS","eps","n"]

# Output pkl filename (without suffix; .pkl will be appended automatically)
OUTPUT_NAME = "split_random"

# Dataset root directory (relative to this script's location)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")
# ==========================================================


def generate_splits(dataset_name: str):
    json_path = os.path.join(DATASET_DIR, dataset_name, f"{dataset_name}.json")
    output_dir = os.path.join(DATASET_DIR, dataset_name)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n = len(data)
    indices = np.arange(n)

    rng = np.random.RandomState(RANDOM_SEED)
    rng.shuffle(indices)

    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_idx = indices[:n_train].tolist()
    val_idx = indices[n_train:n_train + n_val].tolist()
    test_idx = indices[n_train + n_val:].tolist()

    # ---- Generate splits at different training set ratios ----
    # Ratios: 20%, 40%, 60%, 80%, 100%
    train_fractions = [0.2, 0.4, 0.6, 0.8, 1.0]

    for frac in train_fractions:
        pct = int(frac * 100)
        n_sub = max(1, int(len(train_idx) * frac))
        # Directly take the first n_sub from the already shuffled train_idx, guaranteeing 20% ⊂ 40% ⊂ 60% ⊂ 80% ⊂ 100%
        sub_train_idx = train_idx[:n_sub]

        if frac >= 1.0:
            pkl_path = os.path.join(output_dir, f"{OUTPUT_NAME}.pkl")
        else:
            pkl_path = os.path.join(output_dir, f"{OUTPUT_NAME}_{pct}pct.pkl")

        split = {
            "train": sub_train_idx,
            "val": val_idx,
            "test": test_idx,
        }
        with open(pkl_path, "wb") as f:
            pickle.dump(split, f)
        print(f"[{dataset_name}] {pct}% split saved: {pkl_path}")
        print(f"  train={len(sub_train_idx)}, val={len(val_idx)}, test={len(test_idx)}")


if __name__ == "__main__":
    for ds in DATASETS:
        print(f"\n{'='*50}")
        print(f"Processing dataset: {ds}")
        print(f"{'='*50}")
        generate_splits(ds)
    print("\nDone.")
