"""
generate_split_article.py — Article-isolated dataset splitting (strict separation)

Ensures that all samples from the same article (identified by the `title` field)
go exclusively to ONE split — train, val, or test. This prevents article-level
feature leakage across splits and forces the model to generalize across articles.

Algorithm (stratified range-aware assignment):
  1. Read JSON dataset, group samples by `title` field.
  2. For the `eps` dataset: exclude 3 known extreme-outlier samples
     (negative dielectric constant / ultra-high composites) from all splits.
  3. Compute each article's mean target value, sort articles by mean ascending.
  4. Divide the sorted articles into NUM_STRATA equal-sized bins (strata).
  5. Within each stratum: force at least 1 article to train (ensures train
     covers the full value range), then assign remaining articles greedily
     to approach the target 70/10/20 split ratio within that stratum.
  6. Shuffle the training indices, then generate five nested subsets
     at 20%, 40%, 60%, 80%, 100% of the full training set.

Output files (stored under dataset/<name>/):
  - split_article.pkl          — 100% training + val + test
  - split_article_20pct.pkl    — 20% training subset + val + test
  - split_article_40pct.pkl
  - split_article_60pct.pkl
  - split_article_80pct.pkl

pkl format: {"train": [...], "val": [...], "test": [...]}
"""

import os
import json
import pickle
import numpy as np
from collections import OrderedDict

# ======================== Parameters ========================
RANDOM_SEED = 42                          # Random seed for shuffling article groups
TRAIN_RATIO = 0.7                         # Target training set ratio
VAL_RATIO = 0.1                           # Target validation set ratio
TEST_RATIO = 0.2                          # Target test set ratio

# List of datasets to process
DATASETS = ["Tg", "Tm", "E", "UTS", "eps", "n"]

# Number of value strata (bins) for stratified assignment.
# Articles are sorted by mean target value and divided into this many equal-sized
# bins. Within each bin, 70/10/20 proportional assignment is enforced, and train
# is guaranteed at least one article from every bin to cover the full value range.
NUM_STRATA = 10

# Output pkl filename prefix (without suffix; .pkl / _{pct}pct.pkl appended automatically)
OUTPUT_NAME = "split_article"

# Indices to exclude from the eps dataset (extreme outliers with fundamentally
# different physical mechanisms: negative-epsilon materials / ultra-high composites)
EPS_EXCLUDED_INDICES = {64, 201, 317}

# Dataset root directory (relative to this script's location)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")
# ==========================================================


def _group_by_title(data, exclude_indices=None):
    """
    Group entry indices by title field, returning a list of (title, [idx, ...])
    tuples for articles with at least one valid sample.

    Args:
        data: list of JSON entries.
        exclude_indices: set of indices to exclude (e.g., eps outliers);
                         excluded entries never appear in any group.

    Returns:
        List of (title, [idx1, idx2, ...]) in arbitrary order (caller will sort).
    """
    if exclude_indices is None:
        exclude_indices = set()

    groups = OrderedDict()
    for idx, entry in enumerate(data):
        if idx in exclude_indices:
            continue
        title = entry.get("title", "") or ""
        groups.setdefault(title, []).append(idx)

    return [(t, indices) for t, indices in groups.items()]


def _stratified_assign(article_list, data, dataset_name, rng):
    """
    Stratified article-level assignment with train range maximization.

    1. Compute each article's mean target value, sort articles by mean ascending.
    2. Divide sorted articles into NUM_STRATA equal-sized bins.
    3. Within each bin:
       a. Shuffle articles randomly.
       b. Force at least 1 article to train (ensures train covers full value range).
       c. Assign remaining articles greedily to approach 70/10/20 within the bin.

    This guarantees:
      - Article isolation (each article assigned exactly once).
      - Train spans all value strata (train range covers val/test ranges).
      - Each value stratum contributes proportionally to all three splits.

    Args:
        article_list: list of (title, [idx1, idx2, ...]).
        data: full JSON dataset (needed to compute per-article means).
        dataset_name: property name (e.g. 'Tg', 'E').
        rng: numpy RandomState for shuffling.

    Returns:
        train_idx, val_idx, test_idx — three lists of integer indices.
    """
    # Compute per-article statistics: (title, indices, article_mean, n_samples,
    # article_min, article_max). The min/max are used for boundary bin forced
    # assignment to guarantee train covers absolute extremes (not just means).
    article_info = []
    for title, indices in article_list:
        vals = [float(data[i]["properties"][dataset_name]["value"][0]) for i in indices]
        article_mean = np.mean(vals)
        article_min = np.min(vals)
        article_max = np.max(vals)
        article_info.append((title, indices, article_mean, len(indices),
                             article_min, article_max))

    # Sort by article mean ascending
    article_info.sort(key=lambda x: x[2])

    K = len(article_info)
    num_strata = min(NUM_STRATA, K)

    # Divide sorted articles into num_strata equal-sized bins
    bin_size = K // num_strata
    bins = []
    for s in range(num_strata):
        start = s * bin_size
        if s == num_strata - 1:
            end = K  # last bin gets the remainder
        else:
            end = start + bin_size
        bins.append(article_info[start:end])

    train_idx = []
    val_idx = []
    test_idx = []

    for bin_idx, bin_articles in enumerate(bins):
        # Shuffle articles within this bin (for middle bins, random order;
        # for boundary bins we will override the first pick below).
        bin_order = list(range(len(bin_articles)))
        rng.shuffle(bin_order)
        bin_shuffled = [bin_articles[i] for i in bin_order]

        # Step 1: Force at least 1 article to train (coverage guarantee).
        # For the first bin, force the article with the absolute minimum
        # sample value to train. For the last bin, force the article with
        # the absolute maximum sample value. This ensures train range
        # strictly contains val/test ranges even when article means
        # smooth out extreme individual samples.
        # For middle bins, pick randomly (already shuffled).
        if bin_idx == 0:
            # Lowest bin: pick article with minimum sample value for train
            forced_idx = min(range(len(bin_shuffled)),
                             key=lambda i: bin_shuffled[i][4])  # article_min
        elif bin_idx == num_strata - 1:
            # Highest bin: pick article with maximum sample value for train
            forced_idx = max(range(len(bin_shuffled)),
                             key=lambda i: bin_shuffled[i][5])  # article_max
        else:
            forced_idx = 0  # Random (already shuffled)

        forced = bin_shuffled.pop(forced_idx)
        train_idx.extend(forced[1])

        # Step 2: Assign remaining articles greedily within this bin
        # (use bin-local counters for proportional within-stratum assignment)
        bin_train = []
        bin_val = []
        bin_test = []

        for _title, indices, _mean, _n, _min_v, _max_v in bin_shuffled:
            cur_t = len(bin_train)
            cur_v = len(bin_val)
            cur_ts = len(bin_test)
            cur_total = cur_t + cur_v + cur_ts

            if cur_total == 0:
                # First remaining article: use global deviation to decide
                # (train already got the forced article, so it's ahead)
                g_t = len(train_idx)
                g_v = len(val_idx)
                g_ts = len(test_idx)
                g_total = g_t + g_v + g_ts
                frac_t = g_t / g_total
                frac_v = g_v / g_total
                frac_ts = g_ts / g_total
            else:
                frac_t = cur_t / cur_total
                frac_v = cur_v / cur_total
                frac_ts = cur_ts / cur_total

            dev_t = TRAIN_RATIO - frac_t
            dev_v = VAL_RATIO - frac_v
            dev_ts = TEST_RATIO - frac_ts

            if dev_t >= dev_v and dev_t >= dev_ts:
                bin_train.extend(indices)
            elif dev_v >= dev_ts:
                bin_val.extend(indices)
            else:
                bin_test.extend(indices)

        train_idx.extend(bin_train)
        val_idx.extend(bin_val)
        test_idx.extend(bin_test)

    return train_idx, val_idx, test_idx


def generate_splits(dataset_name: str):
    json_path = os.path.join(DATASET_DIR, dataset_name, f"{dataset_name}.json")
    output_dir = os.path.join(DATASET_DIR, dataset_name)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n_total = len(data)

    # Determine excluded indices
    if dataset_name == "eps":
        exclude_indices = EPS_EXCLUDED_INDICES
    else:
        exclude_indices = set()

    n_excluded = len(exclude_indices)
    n_effective = n_total - n_excluded

    # Group by title
    article_list = _group_by_title(data, exclude_indices)
    rng = np.random.RandomState(RANDOM_SEED)

    # Stratified article-level assignment with train range maximization
    train_idx, val_idx, test_idx = _stratified_assign(
        article_list, data, dataset_name, rng)

    # ---- Verify no overlap and full coverage ----
    all_set = set(train_idx) | set(val_idx) | set(test_idx)
    assert len(all_set) == len(train_idx) + len(val_idx) + len(test_idx), \
        "ERROR: Duplicate indices across splits!"
    assert all_set.isdisjoint(exclude_indices), \
        "ERROR: Excluded indices leaked into splits!"
    assert len(all_set) == n_effective, \
        f"ERROR: Coverage mismatch: {len(all_set)} assigned vs {n_effective} effective"

    # ---- Verify article isolation constraint ----
    _verify_article_isolation(data, train_idx, val_idx, test_idx)

    # ---- Shuffle training set (ensures nested subset relationships) ----
    train_arr = np.array(train_idx)
    rng.shuffle(train_arr)
    train_idx = train_arr.tolist()

    # ---- Print statistics ----
    _print_stats(data, dataset_name, n_total, n_excluded, len(article_list),
                 train_idx, val_idx, test_idx)

    # ---- Generate splits at different training set ratios ----
    train_fractions = [0.2, 0.4, 0.6, 0.8, 1.0]

    for frac in train_fractions:
        pct = int(frac * 100)
        n_sub = max(1, int(len(train_idx) * frac))
        # Take first n_sub from the already shuffled train_idx,
        # guaranteeing 20% ⊂ 40% ⊂ 60% ⊂ 80% ⊂ 100% nesting
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
        print(f"  {pct}% split saved: {pkl_path}")
        print(f"    train={len(sub_train_idx)}, val={len(val_idx)}, test={len(test_idx)}")


def _verify_article_isolation(data, train_idx, val_idx, test_idx):
    """
    Verify that every article appears in at most one split (strict isolation).
    """
    def get_title(idx):
        return data[idx].get("title", "") or ""

    train_titles = {get_title(i) for i in train_idx}
    val_titles = {get_title(i) for i in val_idx}
    test_titles = {get_title(i) for i in test_idx}

    # Check pairwise intersections
    tv_overlap = train_titles & val_titles
    tt_overlap = train_titles & test_titles
    vt_overlap = val_titles & test_titles

    if tv_overlap or tt_overlap or vt_overlap:
        print(f"  [FAIL] Article isolation VIOLATED!")
        if tv_overlap:
            print(f"    train ∩ val: {tv_overlap}")
        if tt_overlap:
            print(f"    train ∩ test: {tt_overlap}")
        if vt_overlap:
            print(f"    val ∩ test: {vt_overlap}")
        raise AssertionError("Article isolation constraint violated!")
    else:
        print(f"  [OK] Article isolation constraint verified "
              f"({len(train_titles)} train articles, "
              f"{len(val_titles)} val articles, "
              f"{len(test_titles)} test articles — all disjoint)")


def _print_stats(data, dataset_name, n_total, n_excluded, n_articles,
                 train_idx, val_idx, test_idx):
    """Print dataset split statistics."""
    train_vals = [float(data[i]["properties"][dataset_name]["value"][0]) for i in train_idx]
    val_vals = [float(data[i]["properties"][dataset_name]["value"][0]) for i in val_idx]
    test_vals = [float(data[i]["properties"][dataset_name]["value"][0]) for i in test_idx]

    n_effective = n_total - n_excluded

    print(f"\n{'='*50}")
    print(f"[{dataset_name}] total={n_total}, effective={n_effective}, "
          f"articles={n_articles}")
    if n_excluded > 0:
        print(f"  Excluded {n_excluded} outlier samples: indices {sorted(EPS_EXCLUDED_INDICES)}")
    print(f"  train={len(train_idx)} ({len(train_idx)/n_effective*100:.1f}%), "
          f"val={len(val_idx)} ({len(val_idx)/n_effective*100:.1f}%), "
          f"test={len(test_idx)} ({len(test_idx)/n_effective*100:.1f}%)")
    if train_vals:
        print(f"  mean  — train: {np.mean(train_vals):.2f}, "
              f"val: {np.mean(val_vals):.2f}, "
              f"test: {np.mean(test_vals):.2f}")
        print(f"  std   — train: {np.std(train_vals):.2f}, "
              f"val: {np.std(val_vals):.2f}, "
              f"test: {np.std(test_vals):.2f}")
        print(f"  range — train: [{np.min(train_vals):.2f}, {np.max(train_vals):.2f}], "
              f"val: [{np.min(val_vals):.2f}, {np.max(val_vals):.2f}], "
              f"test: [{np.min(test_vals):.2f}, {np.max(test_vals):.2f}]")


if __name__ == "__main__":
    for ds in DATASETS:
        generate_splits(ds)
    print("\nDone.")
