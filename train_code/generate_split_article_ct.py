"""
generate_split_article_ct.py — Article-isolated continue-train dataset splitter

Merges two JSON data sources (typically val_dataset/val_X_1.json as train+val and
continue_train_dataset/X/X.json as test), then re-splits the merged data using
stratified article-isolated assignment to ensure:
  - All samples from the same article go exclusively to ONE split
  - Train set covers the full value range
  - Target split ratios: 70% train, 10% val, 20% test

Algorithm:
  1. Read both source JSON files, merge into one list in memory
  2. Group entries by `title` field, compute per-article mean/min/max
  3. Sort articles by mean, divide into NUM_STRATA equal-sized bins
  4. Within each bin: force 1 article to train (boundary bins by sample min/max),
     assign remaining proportionally (70/10/20) using global+local greedy counters
  5. Shuffle train indices, remap for split_continue_train.pkl compatibility

Output:
  - OUTPUT_CT_DIR/OUTPUT_CT_JSON  — train+val entries (for continue_train_dataset)
  - OUTPUT_VAL_FILE               — test entries (for val_dataset)
  - OUTPUT_CT_DIR/SPLIT_PKL_NAME  — train/val index pkl with remapped 0-based indices

After running, update continue_train_inference.py:
  - TEST_FILE = OUTPUT_VAL_FILE
  - DATASET_NAME = directory name under continue_train_dataset/
"""

import os
import json
import pickle
import numpy as np
from collections import OrderedDict

# ======================== Parameters ========================
RANDOM_SEED = 42                          # Random seed for shuffling article groups
TRAIN_RATIO = 0.6                         # Target training set ratio
VAL_RATIO = 0.2                           # Target validation set ratio
TEST_RATIO = 0.2                          # Target test set ratio

# Number of value strata (bins) for stratified assignment.
# Articles are sorted by mean target value and divided into this many equal-sized
# bins. Within each bin, 70/10/20 proportional assignment is enforced, and train
# is guaranteed at least one article from every bin to cover the full value range.
NUM_STRATA = 10

# ---- Source files ----
SOURCE_VAL_FILE = "val_dataset/val_E_1.json"
SOURCE_CT_FILE = "continue_train_dataset/E/E.json"

# ---- Target property name (key in the `properties` dict) ----
PROPERTY_NAME = "E"

# ---- Output paths ----
OUTPUT_CT_DIR = "continue_train_dataset/E_article"
OUTPUT_CT_JSON = "E_article.json"
OUTPUT_VAL_FILE = "val_dataset/val_E_article_1.json"
SPLIT_PKL_NAME = "split_continue_train.pkl"

# Script directory for resolving relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# ============================================================


def _group_by_title(data):
    """
    Group entry indices by title field, returning a list of (title, [idx, ...])
    tuples for articles with at least one entry.
    """
    groups = OrderedDict()
    for idx, entry in enumerate(data):
        title = entry.get("title", "") or ""
        groups.setdefault(title, []).append(idx)
    return [(t, indices) for t, indices in groups.items()]


def _stratified_assign(article_list, data, property_name, rng):
    """
    Stratified article-level assignment with train range maximization.

    1. Compute each article's mean/min/max target value, sort by mean ascending.
    2. Divide sorted articles into NUM_STRATA equal-sized bins.
    3. Within each bin:
       a. Shuffle articles randomly.
       b. Force at least 1 article to train (boundary bins by sample min/max).
       c. Assign remaining articles greedily to approach 70/10/20 within the bin.

    Args:
        article_list: list of (title, [idx1, idx2, ...]).
        data: full merged JSON dataset.
        property_name: target property key (e.g. 'E').
        rng: numpy RandomState for shuffling.

    Returns:
        train_idx, val_idx, test_idx — three lists of integer indices.
    """
    # Compute per-article statistics: (title, indices, mean, n, min, max)
    article_info = []
    for title, indices in article_list:
        vals = [float(data[i]["properties"][property_name]["value"][0])
                for i in indices]
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
        # Shuffle articles within this bin
        bin_order = list(range(len(bin_articles)))
        rng.shuffle(bin_order)
        bin_shuffled = [bin_articles[i] for i in bin_order]

        # Step 1: Force at least 1 article to train (coverage guarantee).
        # First bin → article with absolute minimum sample value to train
        # Last bin  → article with absolute maximum sample value to train
        # Middle bins → random (already shuffled)
        if bin_idx == 0:
            forced_idx = min(range(len(bin_shuffled)),
                             key=lambda i: bin_shuffled[i][4])  # article_min
        elif bin_idx == num_strata - 1:
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


def _verify_article_isolation(data, train_idx, val_idx, test_idx):
    """
    Verify that every article appears in at most one split (strict isolation).
    """
    def get_title(idx):
        return data[idx].get("title", "") or ""

    train_titles = {get_title(i) for i in train_idx}
    val_titles = {get_title(i) for i in val_idx}
    test_titles = {get_title(i) for i in test_idx}

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
        print(f"  [OK] Article isolation verified "
              f"({len(train_titles)} train articles, "
              f"{len(val_titles)} val articles, "
              f"{len(test_titles)} test articles — all disjoint)")


def _print_stats(data, property_name, train_idx, val_idx, test_idx, n_total,
                 n_articles, n_overlap_titles):
    """Print dataset split statistics."""
    train_vals = [float(data[i]["properties"][property_name]["value"][0])
                  for i in train_idx]
    val_vals = [float(data[i]["properties"][property_name]["value"][0])
                for i in val_idx]
    test_vals = [float(data[i]["properties"][property_name]["value"][0])
                 for i in test_idx]

    print(f"\n{'='*50}")
    print(f"[{property_name}] total={n_total}, articles={n_articles}")
    if n_overlap_titles > 0:
        print(f"  (Note: {n_overlap_titles} articles appear in both source files)")
    print(f"  train={len(train_idx)} ({len(train_idx)/n_total*100:.1f}%), "
          f"val={len(val_idx)} ({len(val_idx)/n_total*100:.1f}%), "
          f"test={len(test_idx)} ({len(test_idx)/n_total*100:.1f}%)")
    def _fmt_stats(vals):
        if vals:
            return (f"{np.mean(vals):.4f}", f"{np.std(vals):.4f}",
                    f"[{np.min(vals):.4f}, {np.max(vals):.4f}]")
        else:
            return ("N/A", "N/A", "N/A")

    t_mean, t_std, t_range = _fmt_stats(train_vals)
    v_mean, v_std, v_range = _fmt_stats(val_vals)
    ts_mean, ts_std, ts_range = _fmt_stats(test_vals)
    print(f"  mean  — train: {t_mean}, val: {v_mean}, test: {ts_mean}")
    print(f"  std   — train: {t_std}, val: {v_std}, test: {ts_std}")
    print(f"  range — train: {t_range}, val: {v_range}, test: {ts_range}")

    # Check value coverage
    if train_vals:
        train_min, train_max = np.min(train_vals), np.max(train_vals)
        coverage_issues = []
        if val_vals and (np.min(val_vals) < train_min or np.max(val_vals) > train_max):
            coverage_issues.append("val")
        if test_vals and (np.min(test_vals) < train_min or np.max(test_vals) > train_max):
            coverage_issues.append("test")
        if coverage_issues:
            print(f"  [WARN] Train range does not fully cover: {', '.join(coverage_issues)}")
        else:
            print(f"  [OK] Train range covers val and test")


def main():
    # ---- Resolve absolute paths ----
    source_val_path = os.path.join(SCRIPT_DIR, SOURCE_VAL_FILE)
    source_ct_path = os.path.join(SCRIPT_DIR, SOURCE_CT_FILE)
    output_ct_dir = os.path.join(SCRIPT_DIR, OUTPUT_CT_DIR)
    output_ct_json_path = os.path.join(output_ct_dir, OUTPUT_CT_JSON)
    output_val_path = os.path.join(SCRIPT_DIR, OUTPUT_VAL_FILE)

    # ---- Load both source files ----
    print(f"[Load] {SOURCE_VAL_FILE} ...")
    with open(source_val_path, "r", encoding="utf-8") as f:
        data_val = json.load(f)
    print(f"  {len(data_val)} entries")

    print(f"[Load] {SOURCE_CT_FILE} ...")
    with open(source_ct_path, "r", encoding="utf-8") as f:
        data_ct = json.load(f)
    print(f"  {len(data_ct)} entries")

    # ---- Merge in memory ----
    merged_data = data_val + data_ct
    n_total = len(merged_data)

    # Check for overlapping articles between the two source files
    titles_val = {e.get("title", "") or "" for e in data_val}
    titles_ct = {e.get("title", "") or "" for e in data_ct}
    n_overlap_titles = len(titles_val & titles_ct)

    print(f"\n[Merge] Total: {n_total} entries "
          f"({len(data_val)} from val + {len(data_ct)} from ct)")
    if n_overlap_titles > 0:
        print(f"  Note: {n_overlap_titles} articles appear in both sources "
              f"(merged into same article group)")

    # ---- Group by title ----
    article_list = _group_by_title(merged_data)
    n_articles = len(article_list)
    print(f"[Articles] {n_articles} unique articles")

    # ---- Stratified assignment ----
    rng = np.random.RandomState(RANDOM_SEED)
    train_idx, val_idx, test_idx = _stratified_assign(
        article_list, merged_data, PROPERTY_NAME, rng)

    # ---- Verify coverage and overlap ----
    all_set = set(train_idx) | set(val_idx) | set(test_idx)
    assert len(all_set) == len(train_idx) + len(val_idx) + len(test_idx), \
        "ERROR: Duplicate indices across splits!"
    assert len(all_set) == n_total, \
        f"ERROR: Coverage mismatch: {len(all_set)} assigned vs {n_total} total"

    _verify_article_isolation(merged_data, train_idx, val_idx, test_idx)

    # ---- Print statistics ----
    _print_stats(merged_data, PROPERTY_NAME, train_idx, val_idx, test_idx,
                 n_total, n_articles, n_overlap_titles)

    # ---- Shuffle training set ----
    train_arr = np.array(train_idx)
    rng.shuffle(train_arr)
    train_idx = train_arr.tolist()

    # ---- Extract and save train+val JSON ----
    os.makedirs(output_ct_dir, exist_ok=True)
    train_val_entries = (
        [merged_data[i] for i in train_idx] +
        [merged_data[i] for i in val_idx]
    )
    with open(output_ct_json_path, "w", encoding="utf-8") as f:
        json.dump(train_val_entries, f, ensure_ascii=False, indent=2)
    print(f"\n[Saved] {OUTPUT_CT_DIR}/{OUTPUT_CT_JSON}")
    print(f"  entries: {len(train_val_entries)} "
          f"(train={len(train_idx)} + val={len(val_idx)})")

    # ---- Extract and save test JSON ----
    test_entries = [merged_data[i] for i in test_idx]
    with open(output_val_path, "w", encoding="utf-8") as f:
        json.dump(test_entries, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {OUTPUT_VAL_FILE}")
    print(f"  entries: {len(test_entries)}")

    # ---- Generate split_continue_train.pkl with remapped indices ----
    # Remap: since E_article.json only contains train+val entries,
    # train indices become 0..n_train-1, val indices become n_train..end
    n_train = len(train_idx)
    n_val = len(val_idx)
    remapped_train = list(range(n_train))
    remapped_val = list(range(n_train, n_train + n_val))

    pkl_path = os.path.join(output_ct_dir, SPLIT_PKL_NAME)
    split = {
        "train": remapped_train,
        "val": remapped_val,
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(split, f)
    print(f"[Saved] {OUTPUT_CT_DIR}/{SPLIT_PKL_NAME}")
    print(f"  train={len(remapped_train)}, val={len(remapped_val)}")

    # ---- Final instructions ----
    ct_folder_name = os.path.basename(OUTPUT_CT_DIR)
    print(f"\n{'='*60}")
    print(f"  Done. To use in continue_train_inference.py, set:")
    print(f"    DATASET_NAME = \"{ct_folder_name}\"")
    print(f"    TEST_FILE = \"{OUTPUT_VAL_FILE}\"")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
