"""CPU TF-IDF + Ridge using the six published Type 2 datasets and entry-level splits.

Run from the repository root: python train_code/run_tfidf_ridge.py
TF is computed per Type 2 record, and IDF is fitted across the source DOIs of the training records.
Validation RMSE selects the ridge penalty for each property.
The summary, validation scores, test predictions and run manifest are written to train_code/.
Use --properties Tg for a single property.
"""

import argparse
import csv
import hashlib
import json
import pickle
import platform
from datetime import datetime, timezone
from os import getpid
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import scipy
import sklearn
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from threadpoolctl import threadpool_limits

from config.path import DATASET_DIR
from config.prompt import build_prompt
from target_scaling import fit_target_scaler


HERE = Path(__file__).resolve().parent
PROPERTIES = ("Tg", "Tm", "E", "UTS", "eps", "n")
ALPHAS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
TOKENIZATION = {
    "analyzer": "word",
    "token_pattern": r"\w+|[^\w\s]",  # Retain single-letter elements and SMILES punctuation.
    "lowercase": False,
    "ngram_range": (1, 2),
}
WEIGHTING = {
    "sublinear_tf": True,
    "smooth_idf": True,
    "norm": "l2",
}
RIDGE = {"solver": "lsqr", "fit_intercept": True, "tol": 1e-6, "max_iter": 10000}


def doi(record):
    key = str(record.get("doi") or "").strip().lower()
    return key.removeprefix("https://doi.org/").removeprefix("http://doi.org/").removeprefix("doi:").strip()


def load_inputs(data_dir, name):
    directory = data_dir / name
    data_file = directory / f"{name}.json"
    split_file = directory / "split_random.pkl"
    data_bytes, split_bytes = data_file.read_bytes(), split_file.read_bytes()
    hashes = {
        data_file.name: hashlib.sha256(data_bytes.replace(b"\r\n", b"\n")).hexdigest(),
        split_file.name: hashlib.sha256(split_bytes).hexdigest(),
    }

    records = json.loads(data_bytes)
    split = pickle.loads(split_bytes)
    indices = [i for part in ("train", "val", "test") for i in split[part]]
    if sorted(indices) != list(range(len(records))):
        raise ValueError(f"{name}: split indices must cover every record exactly once.")
    return records, split, hashes


def article_term_presence(train_counts, train_dois):
    """One sparse row per DOI, with the union of terms found within its training records."""
    if any(not key for key in train_dois):
        raise ValueError("A training record lacks the DOI needed for article-level IDF.")
    articles, groups = np.unique(train_dois, return_inverse=True)
    membership = csr_matrix(
        (np.ones(len(groups)), (groups, np.arange(len(groups)))),
        shape=(len(articles), len(groups)),
    )
    presence = membership @ train_counts
    presence.data.fill(1.0)
    return presence


def select_ridge(x_train, y_train, x_val, y_val, scaler):
    scores = {}
    best_model = None
    best_rmse = np.inf
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha, **RIDGE).fit(x_train, y_train)
        prediction = scaler.inverse_transform(model.predict(x_val)).ravel()
        rmse = float(np.sqrt(mean_squared_error(y_val, prediction)))
        scores[alpha] = rmse
        if rmse < best_rmse:  # The ascending grid favors the smaller alpha on an exact tie.
            best_rmse, best_model = rmse, model
    return best_model, scores


def evaluate_property(name, records, split):
    train, val, test = (split[part] for part in ("train", "val", "test"))
    prompts = [build_prompt(record, name, input_type=2) for record in records]
    labels = [record["properties"][name] for record in records]
    units = {label.get("unit") or "dimensionless" for label in labels}
    if len(units) != 1:
        raise ValueError(f"{name}: expected one target unit, found {units}.")
    unit = units.pop()
    targets = np.asarray([label["value"][0] for label in labels], dtype=np.float64)
    scaler = fit_target_scaler(targets, train, property_name=name)
    y_train = scaler.transform(targets[train, None]).ravel()

    vectorizer = CountVectorizer(dtype=np.float64, **TOKENIZATION)
    train_counts = vectorizer.fit_transform(prompts[i] for i in train)
    val_counts = vectorizer.transform(prompts[i] for i in val)
    test_counts = vectorizer.transform(prompts[i] for i in test)
    article_counts = article_term_presence(train_counts, [doi(records[i]) for i in train])

    weighting = TfidfTransformer(**WEIGHTING).fit(article_counts)
    model, scores = select_ridge(weighting.transform(train_counts), y_train,
                                 weighting.transform(val_counts), targets[val], scaler)
    pred = scaler.inverse_transform(model.predict(weighting.transform(test_counts))).ravel()
    summary = {
        "property": name, "unit": unit, "idf_unit": "article",
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "n_train_articles": article_counts.shape[0], "n_idf_documents": article_counts.shape[0],
        "n_features": train_counts.shape[1], "selected_alpha": model.alpha,
        "val_rmse": scores[model.alpha], "ridge_test_r2": r2_score(targets[test], pred),
        "ridge_test_rmse": float(np.sqrt(mean_squared_error(targets[test], pred))),
    }
    validation = [
        {"property": name, "unit": unit, "idf_unit": "article", "alpha": alpha,
         "val_rmse": score, "selected": alpha == model.alpha}
        for alpha, score in scores.items()
    ]

    predictions = [
        {"property": name, "record_index": i, "doi": records[i].get("doi"), "unit": unit,
         "y_true": targets[i], "y_pred": pred[j],
         "type2_text_sha256": hashlib.sha256(prompts[i].encode("utf-8")).hexdigest()}
        for j, i in enumerate(test)
    ]
    return summary, validation, predictions


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(properties, data_dir, output_dir):
    started = datetime.now(timezone.utc)
    settings = {
        "input_hash_convention": "SHA-256 of LF-normalized JSON and unmodified binary split files",
        "blas_threads": 1,
        "started_utc": started.isoformat(), "properties": list(properties),
        "input_type": 2, "split_file": "split_random.pkl", "record_index_base": 0,
        "alpha_grid": ALPHAS, "selection": "Minimum validation RMSE with ties resolved by the smaller alpha.",
        "fit_partitions": {"tfidf": "train", "target_scaler": "train", "ridge": "train"},
        "target_scaling": "train mean and standard deviation with metrics in original units",
        "tfidf": {**TOKENIZATION, **WEIGHTING},
        "idf_documents": "Each source DOI is one IDF document, using the union of terms from its training records.",
        "tf_unit": "single Type 2 record, including all bigrams",
        "ridge": RIDGE,
        "input_sha256": {},
        "prompt_code_sha256": hashlib.sha256((HERE / "config/prompt.py").read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "scikit-learn": sklearn.__version__},
    }
    tables = {"article_idf": [], "validation": [], "test_predictions": []}
    for name in properties:
        print(f"Evaluating {name} ...", flush=True)
        records, split, hashes = load_inputs(data_dir, name)
        with threadpool_limits(limits=1, user_api="blas"):
            result, scores, rows = evaluate_property(name, records, split)
        settings["input_sha256"][name] = hashes
        tables["validation"].extend(scores)
        tables["test_predictions"].extend(rows)
        tables["article_idf"].append(result)
        print(f"  alpha={result['selected_alpha']:g}  test R2={result['ridge_test_r2']:.4f}  "
              f"RMSE={result['ridge_test_rmse']:.5g} ({result['unit']})", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"r3_tfidf_ridge_{started:%Y%m%dT%H%M%S_%fZ}_{getpid()}"
    with TemporaryDirectory(dir=output_dir) as tmp:
        stage = Path(tmp)
        artifacts = []
        settings["files"] = {}
        for kind, rows in tables.items():
            path = stage / f"{prefix}_{kind}.csv"
            write_csv(path, rows)
            settings["files"][kind] = {"name": path.name, "rows": len(rows),
                                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            artifacts.append(path)
        manifest = stage / f"{prefix}_run.json"
        manifest.write_text(json.dumps(settings, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        artifacts.append(manifest)
        published = []
        try:
            for path in artifacts:
                destination = output_dir / path.name
                path.rename(destination)
                published.append(destination)
        except OSError:
            for path in published:
                path.unlink(missing_ok=True)
            raise
    result_manifest = output_dir / manifest.name
    print(f"Saved results directly in {output_dir.resolve()}\nRun manifest: {result_manifest.name}")
    return result_manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--properties", nargs="+", choices=PROPERTIES, default=PROPERTIES)
    parser.add_argument("--output-dir", "--output-root", type=Path, default=HERE,
                        help="Write all result files directly in this directory (default: train_code/).")
    args = parser.parse_args()
    if len(set(args.properties)) != len(args.properties):
        parser.error("Each property may only be selected once.")
    run(args.properties, Path(DATASET_DIR), args.output_dir)
