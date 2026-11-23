# HHT training framework

This directory contains model training, embedding precomputation, inference, article-level baselines and the curated datasets. All commands below assume that the working directory is `train_code/`.

## 1. Files and dependencies

```text
train_code/
├── config/                      # Training settings, model paths, prompts and YAML grids
├── models/                      # LLM wrappers, regression models, PolyBERT and PerioGT
├── dataset/                     # Six main JSON datasets and fixed split indices
├── continue_train_dataset/      # Domain training/validation pools and indices
├── val_dataset/                 # Held-out domain test records
├── train.py                     # Main training and grid search
├── train_transfer.py            # Joint training across properties
├── precompute_embeddings.py     # Main and domain LLM caches
├── continue_train_inference.py  # Domain-specific continual training and evaluation
├── inference.py                 # Inference using trained weights
├── compute_article_baseline.py  # Mean/median article baselines
├── domain_embedding_cache.py    # Shared domain cache implementation
├── target_scaling.py            # Target scaling and optional log transformation
├── utils.py                     # Shared datasets, caches, metrics and plotting
└── requirements.txt
```

Use a CUDA-capable training environment. Install a compatible PyTorch build first, then:

```bash
python -m pip install -r requirements.txt
```

The included PerioGT adaptation uses PyTorch Geometric; it does not require the separate `PerioGT-main` reference checkout. RDKit and Mordred are used for graph features. LoRA and hyperparameter search use PEFT and Ray Tune, respectively.

FlashAttention is optional. After installing compatible PyTorch/CUDA components, it can be installed separately with `python -m pip install flash-attn --no-build-isolation`. `config/path.py` controls the attention backend and SDPA fallback. A missing or incompatible FlashAttention installation does not require disabling embedding precomputation.

## 2. Model weights and paths

Paths in `config/path.py` are relative to this directory. Set `MODEL_PATHS` to your downloaded checkpoints. The following LLM keys are supported:

| Key | Backbone |
|---|---|
| `chemdfm_v1_5_8b` | ChemDFM-v1.5-8B |
| `qwen3_0_6b_base` | Qwen3-0.6B-Base |
| `qwen3_4b_base` | Qwen3-4B-Base |
| `qwen3_4b_instruct_2507` | Qwen3-4B-Instruct-2507 |
| `qwen3_4b_thinking_2507` | Qwen3-4B-Thinking-2507 |
| `qwen3_8b_base` | Qwen3-8B-Base |

PolyBERT and PerioGT have separate paths in the same mapping. The local PerioGT architecture settings are in `models/PerioGT/config.yaml`. A model-directory path can point to a resolved checkpoint or a supported cache directory containing a snapshot; tokenizer files must accompany LLM weights.

Generated files normally go under:

- `autodl-tmp/embedding_cache/`: cached model representations.
- `autodl-tmp/results/`: main and joint-training results.
- `autodl-tmp/baseline_results/`: article-baseline CSV files.
- `tf-logs/`: TensorBoard logs.
- The `OUTPUT_DIR` selected in the domain script: domain-training results.

## 3. Datasets and fixed partitions

Each main dataset is a JSON array at `dataset/{property}/{property}.json`. Entries contain source metadata, `chemical_composition`, `processing_history`, `hierarchical_structure`, `properties`, and two generated-analysis objects. Targets are stored as `properties[property]["value"]`; source metadata and target values are not predictive input fields.

| Property | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Tg | 1,210 | 168 | 340 | 1,718 |
| Tm | 952 | 135 | 261 | 1,348 |
| E | 921 | 132 | 249 | 1,302 |
| UTS | 1,323 | 179 | 375 | 1,877 |
| eps | 899 | 127 | 260 | 1,286 |
| n | 915 | 129 | 256 | 1,300 |

The table gives entry-level random partitions. Each dataset also contains:

- `split_random.pkl`: entry-level random split.
- `split_article.pkl`: disjoint source-article split, grouping by normalized DOI with title fallback when needed.
- `split_holdout_whole.pkl`: complete test articles held out, with the remaining records randomly split into training and validation (Supplementary Note 14). Training and validation can share articles; neither shares a test DOI.
- `split_random_{20,40,60,80}pct.pkl` and corresponding `split_article_*pct.pkl`: nested training subsets, with fixed validation/test indices for each split family.

Pickle indices are zero-based positions in the matching JSON array. Preserve their order and use the supplied JSON/PKL pairs together. Curated counts need not equal exact 70/10/20 proportions. The scripts use these saved partitions; they do not create a new split for each seed.

Domain data use separate files:

| Domain key | Training | Validation | Test JSON | Test records |
|---|---:|---:|---|---:|
| `E` | 561 | 140 | `val_dataset/val_E_1.json` | 178 |
| `Tm` | 50 | 0 | `val_dataset/val_Tm_1.json` | 23 |
| `E_article` | 505 | 173 | `val_dataset/val_E_article_1.json` | 201 |

The corresponding pool and `split_continue_train.pkl` are in `continue_train_dataset/{key}/`. Despite the historical folder name, `val_dataset` contains the held-out **domain test sets**, not the domain early-stopping validation sets.

For the additional E-domain evaluation in Supplementary Note 14, use
`continue_train_dataset/E_article/split_holdout_whole.pkl`. Its 505 training and
173 validation indices refer to the same 678-row `E_article.json`. It reuses all
201 records in `val_E_article_1.json` as the test set. No domain JSON is duplicated
or reordered, and the original article-disjoint split remains available.

## 4. Precompute LLM embeddings

Edit the settings at the top of `precompute_embeddings.py`, then run:

```bash
python precompute_embeddings.py
```

| Setting | Purpose |
|---|---|
| `DATASETS` | Main properties to process, in order |
| `DOMAIN_DATASETS` | Optional domain keys: `E`, `Tm`, `E_article` |
| `LLM_MODELS` | LLM keys, processed in order |
| `INPUT_TYPES` | Any requested Types 1–9 and/or `"struct"` |
| `BATCH_SIZE` | Embedding forward-pass batch size |

One LLM is loaded for all selected properties before the next LLM is loaded. Each property/input cache is saved to disk before proceeding. Domain pools and test files use separate caches. For domain-only caching, set `DATASETS=[]` and select `DOMAIN_DATASETS`.

Main training and domain training also compute missing required caches on demand. Main training checks the input types required by its configuration rather than embedding all nine types unconditionally. All LLM input modes use prediction prompts without chat-template wrapping. Pooling operates on the cached final-layer hidden states for those prompts, with padding excluded.

## 5. Main training and model comparisons

Select properties in `TRAIN_DATASETS` near the top of `train.py`. Set `GRID_YAML=None` for a single configuration, choose `CONFIG_MODULE`, and run:

```bash
python train.py
```

### Standard HHT and HHT-C

For the reported fixed configurations, an alternative to editing the working
defaults is `CONFIG_MODULE="config.config_paper"` with `GRID_YAML=None`. Select
`PAPER_MODEL` and `PAPER_PROTOCOL` in that module as described below.

Use `CONFIG_MODULE="config.config_article"`. For the standard HHT configuration:

```python
MODEL_TYPE = "chemdfm_v1_5_8b"
FROZEN_BACKBONE = True
ENABLE_HYBRID_ENCODING = False
INPUT_CONTENT = [2, 4]
EVAL_INPUT_CONTENT = [2]
POOLING_TYPE = "sigmoid_pooling"
MLP_HIDDEN_DIM = [128, 64]
SEEDS = [42, 62, 82]
SPLIT_PKL = "split_random.pkl"
LOG_TARGET_DATASETS = []
```

HHT-C uses `INPUT_CONTENT=[9]` and `EVAL_INPUT_CONTENT=[9]`. Input Type 9 supplies the complete `chemical_composition` object, including names and additives, without `processing_history`. Other input-strategy ablations are selected through these same lists; multiple training types provide dynamic text augmentation. Type 2 contains chemical composition and processing history, while Type 4 is the corresponding hierarchy-free generated description.

### PolyBERT, PerioGT and LLM-struct

Use `CONFIG_MODULE="config.config_stru_article"` and choose `STRUCT_ENCODER`:

- `"polybert"`: monomer SMILES embeddings with ratio information, pooled before regression.
- `"periogt"`: monomer graphs, graph features and ratio information. Its graph backbone remains trainable.
- `"llm"`: LLM-struct, using monomer `smiles`, `ratio_value`, `ratio_unit`, and `is_homopolymer`. Names, additives, repeat-unit pSMILES and processing history are excluded. Set `MODEL_TYPE` to the intended LLM; use `"struct"` for its precomputation cache. Pooling is selected with `STRUCT_POOLING_TYPE`, including `"sigmoid_pooling"`.

The adapted structure encoders read `monomers[].smiles`, not `repeat_unit_psmiles`. Missing structures and composition ratios follow encoder-specific rules in the implementation; LLM-struct matches available field categories, not every encoder's preprocessing operation.

**For the reported comparisons, set `LOG_TARGET_DATASETS=[]` in both configuration modules.** The retained `config_stru_article.py` working default is `["E", "UTS"]`, which enables an optional experiment and does not match the reported result tables. The retained LLM-struct working default is Qwen3-4B-Base; select ChemDFM for the reported ChemDFM structural comparison. Set `NUM_GPUS=1` for one GPU per run, as in the independent-run protocol; changing it changes the effective batch size.

### Target transformation and training schedule

`LOG_TARGET_DATASETS` selects properties modeled as natural-log targets before train-only standardization. Values must be positive. Predictions are inverse-transformed before physical-scale R², RMSE and MAE are computed. An empty list disables the log transformation.

Main training uses a maximum of 200 epochs, warmup followed by cosine annealing, and validation-loss early stopping with patience 15. The selected checkpoint is evaluated on the fixed test split. Three seeds repeat training, not partition generation. Result folders include configuration, target scaler, per-seed trainable weights, metrics and predictions. Structural configurations also require the saved ratio encoding when loading a model.

CSV summaries report the mean and **sample** standard deviation across seeds
(`ddof=1`), matching the paper. With only one seed, the standard deviation is
undefined (`nan`); each seed's metrics remain available separately.

## 6. Data scaling, grid search and joint training

For data scaling, set `SWEEP_TRAIN_FRACTIONS=True`. With `SPLIT_PKL="split_random.pkl"`, the program runs the 20%, 40%, 60%, 80% and full random training partitions in order. Selecting `split_article.pkl` does the same for article splits.

For grid search, set `GRID_YAML` in `train.py` to a path below. Each YAML declares `base_config`, independent `grid` values and, when used, `linked_grid` combinations.

| YAML file | Search |
|---|---|
| `exp1_semantic_frozen.yaml` | Frozen semantic-stream pooling and MLP settings |
| `exp2_semantic_lora.yaml` | Semantic-stream LoRA settings |
| `exp3a_periogt.yaml` | PerioGT regression head and periodic prompt |
| `exp3b_polybert.yaml` | Frozen PolyBERT pooling and head |
| `exp3b_polybert_lora.yaml` | PolyBERT LoRA |
| `exp3c_llm_struct.yaml` | Frozen LLM-struct pooling and head |
| `exp3c_llm_struct_lora.yaml` | LLM-struct LoRA |
| `exp4a_article_semantic.yaml` | Semantic article-aware loss weights |
| `exp4b_article_struct.yaml` | Structural article-aware loss weights |

The last two grids search consistency, bias and ranking weights independently over `[0.0, 0.01, 0.1, 0.2]`: 64 combinations per selected encoder. They optimize loss coefficients, not article properties or labels. Their fixed MSE coefficient comes from the base configuration. Adjust Ray concurrency and GPU fractions to your hardware. Validation results drive grid selection; test evaluation is skipped during grid search.

For joint training, choose `TRAIN_DATASETS` and the configuration in `train_transfer.py`, then run `python train_transfer.py`. It combines the selected training/validation partitions and reports property-specific test metrics. The separately reported hybrid experiments use `train.py` and the fixed settings below; they have no dedicated hyperparameter grid.

### Article-held-out evaluation and hybrid models

Set `CONFIG_MODULE="config.config_paper"` in `train.py` and keep `GRID_YAML=None`.
In `config/config_paper.py`, choose `PAPER_MODEL` from `HHT`, `HHT-C`, `LLM-struct`,
`PolyBERT`, or `PerioGT`, and select `PAPER_PROTOCOL`:

| Protocol | Index file | Evaluation |
|---|---|---|
| `random` | `split_random.pkl` | Entry-level random split |
| `article` | `split_article.pkl` | Article-disjoint training, validation and test |
| `article_holdout` | `split_holdout_whole.pkl` | Unseen test articles, random training–validation split |

These presets use the existing training loop and saved JSON data, three seeds
(42, 62, 82), one GPU per run, no log target transformation, and the reported
encoder, pooling and regression-head settings. They do not search hyperparameters
or select new articles. Run one model at a time; the six properties are already
listed in `TRAIN_DATASETS`.

The fixed `article_holdout` partitions have the following sizes:

| Property | Training | Validation | Test | Test articles |
|---|---:|---:|---:|---:|
| Tg | 1,316 | 188 | 214 | 32 |
| Tm | 1,040 | 149 | 159 | 33 |
| E | 1,009 | 145 | 148 | 30 |
| UTS | 1,458 | 209 | 210 | 39 |
| eps | 871 | 125 | 290 | 37 |
| n | 1,000 | 143 | 157 | 25 |

These test articles were selected with consideration of target-range coverage,
as described in Supplementary Note 14. This is a fixed exploratory protocol,
not a new random split for each run. The test cohort differs from `split_article.pkl`;
numerical comparisons between the two protocols therefore do not isolate the
effect of the validation split alone. The supplied indices reproduce the reported
experiment; no index-generation or one-time orchestration script is needed.

For the two hybrid experiments in Supplementary Note 15, choose
`PAPER_MODEL="HHT+PolyBERT"` or `"HHT+PerioGT"` with `PAPER_PROTOCOL="random"`.
The existing model projects both pooled streams to 512 dimensions, combines them
using a sigmoid gate with 256 hidden units, and predicts with the [128, 64, 1]
MLP. Both use the standard HHT [2,4]→[2] text inputs. Only the reported fixed fusion
configuration is supplied, without a dedicated hybrid search.

## 7. Domain-specific continual training

In `continue_train_inference.py`, set `WEIGHTS_DIR` to a main-task result folder, `DATASET_NAME` to a domain key, `TEST_FILE` to its test JSON, and `INPUT_TYPE` to the desired input. Use `2` for standard HHT evaluation/training here, `9` for HHT-C, or `"struct"` for LLM-struct. Structural model architecture is inferred from the saved configuration.

Set domain settings in `config/config_continue_train.py`. To match the reported training budgets, override its working default of 30 epochs with:

```python
CONTINUE_TRAIN_EPOCHS = 100  # E or E_article; use 10 for Tm
CONTINUE_TRAIN_PATIENCE = 10
```

Then run:

```bash
python continue_train_inference.py
```

The domain loop uses AdamW and linear warmup followed by a constant learning rate, independently of the main-training scheduler. E uses validation-based checkpoint selection; Tm has no validation records and uses its final checkpoint. Set the intended learning rate explicitly rather than assuming that all domains and encoders share one rate.

With `RANDOM_INIT_WEIGHTS=False`, the script starts from the selected main-task weights and their target scaler. With `True`, it resets task-specific trainable parameters and fits a new scaler on the domain training data; downloaded pretrained encoder initialization is retained. Repeat with `SEED=42`, `62`, and `82` for three-seed results. In the pretrained condition, each seed must use the corresponding `weights_seed{SEED}.pth` from the main-task run.

Cache precomputation is shared with `precompute_embeddings.py`; available domain LLM caches are reused. A domain result contains its configuration, trainable weights, scaler, predictions, metrics and plots. One-time multi-experiment replay and seed-supplement orchestration scripts are intentionally excluded; the reusable domain entry point remains available.

### E domain with fixed test articles and random training–validation splitting

For Supplementary Note 14, set the following in `continue_train_inference.py`:

```python
CONTINUE_CONFIG_MODULE = "config.config_continue_article_holdout"
WEIGHTS_DIR = "autodl-tmp/results/<main_E_random_split_run>"
DATASET_NAME = "E_article"
TEST_FILE = "val_dataset/val_E_article_1.json"
INPUT_TYPE = 2  # HHT; use 9 for HHT-C; LLM-struct is inferred from its config.
SEED = 42      # Repeat with 62 and 82, matching the main-task seed.
```

Then run `python continue_train_inference.py` for each model, seed and
initialization condition. The selected preset uses 100 epochs, patience 10,
batch size 16, 10 warmup steps and a learning rate of **1e-4 for both conditions**.
Set `RANDOM_INIT_WEIGHTS=False` or `True` in
`config/config_continue_article_holdout.py`. Use the corresponding main E random
experiment directory for each of the five models. The same main configuration is
required in the random condition; its task weights and scaler are not loaded.
In the inherited condition, the matching seed's task weights and main-task scaler
are loaded. Test predictions are saved before and after continual training.

The LLM caches are exactly the existing `E_article` pool/test caches; changing
the split only changes their index views. Precompute them using
`DOMAIN_DATASETS=["E_article"]`, `INPUT_TYPES=[2, 9, "struct"]` and the ChemDFM
backbone in `precompute_embeddings.py`. This does not require a separate domain
name or a copied dataset. The 35 test DOIs are absent from both development
partitions. The 149 domain DOIs are also absent from the main E dataset.

## 8. Inference and article-level baseline

For inference, edit `WEIGHTS_DIR`, `DATA_FILE`, `INPUT_TYPE` and `SEED` in `inference.py`, then run `python inference.py`. Use the original architecture, model paths and preprocessing metadata associated with those weights. This entry point supports both main and domain result folders.

Run `python compute_article_baseline.py` to evaluate the supplied splits. For a seen source article, the mean or median of its **training** labels is predicted. For an unseen article, the respective global training mean or median is predicted. DOI defines article identity, with title fallback. The program writes mean, median and best-rule summary CSVs; domain baselines use the supplied domain pools and held-out test files.

## 9. Reproducibility boundaries

Training labels, corrected structure fields and split indices are distributed together. Original extraction outputs, manual-audit intermediates, old backups, pretrained weights and one-time repair scripts are not required at runtime. The current configuration files remain editable working configurations; use the explicit settings above and the manuscript's protocol when reproducing a specific comparison.

## CPU matched-text baseline

Run this command from `train_code` using an environment with NumPy, SciPy, scikit-learn and threadpoolctl.

```bash
python run_tfidf_ridge.py
```

The script uses the six fixed entry-level splits and the same Type 2 text as the matched HHT control. Term frequency is computed per record. Inverse document frequency counts a term once for each source DOI where it occurs in at least one training record. The validation split selects the ridge penalty separately for each property.

Each run writes the ridge results (`*_article_idf.csv`), validation scores, test predictions and a JSON run manifest with checksums for the dataset and split inputs. The corresponding HHT ablation results are reported in Supplementary Tables 9 and 23. Use `--properties Tg` for a single property or `--output-dir PATH` to store results elsewhere.
