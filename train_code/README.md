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
- `split_random_{20,40,60,80}pct.pkl` and corresponding `split_article_*pct.pkl`: nested training subsets, with fixed validation/test indices for each split family.

Pickle indices are zero-based positions in the matching JSON array. Preserve their order and use the supplied JSON/PKL pairs together. Curated counts need not equal exact 70/10/20 proportions. The scripts use these saved partitions; they do not create a new split for each seed.

Domain data use separate files:

| Domain key | Training | Validation | Test JSON | Test records |
|---|---:|---:|---|---:|
| `E` | 561 | 140 | `val_dataset/val_E_1.json` | 178 |
| `Tm` | 50 | 0 | `val_dataset/val_Tm_1.json` | 23 |
| `E_article` | 505 | 173 | `val_dataset/val_E_article_1.json` | 201 |

The corresponding pool and `split_continue_train.pkl` are in `continue_train_dataset/{key}/`. Despite the historical folder name, `val_dataset` contains the held-out **domain test sets**, not the domain early-stopping validation sets.

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

For joint training, choose `TRAIN_DATASETS` and the configuration in `train_transfer.py`, then run `python train_transfer.py`. It combines the selected training/validation partitions and reports property-specific test metrics. The model contains optional hybrid-stream machinery, but hybrid fusion was not part of the reported comparison protocol and has no dedicated grid in this release.

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

## 8. Inference and article-level baseline

For inference, edit `WEIGHTS_DIR`, `DATA_FILE`, `INPUT_TYPE` and `SEED` in `inference.py`, then run `python inference.py`. Use the original architecture, model paths and preprocessing metadata associated with those weights. This entry point supports both main and domain result folders.

Run `python compute_article_baseline.py` to evaluate the supplied splits. For a seen source article, the mean or median of its **training** labels is predicted. For an unseen article, the respective global training mean or median is predicted. DOI defines article identity, with title fallback. The program writes mean, median and best-rule summary CSVs; domain baselines use the supplied domain pools and held-out test files.

## 9. Reproducibility boundaries

Training labels, corrected structure fields and split indices are distributed together. Original extraction outputs, manual-audit intermediates, old backups, pretrained weights and one-time repair scripts are not required at runtime. The current configuration files remain editable working configurations; use the explicit settings above and the manuscript's protocol when reproducing a specific comparison.
