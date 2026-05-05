# HHT Training Framework

Training framework for the HHT (Holistic History-aware Text) model — polymer property prediction using frozen LLM embeddings with optional structural encoder fusion.

## Architecture

### Semantic Stream (LLM)
- Frozen pretrained LLM backbone (Qwen3 family, ChemDFM) encodes polymer information as natural language text
- Final-layer hidden states pass through LayerNorm → Pooling → MLP regression head
- Supports 9 input types combining different information categories (chemical composition, processing history, hierarchical structure, LLM-generated summaries)
- Dynamic input sampling: randomly selects from multiple input types per training step for augmentation

### Structural Stream (Encoder)
- **PolyBERT** (DeBERTa-v2, hidden_dim=600): Encodes monomer SMILES with ratio information concatenation
- **PerioGT** (Graph Transformer, hidden_dim=3840): Periodic-aware graph transformer with copolymer ratio injection. Originally implemented with DGL by the authors; rewritten using PyTorch Geometric here to resolve environment dependency conflicts.
- **LLM Structure Stream**: Reuses the semantic LLM to process chemical-composition-only inputs as structural features

### Hybrid Encoding
- Dynamic gated fusion of semantic and structural streams: v_final = β · v_text + (1-β) · v_struct
- Both streams projected to a shared dimension, fused via a learned gate network

### Pooling Strategies
Five strategies available: Last Token, Mean Pooling, Sum Pooling, Attention Pooling, Sigmoid Gating

## Directory Structure

```
train_code/
├── config/
│   ├── config_article.py          # Semantic/hybrid stream configuration
│   ├── config_stru_article.py     # Structure-only control group configuration
│   ├── config_continue_train.py   # Continue-training configuration
│   ├── path.py                    # Centralized path management
│   ├── prompt.py                  # Prompt templates for 9 input types
│   └── grid/                      # YAML grid search experiment definitions
│       ├── exp1_semantic_frozen.yaml
│       ├── exp2_semantic_lora.yaml
│       ├── exp3a_periogt.yaml
│       ├── exp3b_polybert.yaml / exp3b_polybert_lora.yaml
│       ├── exp3c_llm_struct.yaml / exp3c_llm_struct_lora.yaml
│       ├── exp4_hybrid.yaml
│       └── exp5a_article_semantic.yaml / exp5b_article_struct.yaml
├── models/
│   ├── base_model.py              # Core Lightning modules (PropertyPredictionModel, StructureOnlyModel)
│   ├── polybert.py                # PolyBERT encoder wrapper
│   ├── qwen3_*.py                 # Qwen3 model wrappers (0.6B, 4B, 8B, instruct, thinking, CPT)
│   ├── chemdfm_v1_5_8b.py         # ChemDFM model wrapper
│   └── PerioGT/                   # PerioGT modules (graph builder, encoder, augmentations)
├── dataset/                       # Training datasets ({Tg,Tm,n,eps,E,UTS}.json + split pkl files)
├── val_dataset/                   # External validation datasets
├── continue_train_dataset/        # Continue-training datasets
├── train.py                       # Main training script (single-config + YAML grid search)
├── train_transfer.py              # Multi-dataset joint training
├── inference.py                   # General inference script
├── continue_train_inference.py    # Continue-training fine-tuning + inference
├── utils.py                       # Data loading, embedding caching, evaluation, visualization
├── generate_split_random.py       # Random train/val/test split generation
├── generate_split_continue_train.py  # Continue-training split generation
├── compute_article_baseline.py    # Article memory hypothesis baseline
├── analyze_article_bias.py        # Article-level bias quantification + post-hoc metrics
└── requirements.txt
```

## Scripts

### Training Scripts

- **`train.py`**: Main training entry point. Supports two modes:
  - **Single-config**: Set `CONFIG_MODULE` to a config module and `GRID_YAML = None`
  - **YAML Grid Search**: Set `GRID_YAML` to a YAML file path for Ray Tune hyperparameter search
- **`train_transfer.py`**: Joint training across multiple datasets. Same interface as `train.py` but `TRAIN_DATASETS` accepts multiple dataset names.

### Inference Scripts

- **`inference.py`**: Evaluate trained models on test/validation datasets. Outputs predictions JSON, metrics CSV, and optional pooling/gate weight visualizations.
- **`continue_train_inference.py`**: Fine-tune a pretrained model on new data, then evaluate. Supports NaN protection, warmup scheduler, and TensorBoard logging.

### Data Preparation Scripts

- **`generate_split_random.py`**: Pure random 70/10/20 train/val/test split with nested 20/40/60/80/100% training subsets.
- **`generate_split_continue_train.py`**: Random split for continue-training datasets.

### Analysis Scripts

- **`compute_article_baseline.py`**: Computes the Article Memory Hypothesis baseline — for each test sample, predicts the training-set mean/median of its source article (if seen) or the global mean/median (if unseen). Quantifies how much performance comes from article-level memorization.
- **`analyze_article_bias.py`**: ANOVA-style variance decomposition (SS_total = SS_between + SS_within), computes η² and ICC(1) metrics, generates σ-within/between/total bar charts, and provides post-hoc normalized metrics (NRMSE_within, R²_within, SkillScore) from existing R²/MAE/RMSE values alone.

## Configuration System

Two primary config modules:

- **`config_article.py`**: Semantic/hybrid stream training. All hyperparameters including model type, pooling strategy, MLP dimensions, LoRA settings, hybrid encoding parameters, Gaussian noise, and article-aware loss weights.
- **`config_stru_article.py`**: Structure-only control group training. Same structure but without LLM semantic stream parameters.

Additional configs:
- **`config_continue_train.py`**: Continue-training hyperparameters (epochs, patience, learning rates).
- **`config/path.py`**: Centralized relative path management for datasets, model weights, embedding caches, and output directories.
- **`config/prompt.py`**: 9 input type prompt templates. Composes polymer data into text prompts for the LLM.

### Key Configuration Parameters

| Category | Key Parameters |
|----------|---------------|
| Training | `LR`, `BACKBONE_LR`, `BATCH_SIZE`, `EPOCHS`, `PATIENCE`, `SEEDS` |
| Model | `MODEL_TYPE`, `FROZEN_BACKBONE`, `POOLING_TYPE`, `PROJ_DIM` |
| LoRA | `LORA_R`, `LORA_ALPHA`, `LORA_DROPOUT`, `LORA_TARGET_MODULES_LLM` |
| MLP | `MLP_HIDDEN_DIM`, `MLP_ACTIVATION`, `MLP_DROPOUT` |
| Hybrid | `ENABLE_HYBRID_ENCODING`, `STRUCT_ENCODER`, `HYBRID_PROJ_DIM`, `HYBRID_GATE_DIM` |
| Article Loss | `MSE_WEIGHT`, `ARTICLE_CONSISTENCY_WEIGHT`, `ARTICLE_BIAS_WEIGHT`, `ARTICLE_RANKING_WEIGHT` |
| Noise | `NOISE_ENABLED`, `NOISE_STD`, `NOISE_ANNEAL` |
| Ray Tune | `RAY_GPU_FRACTION`, `MAX_CONCURRENT_TRIALS`, `RAY_CPU_PER_TRIAL` |

## YAML Grid Search

YAML files in `config/grid/` define hyperparameter search experiments. Each YAML specifies:

- `base_config`: Config module to inherit defaults from
- `grid`: Independent search parameters (Cartesian product)
- `linked_grid`: Grouped parameters searched together (e.g., LoRA rank + alpha)

10 predefined experiments cover:
- **Exp 1**: Semantic stream frozen (pooling + MLP hyperparameters)
- **Exp 2**: Semantic stream LoRA fine-tuning
- **Exp 3a/3b/3c**: Structure stream (PerioGT / PolyBERT / LLM) with and without LoRA
- **Exp 4**: Hybrid stream fusion parameters
- **Exp 5a/5b**: Article-aware loss weights (semantic + structure)

Grid search uses Ray Tune for parallel execution. Results saved to `autodl-tmp/results/` with per-trial configs, weights, and summary CSVs.

## Dataset Format

Training datasets are JSON arrays stored in `dataset/{name}/{name}.json`:

```json
[{
  "doi": "10.1016/...",
  "title": "Paper Title",
  "year": 2020,
  "chemical_composition": {
    "repeat_unit_psmiles": "[*]CC...[*]",
    "monomers": [{"name": "...", "smiles": "...", "ratio_value": 50, "ratio_unit": "mol%"}],
    "additives": [{"name": "...", "type": "filler", "amount_value": 10, "amount_unit": "wt%"}],
    "is_homopolymer": false
  },
  "processing_history": "Synthesis and characterization details...",
  "hierarchical_structure": "Crystallinity, morphology...",
  "properties": {"Tg": {"value": [143.3], "unit": "°C"}},
  "analysis_with_hierarchical_structure": {"descriptive_text": "...", "reasoning_analysis": "..."},
  "analysis_without_hierarchical_structure": {"descriptive_text": "...", "reasoning_analysis": "..."}
}]
```

Split indices are stored as `.pkl` files alongside each dataset JSON.

## Supported Models

| Model | Type | Hidden Dim | Notes |
|-------|------|-----------|-------|
| `qwen3_0_6b_base` | LLM | 896 | Smallest Qwen3 |
| `qwen3_4b_base` | LLM | 2560 | Default backbone |
| `qwen3_4b_instruct_2507` | LLM | 2560 | Instruction-tuned |
| `qwen3_4b_thinking_2507` | LLM | 2560 | Reasoning-tuned |
| `qwen3_8b_base` | LLM | 4096 | Larger Qwen3 |
| `chemdfm_v1_5_8b` | LLM | 4096 | Chemistry-domain LLM |
| `qwen3_4b_base_cpt_{1,2,3}` | LLM | 2560 | Qwen3-4B with domain CPT (LoRA r=4/8/16) |
| `polybert` | Encoder | 600 | Polymer SMILES BERT (DeBERTa-v2) |
| `periogt` | Encoder | 3840 | Periodic Graph Transformer |

> **Note on CPT models**: The original plan included domain-specific continued pre-training (CPT) for the LLM backbone, with corresponding code in `train_code/models/qwen3_4b_base_cpt.py` and `config/config_continue_train.py`. To avoid copyright conflicts, only CC BY 4.0 licensed literature was eligible as training data. However, collecting a sufficient volume of polymer-domain literature meeting the CC BY 4.0 requirement proved difficult, so these CPT models were never actually trained. The code has been retained to faithfully reflect the planned experimental design.

## Embedding Caching

When `FROZEN_BACKBONE=True`, LLM hidden states for all dataset entries are precomputed once and cached as memmap files in `autodl-tmp/embedding_cache/`. Training then loads these cached embeddings without loading the full LLM, substantially reducing GPU memory requirements. Caches are keyed by dataset name and model type to prevent cross-contamination.

PerioGT graph data is similarly precomputed once before grid search trials to avoid redundant pretrained model loading.

## Output Directory

Training results are saved under `autodl-tmp/results/{dataset}_{experiment}_{timestamp}/` including:
- `config.yaml`: Complete training configuration
- `results.csv` / `grid_summary.csv`: Evaluation metrics
- `weights_seed{seed}.pth`: Trainable weights only (no full backbone)
- `scaler.pkl`: Target value scaler for inverse transformation
- `parity_seed{seed}.png`: True vs. predicted scatter plots
- `ratio_encoding.json`: Monomer ratio unit one-hot encoding map
