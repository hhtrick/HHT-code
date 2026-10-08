"""
config_stru_article.py — Hyperparameter configuration for structural stream training (control group)
Uses only a structural encoder (e.g., PolyBERT, PerioGT, LLM) to extract molecular structure features, then predicts properties via MLP.
Includes article-aware loss parameters (set weights to 0 to disable, equivalent to standard training).
Uses pure random split (split_random.pkl) by default, paired with article-memorization hypothesis baseline for model generalization evaluation.
"""

# ======================== Training Hyperparameters ========================
NUM_WORKERS = 1                           # DataLoader worker processes
NUM_GPUS = 3                              # Max GPUs per training process; under Ray Tune, actual visible GPUs per trial are further limited by GPU fraction scheduling
SEEDS = [42, 62, 82]                      # Random seeds, each seed runs an independent training to assess stability
LR = 1e-4                                 # Task head (pooling + MLP) learning rate
BACKBONE_LR = 1e-6                        # Structural encoder LoRA fine-tuning LR (effective when FROZEN_BACKBONE=False)
WEIGHT_DECAY = 1e-2                       # AdamW weight decay
BATCH_SIZE = 64                           # Samples per GPU per step. Note: for non-YAML single training + DDP multi-GPU, effective batch = BATCH_SIZE * NUM_GPUS; for YAML grid search (Ray Tune), each trial sees only 1 GPU, effective batch equals this value
EPOCHS = 200                              # Max training epochs
WARMUP_STEPS = 100                        # LR linear warmup steps
PATIENCE = 15                             # Early stopping patience: stop if val/loss does not improve for this many epochs
GRADIENT_CLIP_VAL = 1.0                   # Gradient clipping threshold to prevent gradient explosion
GRADIENT_ACCUMULATION_STEPS = 1           # Gradient accumulation steps, effective batch = GRADIENT_ACCUMULATION_STEPS * BATCH_SIZE
FLOAT32_MATMUL_PRECISION = "high"         # PyTorch matmul precision ('highest' | 'high' | 'medium')
SPLIT_PKL = "split_random.pkl"            # Dataset split index filename (default: pure random split; alternatives: 'split.pkl' for article-aware leaky split, 'split_article.pkl' for strict article-isolated split)
LOG_TARGET_DATASETS = ["E", "UTS"]        # ln(y), then train-only StandardScaler; [] disables. All targets must be >0; reported metrics use original units.
SWEEP_TRAIN_FRACTIONS = False              # True: SPLIT_PKL's 20/40/60/80% files, then full file; sequential runs with all SEEDS

# ======================== Structural Encoder ========================
STRUCT_ENCODER = "periogt"                # 'polybert' | 'periogt' | 'llm'; llm uses reduced structural fields (see below)
FROZEN_BACKBONE = True                    # True: freeze encoder, use precomputed embedding cache; False: LoRA fine-tune encoder

# ======================== LoRA Parameters (effective when FROZEN_BACKBONE=False) ========================
LORA_R = 8                                # LoRA rank (dimension of low-rank matrices)
LORA_ALPHA = 16                           # LoRA alpha scaling factor (effective scaling = alpha / r)
LORA_DROPOUT = 0.1                        # LoRA layer dropout ratio
LORA_TARGET_MODULES_STRUCT = [            # Structural encoder LoRA target module names (DeBERTa-v2 attention projection layers)
    "query_proj", "key_proj", "value_proj"
]

# ======================== Structural Stream Pooling Settings ========================
USE_PERIOGT_PROMPT = True                 # True: enable periodic prompt enhancement (PA + pretrained model generates prompt); False: use only node initial chemical features

# LLM: 'last_token' | 'mean_pooling' | 'sum_pooling' | 'attention_pooling' | 'sigmoid_pooling'
# PolyBERT: 'mean_pooling' | 'attention_pooling'; PerioGT uses its own graph readout.
STRUCT_POOLING_TYPE = "mean_pooling"      # For LLM sigmoid gating, set this to 'sigmoid_pooling' (not POOLING_TYPE)
STRUCT_PROJ_DIM = 128                     # LLM attention/sigmoid or PolyBERT attention projection dimension

# ======================== MLP Projection Head Parameters ========================
# Regression head: input_dim -> d1 -> d2 -> 1
MLP_HIDDEN_DIM = [512, 256]             # MLP hidden layer dimensions [d1, d2], final projection to scalar 1
MLP_ACTIVATION = "SiLU"                   # Activation function: 'SiLU' | 'ReLU' | 'GELU'
MLP_DROPOUT = 0.2                         # MLP inter-layer Dropout ratio

# ======================== LLM Structural Stream Parameters (effective when STRUCT_ENCODER='llm') ========================
# Input: chemical_composition.monomers[].{smiles, ratio_value, ratio_unit}
# plus is_homopolymer. Excludes names, pSMILES, additives and processing history.
# Same prediction prompt as Type 9, no chat template; preserves raw ratios/all monomers.
# This matches field selection, not encoder-specific truncation/missing-value handling.
# Precompute via precompute_embeddings.py: INPUT_TYPES = ["struct"]. Type 9 stays unchanged.
MODEL_TYPE = "qwen3_4b_base"              # LLM type: 'qwen3_4b_instruct_2507' | 'qwen3_4b_thinking_2507' | 'qwen3_4b_base' | 'qwen3_8b_base' | 'qwen3_0_6b_base' | 'chemdfm_v1_5_8b'
LORA_TARGET_MODULES_LLM = [              # LLM LoRA target module names (Qwen3 attention and FFN projection layers)
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# ======================== Gaussian Noise Parameters ========================
# Add Gaussian noise to predicted values during training for regularization
NOISE_ENABLED = False                     # Whether to enable Gaussian noise
NOISE_MEAN = 0.0                          # Noise mean
NOISE_STD = 0.1                           # Noise initial standard deviation
NOISE_ANNEAL = True                       # Whether to use cosine annealing to gradually reduce noise std

# ======================== Article-Aware Loss Parameters ========================
# Total loss formula:
#   L = w_mse * L_mse + w_consist * L_consist + w_bias * L_bias + w_rank * L_rank
#
# L_mse = (1/N) * \Sigma (\hat{y}_i - y_i)^2                                           — Standard mean squared error
# L_consist = (1/|A'|) * \Sigma_{a\in A'} Var({\hat{y}_i - y_i | i \in a})                  — Intra-article error variance (A' = articles with >1 samples)
# L_bias = (1/|A|) * \Sigma_{a\in A} ( mean_{i\in a}(\hat{y}_i - y_i) )^2                   — Article mean error squared
# L_rank = (1/|A'|) * \Sigma_{a\in A'} (1/|P_a|) * \Sigma_{(i,j)\in P_a} log(1+exp(-(\hat{y}_i-\hat{y}_j)))  — Intra-article pairwise ranking loss (P_a = pairs with y_i>y_j)

MSE_WEIGHT = 1.0                          # w_mse: MSE loss weight (default 1.0; set to 0 to disable MSE entirely)

ARTICLE_CONSISTENCY_WEIGHT = 0          # w_consist: Article consistency loss weight (0 disables)
ARTICLE_BIAS_WEIGHT = 0                 # w_bias: Article bias loss weight (0 disables)
ARTICLE_RANKING_WEIGHT = 0              # w_rank: Article ranking loss weight (0 disables)

# ======================== Ray Tune Grid Search Parallel Settings ========================
RAY_GPU_FRACTION = 0.1                    # GPU share per trial
MAX_CONCURRENT_TRIALS = 30                # Global max concurrent trial count
RAY_CPU_PER_TRIAL = 1                     # CPU cores allocated per trial
RAY_MAX_FAILURES = 3                      # Max retry attempts for a single trial failure
