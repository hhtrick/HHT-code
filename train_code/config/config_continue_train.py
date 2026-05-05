"""
config_continue_train.py — Continue-Train fine-tuning inference configuration
Fine-tune on a small set of new data from a pretrained model and run inference.

Usage:
  1. Specify the pretrained model weights directory in continue_train_inference.py
  2. Point TEST_FILE to a JSON file under val_dataset/ as the test set
  3. Training/validation data is split from continue_train_dataset/{name}/{name}.json
  4. Run: python continue_train_inference.py
"""

# ======================== Continue Training Parameters ========================
CONTINUE_TRAIN_EPOCHS = 30                      # Max training epochs
CONTINUE_TRAIN_PATIENCE = 10                    # Early stopping patience
CONTINUE_TRAIN_LR = 1e-5                        # Fine-tuning LR (typically lower than main training)
CONTINUE_TRAIN_BACKBONE_LR = 1e-7              # Backbone fine-tuning LR (only effective when FROZEN_BACKBONE=False)
CONTINUE_TRAIN_BATCH_SIZE = 16                  # Fine-tuning batch size
CONTINUE_TRAIN_WARMUP_STEPS = 10               # Warmup steps
CONTINUE_TRAIN_GRADIENT_CLIP_VAL = 1.0          # Gradient clipping threshold
CONTINUE_TRAIN_GRADIENT_ACCUMULATION_STEPS = 1  # Gradient accumulation steps (>1 effectively increases batch_size)
CONTINUE_TRAIN_MLP_DROPOUT = None               # MLP dropout rate (None = use original model config; set to float to override)
RANDOM_INIT_WEIGHTS = False                     # Randomly initialize model weights and scaler (True: skip pretrained weights, refit scaler from training data)

# ======================== Data Source (allocation from continue_train_dataset) ========================
# Training and validation data all come from continue_train_dataset/{name}/{name}.json,
# and the split is determined by the pre-generated
# continue_train_dataset/{name}/split_continue_train.pkl (containing "train" and "val" index lists).
# If the val list in the pkl is empty (VAL_RATIO=0.0), no validation set is used during training.
CONTINUE_TRAIN_SPLIT_PKL_NAME = "split_continue_train.pkl"

# ======================== Article-Aware Loss Parameters (independently adjustable during continue training) ========================
#   L = w_mse * L_mse + w_consist * L_consist + w_bias * L_bias + w_rank * L_rank
MSE_WEIGHT = 1.0                          # w_mse: MSE loss weight
ARTICLE_CONSISTENCY_WEIGHT = 0.2          # w_consist: Article consistency loss weight
ARTICLE_BIAS_WEIGHT = 0.2                 # w_bias: Article bias loss weight
ARTICLE_RANKING_WEIGHT = 0.2              # w_rank: Article ranking loss weight

# ======================== Gaussian Noise Parameters ========================
NOISE_ENABLED = False                     # Noise is typically not added during continue training
NOISE_MEAN = 0.0
NOISE_STD = 0.05
NOISE_ANNEAL = True
