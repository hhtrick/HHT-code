"""Supplementary Note 14: fixed E_article test set, random train/validation.

Select with CONTINUE_CONFIG_MODULE in continue_train_inference.py.
Use DATASET_NAME="E_article" and TEST_FILE="val_dataset/val_E_article_1.json".
The test JSON and cache namespaces are the same as the original article split.
"""
from config.config_continue_train import *

CONTINUE_TRAIN_SPLIT_PKL_NAME = "split_holdout_whole.pkl"
CONTINUE_TRAIN_EPOCHS = 100
CONTINUE_TRAIN_PATIENCE = 10
CONTINUE_TRAIN_LR = 1e-4  # Same for both initialization conditions in this experiment.
CONTINUE_TRAIN_BACKBONE_LR = 1e-7
CONTINUE_TRAIN_BATCH_SIZE = 16
CONTINUE_TRAIN_WARMUP_STEPS = 10
CONTINUE_TRAIN_GRADIENT_CLIP_VAL = 1.0
CONTINUE_TRAIN_GRADIENT_ACCUMULATION_STEPS = 1
CONTINUE_TRAIN_MLP_DROPOUT = None
RANDOM_INIT_WEIGHTS = False  # Set True for the random-initialization condition.
LOG_TARGET_DATASETS = None  # Inherit the main-task configuration (reported runs use []).
MSE_WEIGHT = 1.0
ARTICLE_CONSISTENCY_WEIGHT = 0
ARTICLE_BIAS_WEIGHT = 0
ARTICLE_RANKING_WEIGHT = 0
NOISE_ENABLED = False
