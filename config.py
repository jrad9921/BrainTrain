"""
Simple configuration file for brain MRI classification training
"""
import os 
# ============================================================================
# BASIC SETTINGS
# ============================================================================

COLUMN_NAME = 'age'
CSV_NAME = 'dlbs'
TRAINING_MODE = 'linear'  # Options: 'sfcn', 'dense', 'linear', 'ssl-finetuned', 'lora'
TASK = 'regression'

# ============================================================================
# DATA PATHS
# ============================================================================

TRAIN_COHORT = 'ukb'
TEST_COHORT = 'ukb'

CSV_TRAIN = f'/mnt/bulk-neptune/radhika/project/data/{TRAIN_COHORT}/train/{CSV_NAME}.csv'
CSV_VAL = f'/mnt/bulk-neptune/radhika/project/data/{TRAIN_COHORT}/val/{CSV_NAME}.csv'
CSV_TEST = f'/mnt/bulk-neptune/radhika/project/data/{TEST_COHORT}/test/{CSV_NAME}.csv'

TENSOR_DIR = f'/mnt/bulk-neptune/radhika/project/images/{TRAIN_COHORT}/npy96'
TENSOR_DIR_TEST = f'/mnt/bulk-neptune/radhika/project/images/{TEST_COHORT}/npy96'

# ============================================================================
# MODEL SETTINGS
# ============================================================================
IMG_SIZE = 96
N_CHANNELS = 1
N_CLASSES = 2

# LoRA Parameters
LORA_RANK = 16
LORA_ALPHA = 32
LORA_TARGET_MODULES = ['feature_extractor.conv_']

# SSL Pretrained Model
SSL_COHORT = 'ukb-nako'
SSL_BATCH_SIZE = 16
SSL_EPOCHS = 1000
PRETRAINED_MODEL = (f'/mnt/bulk-neptune/radhika/project/models/ssl/sfcn/{SSL_COHORT}/'
                   f'{SSL_COHORT}{IMG_SIZE}/final_model_b{SSL_BATCH_SIZE}_e{SSL_EPOCHS}.pt')

# Transformer Parameters (for Swin/ViT)
PATCH_SIZE = [8, 8, 8]
WINDOW_SIZE = [16, 16, 16]
NUM_HEADS = [3, 6, 12, 24]
DEPTHS = [2, 2, 2, 2]
FEATURE_SIZE = 96

# ============================================================================
# TRAINING SETTINGS
# ============================================================================
BATCH_SIZE = 32
NUM_EPOCHS = 1000
LEARNING_RATE = 0.1
NUM_WORKERS = 8
DEVICE = "cuda:1"
SEED = 42
NROWS = None  # Set to None to use all data, or int for subset

# Early Stopping
PATIENCE = 20

# Learning Rate Scheduler
SCHEDULER_MODE = 'min'
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 3

# ============================================================================
# OUTPUT PATHS
# ============================================================================
# Experiment name
EXPERIMENT_NAME = f"{CSV_NAME}_e{NUM_EPOCHS}_b{BATCH_SIZE}_lr{LEARNING_RATE}_im{IMG_SIZE}"

# Output directories
SAVE_MODEL_DIR = f'/mnt/bulk-neptune/radhika/project/models/{TRAINING_MODE}'
SCORES_TRAIN_DIR = f'/mnt/bulk-neptune/radhika/project/scores/{TRAINING_MODE}/train/{TRAIN_COHORT}'
SCORES_VAL_DIR = f'/mnt/bulk-neptune/radhika/project/scores/{TRAINING_MODE}/val/{TRAIN_COHORT}'
SCORES_TEST_DIR = f'/mnt/bulk-neptune/radhika/project/scores/{TRAINING_MODE}/test/{TEST_COHORT}'
TRAINLOG_DIR = f'/mnt/bulk-neptune/radhika/project/logs/trainlog/{TRAINING_MODE}'
VALLOG_DIR = f'/mnt/bulk-neptune/radhika/project/logs/vallog/{TRAINING_MODE}'
TIMELOG_DIR = f'/mnt/bulk-neptune/radhika/project/logs/timelog/{TRAINING_MODE}'
EVALUATION_DIR = f'/mnt/bulk-neptune/radhika/project/logs/'
MODEL_PATH = os.path.join(SAVE_MODEL_DIR, f'{EXPERIMENT_NAME}.pth')
  
