"""
Simple configuration file for brain MRI classification training
"""
import argparse

parser = argparse.ArgumentParser(description='run training')
parser.add_argument('-c', '--column', type=str, nargs='*', default=None, help='Select one  target column for trainig')
parser.add_argument('-m', '--mode', type=str, nargs='*', default=None, help='Select one of sfcn, dense, linear, ssl-finetuned, lora')
parser.add_argument('-g', '--gpu', type=str, nargs='*', default=None, help='Select one of sfcn, dense, linear, ssl-finetuned, lora')
args = parser.parse_args()

# ============================================================================
# BASIC SETTINGS
# ============================================================================

COLUMN_NAME = 'ms'
CSV_NAME = 'ms-cn_balanced'

TRAINING_MODE = 'sfcn'  # Options: 'sfcn', 'dense', 'linear', 'ssl-finetuned', 'lora'
TASK = 'classification'

# ============================================================================
# DATA PATHS
# ============================================================================

TRAIN_COHORT = 'mspaths'
TEST_COHORT = 'BrainLat'

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

# LoRA Parameters∂
LORA_RANK = 16
LORA_ALPHA = 32
LORA_TARGET_MODULES = ['feature_extractor.conv_']

# SSL Pretrained Model
SSL_COHORT = 'ukb-nako'
SSL_BATCH_SIZE = 16
SSL_EPOCHS = 1000
PRETRAINED_MODEL = (f'/mnt/bulk-neptune/radhika/project/models/ssl/sfcn/{SSL_COHORT}/'
                   f'{SSL_COHORT}{IMG_SIZE}/final_model_b{SSL_BATCH_SIZE}_e{SSL_EPOCHS}.pt')


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
EXPERIMENT_NAME = f"{CSV_NAME}_b{BATCH_SIZE}_im{IMG_SIZE}"

# Output directories
MODEL_DIR = f'/mnt/bulk-neptune/radhika/project/models'
SCORES_DIR = f'/mnt/bulk-neptune/radhika/project/scores'
LOG_DIR = f'/mnt/bulk-neptune/radhika/project/logs'
EVALUATION_DIR = f'/mnt/bulk-neptune/radhika/project/evaluations'
EXPLAINABILITY_DIR = f"/mnt/bulk-neptune/radhika/project/explainability"
  
# ============================================================================
# HEATMAP CONFIGURATION 
# ============================================================================
HEATMAP_MODE = 'top_individual'  # Options: 'single', 'average', 'top_individual'
HEATMAP_TOP_N = 10
ATTENTION_METHOD = 'saliency'  # Options: 'saliency', 'gradcam'
ATTENTION_MODE = 'magnitude'  # Options: 'magnitude', 'signed'
ATTENTION_TARGET = 'logit_diff'  # Options: 'logit_diff', 'pred', 'target_class'
ATTENTION_CLASS_IDX = None
ATLAS_PATH = 'atlas_resampled_96.nii.gz'


