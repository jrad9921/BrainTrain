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

COLUMN_NAME = 'label-name'
CSV_NAME = 'csv-name'

TRAINING_MODE = 'sfcn'  # Options: 'sfcn', 'dense', 'linear', 'ssl-finetuned', 'lora'
TASK = 'classification'

# ============================================================================
# DATA PATHS
# ============================================================================

TRAIN_COHORT = 'cohort-name'
TEST_COHORT = 'cohort-name'

CSV_TRAIN = 'path/to/your/file'
CSV_VAL = 'path/to/your/file'
CSV_TEST = 'path/to/your/file'

TENSOR_DIR = 'path/to/your/folder'
TENSOR_DIR_TEST = 'path/to/your/folder'

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
SSL_COHORT = 'cohort-name'
SSL_BATCH_SIZE = 16
SSL_EPOCHS = 1000
PRETRAINED_MODEL = 'path/to/your/file'


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
MODEL_DIR = 'path/to/your/folder'
SCORES_DIR = 'path/to/your/folder'
LOG_DIR = 'path/to/your/folder'
EVALUATION_DIR = 'path/to/your/folder'
EXPLAINABILITY_DIR = 'path/to/your/folder'
KAPLAN_MEIER = False
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


