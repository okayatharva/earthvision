
from pathlib import Path

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

# Dataset configuration
DATASET_NAME = "EuroSAT"
IMAGE_SIZE = 64
BATCH_SIZE = 32
NUM_WORKERS = 0
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# Training configuration
EPOCHS = 10
LEARNING_RATE = 0.001

# RGB image normalization
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Output files
MODEL_PATH = MODEL_DIR / "earthvision_cnn.pth"
METRICS_PATH = REPORT_DIR / "metrics.json"
CONFUSION_MATRIX_PATH = REPORT_DIR / "confusion_matrix.png"
