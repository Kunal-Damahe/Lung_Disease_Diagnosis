import os
from datetime import datetime
from typing import List
import torch

TIMESTAMP: datetime = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

# =========================
# Data Ingestion Constants
# =========================
ARTIFACT_DIR: str = "artifacts"
DATA_CACHE_DIR: str = "data_cache"

BUCKET_NAME: str = "lungdiseasediagnosis"
S3_DATA_FOLDER: str = "Data"

CLASS_LABEL_1: str = "NORMAL"
CLASS_LABEL_2: str = "PNEUMONIA"

# =========================
# Data Transformation Constants
# =========================
BRIGHTNESS: float = 0.10
CONTRAST: float = 0.10
SATURATION: float = 0.0
HUE: float = 0.0

RESIZE: int = 256
CENTERCROP: int = 224
RANDOMROTATION: int = 7
VALIDATION_SPLIT: float = 0.10
RANDOM_SEED: int = 42

# Normalization (VERY IMPORTANT for ResNet)
NORMALIZE_LIST_1: List[float] = [0.485, 0.456, 0.406]
NORMALIZE_LIST_2: List[float] = [0.229, 0.224, 0.225]

TRAIN_TRANSFORMS_KEY: str = "xray_train_transforms"
TRAIN_TRANSFORMS_FILE: str = "train_transforms.pkl"
TEST_TRANSFORMS_FILE: str = "test_transforms.pkl"

# =========================
# Data Loader Params
# =========================
BATCH_SIZE: int = 64
SHUFFLE: bool = True         # 🔥 FIXED (was False)
PIN_MEMORY: bool = True
NUM_WORKERS: int = 0

# =========================
# Model Training Constants
# =========================
TRAINED_MODEL_DIR: str = "trained_model"
TRAINED_MODEL_NAME: str = "model.pt"

DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learning Rate Scheduler (Improved)
STEP_SIZE: int = 5
GAMMA: float = 0.2

EPOCH: int = 5
LEARNING_RATE: float = 1e-4
CLASSIFIER_LEARNING_RATE: float = 5e-4
WEIGHT_DECAY: float = 1e-4
EARLY_STOPPING_PATIENCE: int = 2

# =========================
# BentoML Deployment
# =========================
BENTOML_MODEL_NAME: str = "xray_model"
BENTOML_SERVICE_NAME: str = "xray_service"
BENTOML_ECR_IMAGE: str = "xray_bento_image"
ECR_REGISTRY: str = os.getenv("ECR_REGISTRY", "")
AWS_REGION: str = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

# =========================
# Prediction Mapping
# =========================
PREDICTION_LABEL: dict = {"0": CLASS_LABEL_1, 1: CLASS_LABEL_2}
