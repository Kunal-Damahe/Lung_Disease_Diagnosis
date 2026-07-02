import os
from dataclasses import dataclass
from torch import device
from xray.constant.training_pipeline import *


# =========================
# Data Ingestion Config
# =========================
@dataclass
class DataIngestionConfig:
    def __init__(self):
        self.s3_data_folder: str = S3_DATA_FOLDER
        self.bucket_name: str = BUCKET_NAME

        self.artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)

        local_data_dir = os.getenv("XRAY_DATA_DIR")
        self.data_path: str = (
            os.path.abspath(local_data_dir)
            if local_data_dir
            else os.path.join(
                ARTIFACT_DIR, DATA_CACHE_DIR, self.s3_data_folder
            )
        )
        self.using_local_data: bool = bool(local_data_dir)

        self.train_data_path: str = os.path.join(self.data_path, "train")
        self.test_data_path: str = os.path.join(self.data_path, "test")


# =========================
# Data Transformation Config
# =========================
@dataclass
class DataTransformationConfig:
    def __init__(self):

        self.color_jitter_transforms: dict = {
            "brightness": BRIGHTNESS,
            "contrast": CONTRAST,
            "saturation": SATURATION,
            "hue": HUE,
        }

        self.RESIZE: int = RESIZE
        self.CENTERCROP: int = CENTERCROP
        self.RANDOMROTATION: int = RANDOMROTATION
        self.validation_split: float = VALIDATION_SPLIT
        self.random_seed: int = RANDOM_SEED

        self.normalize_transforms: dict = {
            "mean": NORMALIZE_LIST_1,
            "std": NORMALIZE_LIST_2,
        }

        self.data_loader_params: dict = {
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "pin_memory": PIN_MEMORY and DEVICE.type == "cuda",
        }

        self.artifact_dir: str = os.path.join(
            ARTIFACT_DIR, TIMESTAMP, "data_transformation"
        )

        self.train_transforms_file: str = os.path.join(
            self.artifact_dir, TRAIN_TRANSFORMS_FILE
        )

        self.test_transforms_file: str = os.path.join(
            self.artifact_dir, TEST_TRANSFORMS_FILE
        )


# =========================
# Model Trainer Config
# =========================
@dataclass
class ModelTrainerConfig:
    def __init__(self):

        self.artifact_dir: str = os.path.join(
            ARTIFACT_DIR, TIMESTAMP, "model_training"
        )

        self.trained_bentoml_model_name: str = "xray_model"

        self.trained_model_path: str = os.path.join(
            self.artifact_dir, TRAINED_MODEL_NAME
        )

        self.train_transforms_key: str = TRAIN_TRANSFORMS_KEY

        self.epochs: int = EPOCH

        # 🔥 FIXED OPTIMIZER
        self.optimizer_params: dict = {
            "lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
        }
        self.classifier_learning_rate: float = CLASSIFIER_LEARNING_RATE

        # 🔥 FIXED SCHEDULER
        self.scheduler_params: dict = {
            "step_size": STEP_SIZE,
            "gamma": GAMMA
        }

        self.device: device = DEVICE
        self.early_stopping_patience: int = EARLY_STOPPING_PATIENCE


# =========================
# Model Evaluation Config
# =========================
@dataclass
class ModelEvaluationConfig:
    def __init__(self):
        self.device: device = DEVICE
        self.test_loss: float = 0.0
        self.test_accuracy: float = 0.0
        self.total: int = 0
        self.total_batch: int = 0


# =========================
# Model Pusher Config
# =========================
@dataclass
class ModelPusherConfig:
    def __init__(self):
        self.bentoml_model_name: str = BENTOML_MODEL_NAME
        self.bentoml_service_name: str = BENTOML_SERVICE_NAME
        self.train_transforms_key: str = TRAIN_TRANSFORMS_KEY
        self.bentoml_ecr_image: str = BENTOML_ECR_IMAGE
        self.ecr_registry: str = ECR_REGISTRY
        self.aws_region: str = AWS_REGION
