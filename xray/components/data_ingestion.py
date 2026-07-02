import sys
import os

from xray.clould_storage.s3_operations import S3Operation
from xray.constant.training_pipeline import *
from xray.entity.artifacts_entity import DataIngestionArtifact
from xray.entity.config_entity import DataIngestionConfig
from xray.exception import XRayException
from xray.logger import logging


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config
        self.s3 = S3Operation()

    def _dataset_is_valid(self) -> bool:
        required_directories = (
            os.path.join(self.data_ingestion_config.train_data_path, CLASS_LABEL_1),
            os.path.join(self.data_ingestion_config.train_data_path, CLASS_LABEL_2),
            os.path.join(self.data_ingestion_config.test_data_path, CLASS_LABEL_1),
            os.path.join(self.data_ingestion_config.test_data_path, CLASS_LABEL_2),
        )
        return all(
            os.path.isdir(directory) and any(os.scandir(directory))
            for directory in required_directories
        )

    def get_data_from_s3(self) -> None:
        try:
            logging.info("Entered the get_data_from_s3 method")

            if self._dataset_is_valid():
                logging.info("Valid dataset exists locally. Skipping S3 download.")
                print("Valid local dataset found. Skipping S3 download.")
                return

            if self.data_ingestion_config.using_local_data:
                raise FileNotFoundError(
                    "Invalid dataset structure at "
                    f"'{self.data_ingestion_config.data_path}'. Expected non-empty "
                    "train/NORMAL, train/PNEUMONIA, test/NORMAL, and "
                    "test/PNEUMONIA folders."
                )

            # ✅ CREATE DIRECTORY IF NOT EXISTS
            os.makedirs(self.data_ingestion_config.data_path, exist_ok=True)

            logging.info("⬇️ Downloading data from S3...")

            self.s3.sync_folder_from_s3(
                folder=self.data_ingestion_config.data_path,
                bucket_name=self.data_ingestion_config.bucket_name,
                bucket_folder_name=self.data_ingestion_config.s3_data_folder,
            )

            if not self._dataset_is_valid():
                raise RuntimeError(
                    "The S3 download did not produce a valid dataset. Configure valid "
                    "AWS credentials, or run: python train.py --data-dir <dataset-folder>"
                )

            logging.info("✅ Data downloaded successfully from S3")

        except Exception as e:
            raise XRayException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        logging.info("Entered initiate_data_ingestion method")

        try:
            self.get_data_from_s3()

            data_ingestion_artifact: DataIngestionArtifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.train_data_path,
                test_file_path=self.data_ingestion_config.test_data_path,
            )

            logging.info("Exited initiate_data_ingestion method")

            return data_ingestion_artifact

        except Exception as e:
            raise XRayException(e, sys)
