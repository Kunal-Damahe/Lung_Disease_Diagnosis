import os
import sys
from typing import Tuple

import joblib
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from xray.entity.artifacts_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
)
from xray.entity.config_entity import DataTransformationConfig
from xray.exception import XRayException
from xray.logger import logging


class DataTransformation:
    def __init__(
        self,
        data_transformation_config: DataTransformationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
    ):
        self.data_transformation_config = data_transformation_config

        self.data_ingestion_artifact = data_ingestion_artifact

    def transforming_training_data(self) -> transforms.Compose:
        try:
            logging.info(
                "Entered the transforming_training_data method of Data transformation class"
            )

            train_transform: transforms.Compose = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        self.data_transformation_config.CENTERCROP,
                        scale=(0.85, 1.0),
                        ratio=(0.9, 1.1),
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(
                        self.data_transformation_config.RANDOMROTATION
                    ),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ColorJitter(
                        **self.data_transformation_config.color_jitter_transforms
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        **self.data_transformation_config.normalize_transforms
                    ),
                ]
            )

            logging.info(
                "Exited the transforming_training_data method of Data transformation class"
            )

            return train_transform

        except Exception as e:
            raise XRayException(e, sys)

    def transforming_testing_data(self) -> transforms.Compose:
        logging.info(
            "Entered the transforming_testing_data method of Data transformation class"
        )

        try:
            test_transform: transforms.Compose = transforms.Compose(
                [
                    transforms.Resize(self.data_transformation_config.RESIZE),
                    transforms.CenterCrop(self.data_transformation_config.CENTERCROP),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        **self.data_transformation_config.normalize_transforms
                    ),
                ]
            )

            logging.info(
                "Exited the transforming_testing_data method of Data transformation class"
            )

            return test_transform

        except Exception as e:
            raise XRayException(e, sys)

    def data_loader(
        self, train_transform: transforms.Compose, test_transform: transforms.Compose
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        try:
            logging.info("Entered the data_loader method of Data transformation class")

            train_data: Dataset = ImageFolder(
                os.path.join(self.data_ingestion_artifact.train_file_path),
                transform=train_transform,
            )

            validation_data: Dataset = ImageFolder(
                os.path.join(self.data_ingestion_artifact.train_file_path),
                transform=test_transform,
            )

            test_data: Dataset = ImageFolder(
                os.path.join(self.data_ingestion_artifact.test_file_path),
                transform=test_transform,
            )

            logging.info("Created train data and test data paths")

            generator = torch.Generator().manual_seed(
                self.data_transformation_config.random_seed
            )
            train_indices, validation_indices = [], []
            targets = torch.tensor(train_data.targets)
            for class_index in range(len(train_data.classes)):
                class_indices = torch.where(targets == class_index)[0]
                order = torch.randperm(len(class_indices), generator=generator)
                class_indices = class_indices[order].tolist()
                validation_size = max(
                    1,
                    int(
                        len(class_indices)
                        * self.data_transformation_config.validation_split
                    ),
                )
                validation_indices.extend(class_indices[:validation_size])
                train_indices.extend(class_indices[validation_size:])

            train_loader: DataLoader = DataLoader(
                Subset(train_data, train_indices),
                shuffle=True,
                **self.data_transformation_config.data_loader_params,
            )

            validation_loader: DataLoader = DataLoader(
                Subset(validation_data, validation_indices),
                shuffle=False,
                **self.data_transformation_config.data_loader_params,
            )

            test_loader: DataLoader = DataLoader(
                test_data,
                shuffle=False,
                **self.data_transformation_config.data_loader_params,
            )

            print("Class mapping:", train_data.class_to_idx)

            logging.info("Exited the data_loader method of Data transformation class")

            return train_loader, validation_loader, test_loader

        except Exception as e:
            raise XRayException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info(
                "Entered the initiate_data_transformation method of Data transformation class"
            )

            train_transform: transforms.Compose = self.transforming_training_data()

            test_transform: transforms.Compose = self.transforming_testing_data()

            os.makedirs(self.data_transformation_config.artifact_dir, exist_ok=True)

            joblib.dump(
                train_transform, self.data_transformation_config.train_transforms_file
            )

            joblib.dump(
                test_transform, self.data_transformation_config.test_transforms_file
            )

            train_loader, validation_loader, test_loader = self.data_loader(
                train_transform=train_transform, test_transform=test_transform
            )

            data_transformation_artifact: DataTransformationArtifact = DataTransformationArtifact(
                transformed_train_object=train_loader,
                transformed_validation_object=validation_loader,
                transformed_test_object=test_loader,
                train_transform_file_path=self.data_transformation_config.train_transforms_file,
                test_transform_file_path=self.data_transformation_config.test_transforms_file,
            )

            logging.info(
                "Exited the initiate_data_transformation method of Data transformation class"
            )

            return data_transformation_artifact

        except Exception as e:
            raise XRayException(e, sys)
