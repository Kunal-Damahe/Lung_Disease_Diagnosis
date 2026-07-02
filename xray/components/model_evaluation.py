import sys
from typing import Tuple

import torch
from torch.nn import CrossEntropyLoss, Module
from torch.utils.data import DataLoader

from xray.entity.artifacts_entity import (
    DataTransformationArtifact,
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
)
from xray.entity.config_entity import ModelEvaluationConfig
from xray.exception import XRayException
from xray.logger import logging
from xray.ml.model.arch import Net


class ModelEvaluation:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_evaluation_config: ModelEvaluationConfig,
        model_trainer_artifact: ModelTrainerArtifact,
    ):

        self.data_transformation_artifact = data_transformation_artifact

        self.model_evaluation_config = model_evaluation_config

        self.model_trainer_artifact = model_trainer_artifact

    def configuration(self) -> Tuple[DataLoader, Module, Module]:
        logging.info("Entered the configuration method of Model evaluation class")

        try:
            test_dataloader: DataLoader = (
                self.data_transformation_artifact.transformed_test_object
            )

            model: Module = Net(pretrained=False)
            checkpoint = torch.load(
                self.model_trainer_artifact.trained_model_path,
                map_location=self.model_evaluation_config.device,
            )
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state_dict)
            self.decision_threshold = float(checkpoint.get("decision_threshold", 0.5))

            model.to(self.model_evaluation_config.device)

            cost: Module = CrossEntropyLoss()

            model.eval()

            logging.info("Exited the configuration method of Model evaluation class")

            return test_dataloader, model, cost

        except Exception as e:
            raise XRayException(e, sys)

    def test_net(self) -> float:
        logging.info("Entered the test_net method of Model evaluation class")

        try:
            test_dataloader, net, cost = self.configuration()

            self.model_evaluation_config.test_loss = 0.0
            self.model_evaluation_config.test_accuracy = 0.0
            self.model_evaluation_config.total = 0
            self.model_evaluation_config.total_batch = 0
            confusion_matrix = torch.zeros(2, 2, dtype=torch.long)

            with torch.no_grad():
                for _, data in enumerate(test_dataloader):
                    images = data[0].to(self.model_evaluation_config.device)

                    labels = data[1].to(self.model_evaluation_config.device)

                    output = net(images)

                    loss = cost(output, labels)

                    pneumonia_probability = torch.softmax(output, dim=1)[:, 1]
                    predictions = (
                        pneumonia_probability >= self.decision_threshold
                    ).long()

                    for actual, predicted in zip(
                        labels.detach().cpu(), predictions.detach().cpu()
                    ):
                        confusion_matrix[actual, predicted] += 1

                    logging.info(
                        f"Actual_Labels : {labels}     Predictions : {predictions}     labels : {loss.item():.4f}"
                    )

                    self.model_evaluation_config.test_loss += loss.item()

                    self.model_evaluation_config.test_accuracy += (
                        (predictions == labels).sum().item()
                    )

                    self.model_evaluation_config.total_batch += 1

                    self.model_evaluation_config.total += labels.size(0)

                    logging.info(
                        f"Model  -->   Loss : {self.model_evaluation_config.test_loss/ self.model_evaluation_config.total_batch} Accuracy : {(self.model_evaluation_config.test_accuracy / self.model_evaluation_config.total) * 100} %"
                    )

            accuracy = (
                self.model_evaluation_config.test_accuracy
                / self.model_evaluation_config.total
            ) * 100

            class_totals = confusion_matrix.sum(dim=1).clamp_min(1)
            recalls = confusion_matrix.diag().float() / class_totals.float()
            balanced_accuracy = 100.0 * recalls.mean().item()
            print(f"Test confusion matrix: {confusion_matrix.tolist()}")
            print(
                f"NORMAL recall: {100 * recalls[0]:.2f}% | "
                f"PNEUMONIA recall: {100 * recalls[1]:.2f}% | "
                f"Balanced accuracy: {balanced_accuracy:.2f}%"
            )

            logging.info("Exited the test_net method of Model evaluation class")

            return accuracy

        except Exception as e:
            raise XRayException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        logging.info(
            "Entered the initiate_model_evaluation method of Model evaluation class"
        )

        try:
            accuracy = self.test_net()

            model_evaluation_artifact: ModelEvaluationArtifact = (
                ModelEvaluationArtifact(model_accuracy=accuracy)
            )

            logging.info(
                "Exited the initiate_model_evaluation method of Model evaluation class"
            )

            return model_evaluation_artifact

        except Exception as e:
            raise XRayException(e, sys)
