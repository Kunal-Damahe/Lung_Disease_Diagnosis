import os
import sys

# import bentoml
import torch
from torch.nn import CrossEntropyLoss, Module
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import StepLR, _LRScheduler
from tqdm import tqdm

from xray.constant.training_pipeline import *
from xray.entity.artifacts_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from xray.entity.config_entity import ModelTrainerConfig
from xray.exception import XRayException
from xray.logger import logging
from xray.ml.model.arch import Net


class ModelTrainer:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.model_trainer_config: ModelTrainerConfig = model_trainer_config

        self.data_transformation_artifact: DataTransformationArtifact = (
            data_transformation_artifact
        )

        torch.manual_seed(RANDOM_SEED)
        self.model: Module = Net()

    def class_weights(self) -> torch.Tensor:
        """Return inverse-frequency weights for the training subset."""
        subset = self.data_transformation_artifact.transformed_train_object.dataset
        if hasattr(subset, "indices") and hasattr(subset.dataset, "targets"):
            labels = torch.tensor(
                [subset.dataset.targets[index] for index in subset.indices],
                dtype=torch.long,
            )
        elif hasattr(subset, "targets"):
            labels = torch.tensor(subset.targets, dtype=torch.long)
        else:
            raise ValueError("Training dataset does not expose class labels")

        counts = torch.bincount(labels, minlength=2).float()
        weights = labels.numel() / (len(counts) * counts.clamp_min(1))
        print(f"Training class counts: {counts.int().tolist()}")
        print(f"Loss class weights: {weights.tolist()}")
        return weights.to(self.model_trainer_config.device)

    def train(self, optimizer: Optimizer, criterion: Module):
        """
        Description: To train the model

        input: model,device,train_loader,optimizer,epoch

        output: loss, batch id and accuracy
        """
        logging.info("Entered the train method of Model trainer class")

        try:
            self.model.train()

            pbar = tqdm(self.data_transformation_artifact.transformed_train_object)

            correct, processed = 0, 0
            running_loss = 0.0

            for batch_idx, (data, target) in enumerate(pbar):
                data = data.to(self.model_trainer_config.device)
                target = target.to(self.model_trainer_config.device)

                # Initialization of gradient
                optimizer.zero_grad()

                # In PyTorch, gradient is accumulated over backprop and even though thats used in RNN generally not used in CNN
                # or specific requirements
                ## prediction on data

                y_pred = self.model(data)

                # Calculating loss given the prediction
                loss = criterion(y_pred, target)

                # Backprop
                loss.backward()

                optimizer.step()

                # Select the class with the highest logit.
                pred = y_pred.argmax(dim=1, keepdim=True)

                correct += pred.eq(target.view_as(pred)).sum().item()

                processed += len(data)
                running_loss += loss.item() * len(data)

                pbar.set_description(
                    desc=f"Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}"
                )

            accuracy = 100.0 * correct / processed
            return running_loss / processed, accuracy

        except Exception as e:
            raise XRayException(e, sys)

    def test(self, data_loader, criterion: Module):
        try:
            """
            Description: To test the model

            input: model, DEVICE, test_loader

            output: average loss and accuracy

            """
            logging.info("Entered the test method of Model trainer class")

            self.model.eval()

            test_loss: float = 0.0

            probability_batches = []
            label_batches = []

            with torch.no_grad():
                for data, target in data_loader:
                    data = data.to(self.model_trainer_config.device)
                    target = target.to(self.model_trainer_config.device)

                    output = self.model(data)

                    test_loss += criterion(output, target).item() * len(data)

                    target_cpu = target.detach().cpu()
                    probability_batches.append(
                        torch.softmax(output, dim=1)[:, 1].detach().cpu()
                    )
                    label_batches.append(target_cpu)

                dataset_size = len(data_loader.dataset)
                test_loss /= dataset_size
                probabilities = torch.cat(probability_batches)
                labels = torch.cat(label_batches)
                best_threshold = 0.5
                best_balanced_accuracy = -1.0
                best_accuracy = -1.0
                best_correct = 0
                for threshold in torch.linspace(0.05, 0.99, 95):
                    predictions = (probabilities >= threshold).long()
                    class_recalls = []
                    for class_index in range(2):
                        mask = labels == class_index
                        class_recalls.append(
                            (predictions[mask] == class_index).float().mean()
                        )
                    balanced = torch.stack(class_recalls).mean().item() * 100.0
                    calibrated_correct = (predictions == labels).sum().item()
                    calibrated_accuracy = 100.0 * calibrated_correct / dataset_size
                    if (balanced, calibrated_accuracy) > (
                        best_balanced_accuracy,
                        best_accuracy,
                    ):
                        best_balanced_accuracy = balanced
                        best_accuracy = calibrated_accuracy
                        best_correct = calibrated_correct
                        best_threshold = float(threshold)

                print(
                    "Evaluation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
                        test_loss,
                        best_correct,
                        len(
                            data_loader.dataset
                        ),
                        best_accuracy,
                    )
                )

            logging.info(
                "Evaluation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
                    test_loss,
                    best_correct,
                    len(
                        data_loader.dataset
                    ),
                    best_accuracy,
                )
            )

            return (
                test_loss,
                best_accuracy,
                best_balanced_accuracy,
                best_threshold,
            )

        except Exception as e:
            raise XRayException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info(
                "Entered the initiate_model_trainer method of Model trainer class"
            )

            model: Module = self.model.to(self.model_trainer_config.device)

            criterion = CrossEntropyLoss(weight=self.class_weights())

            optimizer: Optimizer = AdamW(
                [
                    {
                        "params": model.model.layer4.parameters(),
                        "lr": self.model_trainer_config.optimizer_params["lr"],
                    },
                    {
                        "params": model.model.fc.parameters(),
                        "lr": self.model_trainer_config.classifier_learning_rate,
                    },
                ],
                weight_decay=self.model_trainer_config.optimizer_params["weight_decay"],
            )

            scheduler: _LRScheduler = StepLR(
                optimizer=optimizer, **self.model_trainer_config.scheduler_params
            )

            best_validation_accuracy = 0.0
            best_balanced_accuracy = 0.0
            best_threshold = 0.5
            best_state = None
            epochs_without_improvement = 0

            for epoch in range(1, self.model_trainer_config.epochs + 1):
                print("Epoch : ", epoch)

                train_loss, train_accuracy = self.train(
                    optimizer=optimizer, criterion=criterion
                )
                (
                    validation_loss,
                    validation_accuracy,
                    balanced_accuracy,
                    decision_threshold,
                ) = self.test(
                    self.data_transformation_artifact.transformed_validation_object,
                    criterion,
                )
                print(
                    f"Train loss: {train_loss:.4f}, accuracy: {train_accuracy:.2f}% | "
                    f"Validation loss: {validation_loss:.4f}, "
                    f"accuracy: {validation_accuracy:.2f}%, "
                    f"balanced accuracy: {balanced_accuracy:.2f}%, "
                    f"threshold: {decision_threshold:.2f}"
                )

                if balanced_accuracy > best_balanced_accuracy:
                    best_validation_accuracy = validation_accuracy
                    best_balanced_accuracy = balanced_accuracy
                    best_threshold = decision_threshold
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                scheduler.step()
                if (
                    epochs_without_improvement
                    >= self.model_trainer_config.early_stopping_patience
                ):
                    print("Early stopping: validation accuracy stopped improving.")
                    break

            if best_state is not None:
                model.load_state_dict(best_state)

            os.makedirs(self.model_trainer_config.artifact_dir, exist_ok=True)

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "validation_accuracy": best_validation_accuracy,
                    "validation_balanced_accuracy": best_balanced_accuracy,
                    "decision_threshold": best_threshold,
                },
                self.model_trainer_config.trained_model_path,
            )


            model_trainer_artifact: ModelTrainerArtifact = ModelTrainerArtifact(
                trained_model_path=self.model_trainer_config.trained_model_path
            )

            logging.info(
                "Exited the initiate_model_trainer method of Model trainer class"
            )

            return model_trainer_artifact

        except Exception as e:
            raise XRayException(e, sys)
