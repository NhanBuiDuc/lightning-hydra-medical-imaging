from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, F1Score, Precision, Recall, ConfusionMatrix
from torchmetrics.classification.accuracy import Accuracy
from torchvision.ops.focal_loss import sigmoid_focal_loss
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
from .components.focal_loss import FocalLoss, BinaryFocalLoss
from .components.sensitivity_95 import Sensitivity
from sklearn.metrics import roc_curve, roc_auc_score, auc
import numpy as np
from sklearn.metrics import confusion_matrix


class ResnetModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        criterion: str,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.net = net
        if self.net.num_classes > 2:
            self.task = "multiclass"
        else:
            self.task = "binary"
        # loss function
        if criterion == "entropy":
            if self.net.num_classes > 2:
                self.criterion = torch.nn.CrossEntropyLoss()
            self.criterion = torch.nn.BCELoss()
        elif criterion == "focal":
            if self.net.num_classes > 2:
                self.criterion = FocalLoss(alpha=0.2, gamma=2)
            else:
                self.criterion = BinaryFocalLoss(alpha=0.25, gamma=2)
        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(
            task=self.task, num_classes=self.net.num_classes)
        self.val_acc = Accuracy(
            task=self.task, num_classes=self.net.num_classes)

        self.train_f1 = F1Score(
            task=self.task, num_classes=self.net.num_classes)
        self.val_f1 = F1Score(task=self.task, num_classes=self.net.num_classes)

        self.train_recall = Recall(
            task=self.task, num_classes=self.net.num_classes)
        self.val_recall = Recall(
            task=self.task, num_classes=self.net.num_classes)

        self.train_precision = Precision(
            task=self.task, num_classes=self.net.num_classes)
        self.val_precision = Precision(
            task=self.task, num_classes=self.net.num_classes)

        # self.train_confusion_matrix = ConfusionMatrix(
        #     task=self.task, num_classes=self.net.num_classes)
        # self.val_confusion_matrix = ConfusionMatrix(
        #     task=self.task, num_classes=self.net.num_classes)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        # for tracking best so far validation F1
        # self.val_acc_best = MaxMetric()
        # self.val_f1_best = MaxMetric()
        # self.train_sensitivity_95 = sensitivity(0.95)
        # self.val_sensitivity_95 = Sensitivity(0.95)
        self.val_sensitivity_best = MaxMetric()
        self.pred_list = []
        self.target_list = []
        self.logits_list = []
        self.best_sensitivity = 0
        self.thresh_hold_at_best_sensitivity = 0
        self.auc_at_best_sensitivity = 0
        self.desired_specificity = 0.95

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        # Empty the GPU memory cache
        torch.cuda.empty_cache()
        self.train_loss.reset()
        self.train_recall.reset()
        self.train_precision.reset()
        self.train_acc.reset()

        self.val_sensitivity_best.reset()
        self.val_sensitivity_best(0)
        self.val_loss.reset()
        self.val_recall.reset()
        self.val_precision.reset()
        self.val_acc.reset()
        # self.val_f1_best.reset()
        # self.val_confusion_matrix.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.
        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        if batch is not None:
            x, y = batch

            if self.net.num_classes > 2:
                y = y.view(y.shape[0], -1)
                logits = self.forward(x)
                loss = self.criterion(logits, y)
                preds = torch.argmax(logits, dim=1)
                preds_one_hot = F.one_hot(
                    preds, num_classes=self.net.num_classes)

                ground_truth = torch.argmax(y, dim=1)
                y = F.one_hot(ground_truth, num_classes=self.net.num_classes)
                return loss, preds_one_hot, y
            else:
                y = y.view(y.shape[0], 1)
                logits = self.forward(x)
                loss = self.criterion(logits, y)
                # preds = torch.argmax(logits, dim=1)
                # preds = preds.view(preds.shape[0], 1)
                threshold = 0.5
                binary_predictions = (logits >= threshold).float()
                return loss, logits, binary_predictions, y
        else:
            return None, None, None

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, logits, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.train_f1(preds, targets)
        self.train_recall(preds, targets)
        self.train_precision(preds, targets)

        self.log("train/loss", self.train_loss,
                 on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("train/acc", self.train_acc,
                 on_step=True, on_epoch=False, prog_bar=True,  logger=True)
        self.log("train/f1", self.train_f1,
                 on_step=True, on_epoch=False, prog_bar=True,  logger=True)
        self.log("train/recall", self.train_recall,
                 on_step=True, on_epoch=False, prog_bar=True,  logger=True)
        self.log("train/precision", self.train_precision,
                 on_step=True, on_epoch=False, prog_bar=True,  logger=True)
        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."

        self.train_loss.reset()
        self.train_acc.reset()
        self.train_f1.reset()
        self.train_recall.reset()
        self.train_precision.reset()
        # self.train_sensitivity_95.reset()

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        return super().on_after_batch_transfer(batch, dataloader_idx)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, logits, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.val_f1(preds, targets)
        self.val_recall(preds, targets)
        self.val_precision(preds, targets)
        self.logits_list.append(logits)
        self.pred_list.append(preds)
        self.target_list.append(targets)

    def on_validation_epoch_start(self) -> None:
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_f1.reset()
        self.val_recall.reset()
        self.val_precision.reset()
        self.val_sensitivity_best(self.best_sensitivity)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        # acc = self.val_acc.compute()  # get current val acc
        # update best so far val acc

        merged_preds = torch.cat(self.pred_list, dim=0)
        merged_logits = torch.cat(self.logits_list, dim=0)
        merged_targets = torch.cat(self.target_list, dim=0)

        preds = merged_preds.detach().cpu().numpy()
        logits = merged_logits.detach().cpu().numpy()
        targets = merged_targets.detach().cpu().numpy()
        # Compute the ROC curve
        fpr, tpr, thresholds = roc_curve(targets, logits)
        # Desired specificity

        # Find the index of the threshold that is closest to the desired specificity
        idx = np.argmax(fpr >= (1 - self.desired_specificity))

        # Get the corresponding threshold
        threshold_at_desired_specificity = thresholds[idx]

        # Get the corresponding TPR (sensitivity)
        sensitivity_at_desired_specificity = tpr[idx]

        # Calculate the AUC (Area Under the Curve)
        roc_auc = auc(fpr, tpr)

        target_count_zeros = np.count_nonzero(targets == 0)
        target_count_ones = np.count_nonzero(targets == 1)

        pred_count_zeros = np.count_nonzero(preds == 0)
        pred_count_ones = np.count_nonzero(preds == 1)
        # Get the predicted labels based on the threshold
        predicted_labels = (
            logits >= threshold_at_desired_specificity).astype(int)
        # Compute confusion matrix
        conf_matrix = confusion_matrix(targets, predicted_labels)
        self.log("val/sensitivity", sensitivity_at_desired_specificity,
                 on_step=False, on_epoch=True, prog_bar=True,  logger=True)
        self.log("val/roc_auc", roc_auc,
                 on_step=False, on_epoch=True, prog_bar=True,  logger=True)
        self.log("val/threshold", threshold_at_desired_specificity,
                 on_step=False, on_epoch=True, prog_bar=True,  logger=True)
        self.log("val/length", len(merged_targets),
                 on_step=False, on_epoch=True, prog_bar=True,  logger=True)
        self.log("val/target_count_zeros", target_count_zeros,
                 on_step=False, on_epoch=True, prog_bar=True,  logger=True)
        self.log("val/target_count_ones", target_count_ones,
                 on_step=False, on_epoch=True, prog_bar=True,  logger=True)
        self.log("val/conf_matrix", conf_matrix,
                 on_step=False, on_epoch=True, prog_bar=True,  logger=True)
        # self.log("val/pred_count_zeros", pred_count_zeros,
        #          on_step=False, on_epoch=True, prog_bar=True,  logger=True)
        # self.log("val/pred_count_ones", pred_count_ones,
        #          on_step=False, on_epoch=True, prog_bar=True,  logger=True)

        current_best_sensitivity = self.val_sensitivity_best.compute()

        if sensitivity_at_desired_specificity > current_best_sensitivity:
            self.best_sensitivity = sensitivity_at_desired_specificity
            self.auc_at_best_sensitivity = roc_auc
            self.thresh_hold_at_best_sensitivity = threshold_at_desired_specificity
            self.val_sensitivity_best(sensitivity_at_desired_specificity)

        self.log("val/sensitivity_best", self.val_sensitivity_best.compute(),
                 on_step=False, on_epoch=True, prog_bar=True,  logger=True)
        self.log("val/roc_auc_best", self.auc_at_best_sensitivity,
                 on_step=False, on_epoch=True, prog_bar=True,  logger=True)
        self.log("val/thresh_hold_best", self.thresh_hold_at_best_sensitivity,
                 on_step=False, on_epoch=True, prog_bar=True,  logger=True)

        self.log("val/loss", self.val_loss.compute(),
                 on_step=False, on_epoch=True, prog_bar=True,  logger=True)
        self.log("val/acc", self.val_acc.compute(), on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log("val/f1", self.val_f1.compute(),
                 on_step=False, on_epoch=True, prog_bar=True,  logger=True)
        self.log("val/recall", self.val_recall.compute(),
                 on_step=False, on_epoch=True, prog_bar=True,  logger=True)
        self.log("val/precision", self.val_precision.compute(),
                 on_step=False, on_epoch=True, prog_bar=True,  logger=True)
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        # self.log("val/f1_best", self.val_f1_best.compute(),
        #          sync_dist=True, prog_bar=True)
        self.pred_list = []
        self.target_list = []
        self.logits_list = []

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss.update(loss)
        self.val_acc.update(preds, targets)
        self.val_f1.update(preds, targets)
        self.val_recall.update(preds, targets)
        self.val_precision.update(preds, targets)

        self.log("val/loss", self.val_loss,
                 on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=True,
                 on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1,
                 on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/recall", self.val_recall,
                 on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/precision", self.val_precision,
                 on_step=True, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(
            params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = ResnetModule(None, None, None, None)
