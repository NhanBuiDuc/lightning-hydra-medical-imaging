from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, F1Score, Precision, Recall, ConfusionMatrix
from torchmetrics.classification.accuracy import Accuracy
from torchvision.ops.focal_loss import sigmoid_focal_loss
import matplotlib.pyplot as plt


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
        if criterion == "Entropy":
            if self.net.num_classes > 2:
                self.criterion = torch.nn.CrossEntropyLoss()
            self.criterion = torch.nn.BCELoss()
        elif criterion == "Focal":
            self.criterion = sigmoid_focal_loss
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

        self.train_confusion_matrix = ConfusionMatrix(
            task=self.task, num_classes=self.net.num_classes)
        self.val_confusion_matrix = ConfusionMatrix(
            task=self.task, num_classes=self.net.num_classes)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        # for tracking best so far validation F1
        # self.val_acc_best = MaxMetric()
        self.val_f1_best = MaxMetric()

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
        self.val_loss.reset()
        self.val_recall.reset()
        self.val_precision.reset()
        self.val_acc.reset()
        self.val_f1_best.reset()

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
            x = x.to(dtype=torch.float)
            y = y.unsqueeze(-1).to(dtype=torch.float)
            logits = self.forward(x)
            loss = self.criterion(logits, y)
            preds = torch.argmax(logits, dim=1)
            return loss, preds, y
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
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.train_f1(preds, targets)
        self.train_recall(preds, targets)
        self.train_precision(preds, targets)
        self.train_confusion_matrix(preds, targets)

        self.log("train/loss", self.train_loss.compute(),
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc.compute(), on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log("train/f1", self.train_f1.compute(),
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/recall", self.train_recall.compute(),
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/precision", self.train_precision.compute(),
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/confusion_matrix", self.train_confusion_matrix.compute(),
                 on_step=False, on_epoch=True, prog_bar=True)
        self.train_confusion_matrix.plot()
        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.val_f1(preds, targets)
        self.val_recall(preds, targets)
        self.val_precision(preds, targets)
        self.val_confusion_matrix(preds, targets)

        self.log("val/loss", self.val_loss.compute(),
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc.compute(), on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1.compute(),
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/recall", self.val_recall.compute(),
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/precision", self.val_precision.compute(),
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/confusion_matrix", self.val_confusion_matrix.compute(),
                 on_step=False, on_epoch=True, prog_bar=True)
        self.train_confusion_matrix.plot()

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        # acc = self.val_acc.compute()  # get current val acc
        f1 = self.val_f1.compute()
        self.val_f1_best.update(f1)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/f1_best", self.val_f1_best.compute(),
                 sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.val_f1(preds, targets)
        self.val_recall(preds, targets)
        self.val_precision(preds, targets)

        self.log("val/loss", self.val_loss.compute(),
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc.compute(), on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1.compute(),
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/recall", self.val_recall.compute(),
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/precision", self.val_precision.compute(),
                 on_step=False, on_epoch=True, prog_bar=True)

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
