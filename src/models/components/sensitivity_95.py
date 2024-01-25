from torchmetrics import Metric
import torch
from torch import Tensor
from torchmetrics.utilities import dim_zero_cat
from torchmetrics import ROC
import numpy as np


class Sensitivity(Metric):
    def __init__(self, desired_specificity=0.95, task="binary", ** kwargs):
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.desired_specificity = desired_specificity
        self.roc = ROC(task=task)

    def update(self, preds: Tensor, target: Tensor) -> None:
        self.preds.append(preds)
        self.target.append(target)

    def compute(self):
        # parse inputs
        # Concatenate the lists of tensors into single tensors
        if len(self.preds) > 0 or len(self.target) > 0:
            preds = torch.cat(self.preds, dim=0)
            preds = preds.to(dtype=torch.float32)
            target = torch.cat(self.target, dim=0)
            target = target.to(dtype=torch.int64)
            # preds = preds.clone().cpu().numpy().astype(np.int64)
            # preds = np.squeeze(preds)
            # target = target.clone().cpu().numpy().astype(np.int64)
            # target = np.squeeze(target)
            fpr, tpr, thresholds = self.roc(
                preds, target)

            fpr = fpr.clone().cpu().numpy().astype(np.float32)
            # fpr = np.squeeze(preds)
            # Find the index of the threshold that is closest to the desired specificity
            idx = np.argmax(fpr >= (1 - self.desired_specificity))

            # Get the corresponding threshold
            threshold_at_desired_specificity = thresholds[idx]

            # Get the corresponding TPR (sensitivity)
            sensitivity_at_desired_specificity = tpr[idx]
            sensitivity_at_desired_specificity = sensitivity_at_desired_specificity.clone(
            ).detach().requires_grad_(False).to(dtype=torch.float32, device=self.device)
            return sensitivity_at_desired_specificity, torch.FloatTensor(threshold_at_desired_specificity, dtype=torch.float32)
        else:
            return None, None
