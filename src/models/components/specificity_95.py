from torchmetrics import Metric
import torch
from torch import Tensor
from torchmetrics.utilities import dim_zero_cat
from sklearn.metrics import roc_curve, roc_auc_score, auc
import numpy as np


class Specificity(Metric):
    def __init__(self, desired_specificity=0.95, ** kwargs):
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.desired_specificity = desired_specificity

    def update(self, preds: Tensor, target: Tensor) -> None:
        self.preds.append(preds)
        self.target.append(target)

    def compute(self):
        # parse inputs
        # Concatenate the lists of tensors into single tensors
        preds = torch.cat(self.preds, dim=0)
        target = torch.cat(self.target, dim=0)
        preds = preds.clone().cpu().numpy()
        preds = np.squeeze(preds)
        target = target.clone().cpu().numpy()
        target = np.squeeze(target)
        fpr, tpr, thresholds = roc_curve(
            preds, target)

        # Find the index of the threshold that is closest to the desired specificity
        idx = np.argmax(fpr >= (1 - self.desired_specificity))

        # Get the corresponding threshold
        threshold_at_desired_specificity = thresholds[idx]

        # Get the corresponding TPR (sensitivity)
        sensitivity_at_desired_specificity = tpr[idx]

        return torch.tensor(sensitivity_at_desired_specificity, dtype=torch.float32, device=self.device)
