from torchmetrics import Metric
import torch
from torch import Tensor
from torchmetrics.utilities import dim_zero_cat
import Optional
from sklearn.metrics import roc_curve, roc_auc_score, auc
import numpy as np


class Specificity(Metric):
    def __init__(self, desired_specificity=0.95, ** kwargs):
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.desired_specificity = desired_specificity
        # Set to True if the metric is differentiable else set to False
        is_differentiable: Optional[bool] = None

        # Set to True if the metric reaches it optimal value when the metric is maximized.
        # Set to False if it when the metric is minimized.
        higher_is_better: Optional[bool] = True

        # Set to True if the metric during 'update' requires access to the global metric
        # state for its calculations. If not, setting this to False indicates that all
        # batch states are independent and we will optimize the runtime of 'forward'
        full_state_update: bool = True

    def update(self, preds: Tensor, target: Tensor) -> None:
        self.preds.append(preds)
        self.target.append(target)

    def compute(self):
        # parse inputs
        preds = dim_zero_cat(preds)
        target = dim_zero_cat(target)
        fpr, tpr, thresholds = roc_curve(target, preds)

        # Find the index of the threshold that is closest to the desired specificity
        idx = np.argmax(fpr >= (1 - self.desired_specificity))

        # Get the corresponding threshold
        threshold_at_desired_specificity = thresholds[idx]

        # Get the corresponding TPR (sensitivity)
        sensitivity_at_desired_specificity = tpr[idx]

        return torch.tensor(sensitivity_at_desired_specificity, dtype=torch.float32, device=self.device)
