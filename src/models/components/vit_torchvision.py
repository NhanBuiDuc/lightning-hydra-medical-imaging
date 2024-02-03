from torchvision import models
from torch import nn
import torch


class Swin(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        version: str = "swin_v2_b",
    ):
        super().__init__()
        self.num_classes = num_classes
        switch = {
            'swin_v2_b': models.swin_v2_b,
        }
        # Check if the specified model_type is in the switch case
        if version not in switch:
            if (self.num_classes > 2):
                self.resnet = switch["swin_v2_b"](
                    weights=None, progress=True, num_classes=num_classes)
            else:
                self.resnet = switch["swin_v2_b"](
                    weights=None, progress=True, num_classes=num_classes-1)
        else:
            if (self.num_classes > 2):
                # Initialize the ResNet model without pre-training and with progress
                self.resnet = switch[version](
                    weights=None, progress=True, num_classes=num_classes)
            else:
                self.resnet = switch[version](
                    weights=None, progress=True, num_classes=num_classes-1)

        self.sigmoid = nn.Sigmoid()
        self.soft_max = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet(x)
        if (self.num_classes > 2):
            x = self.soft_max(x)
        else:
            x = self.sigmoid(x)
        return x
