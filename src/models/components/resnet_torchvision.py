from torchvision import models
from torch import nn
import torch


class ResNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        version: str = "resnet50",
    ):
        super().__init__()
        self.num_classes = num_classes
        switch = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
            'resnet152': models.resnet152,
        }
        # Check if the specified model_type is in the switch case
        if version not in switch:
            if (self.num_classes > 2):
                self.resnet = switch["resnet18"](
                    weights=None, progress=True, num_classes=num_classes)
            else:
                self.resnet = switch["resnet18"](
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
