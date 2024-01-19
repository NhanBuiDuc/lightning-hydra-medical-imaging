import torch
from torch import nn


class VggHead(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_size: int = 10,
    ) -> None:
        """Initialize a `SimpleDenseNet` module.

        :param input_size: The number of input features.
        :param lin1_size: The number of output features of the first linear layer.
        :param lin2_size: The number of output features of the second linear layer.
        :param lin3_size: The number of output features of the third linear layer.
        :param output_size: The number of output features of the final linear layer.
        """
        super().__init__()

        # self.features = nn.Sequential(
        #     # conv1
        #     nn.Conv2d(3, 64, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, 3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2, return_indices=True),

        #     # conv2
        #     nn.Conv2d(64, 128, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 128, 3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2, return_indices=True),

        #     # conv3
        #     nn.Conv2d(128, 256, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 256, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 256, 3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2, return_indices=True),

        #     # conv4
        #     nn.Conv2d(256, 512, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(512, 512, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(512, 512, 3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2, return_indices=True),

        #     # conv5
        #     nn.Conv2d(512, 512, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(512, 512, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(512, 512, 3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2, return_indices=True)
        # )
        self.features = nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(2, stride=2, return_indices=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(2, stride=2, return_indices=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(2, stride=2, return_indices=True),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),
        )
        # self.classifier = nn.Sequential(
        #     nn.Linear(64 * 7 * 7, 4096),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     nn.Linear(4096, 8)
        # )
        self.classifier = nn.Sequential(
            nn.Linear(8192, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 8),
            nn.Softmax()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        for layer in self.features:
            if isinstance(layer, nn.MaxPool2d):
                x, location = layer(x)
            else:
                x = layer(x)

        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    _ = VggHead()
