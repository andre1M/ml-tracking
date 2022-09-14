from torch import nn
import torch


__all__ = ["SimpleConv"]


class SimpleConv(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding=0
            ),
            nn.GELU(),
            nn.Conv2d(
                in_channels=3,
                out_channels=1,
                kernel_size=4,
                stride=2,
                padding=0
            ),
            nn.GELU(),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(
                in_features=144,
                out_features=num_classes
            ),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
