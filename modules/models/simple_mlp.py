from torch import nn
import torch


__all__ = ["SimpleMLP"]


class SimpleMLP(nn.Module):
    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=in_features,
                out_features=100
            ),
            nn.GELU(),
            nn.Linear(
                in_features=100,
                out_features=num_classes
            ),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
