import torch
from torch import nn


class MomentsCalculator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X_r, mask_r, X_a, y_a, mask_a) -> torch.Tensor:
        pass