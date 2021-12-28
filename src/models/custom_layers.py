import torch
import torch.nn as nn


class MultiplicationLayer(nn.Module):
    def __init__(self, size):
        super().__init__()
        weights = torch.Tensor(size)
        self.weights = nn.Parameter(weights)

        # Initialize
        nn.init.uniform_(self.weights, -0.01, +0.01)

    def forward(self, x):
        return x * self.weights
