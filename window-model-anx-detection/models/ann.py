import torch
import torch.nn as nn

from .base import BaseNet


class SimpleANN(BaseNet):
    def __init__(self, input_dim=161, hidden_sizes=[256, 128, 64], dropout=0.5):
        super().__init__()

        layers = []
        last_dim = input_dim

        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            last_dim = h

        self.mlp = nn.Sequential(*layers)
        self.output = nn.Linear(last_dim, 1)

    #  collapse time dimension
    def feature_extractor(self, x):
        # x: [B, T, 161]
        return x.mean(dim=1)   # → [B, 161]

    def classifier(self, x):
        x = self.mlp(x)
        return self.output(x)