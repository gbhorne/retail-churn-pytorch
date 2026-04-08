# src/mlp.py
# Three-layer MLP for binary churn classification.
# BatchNorm stabilizes training. Dropout prevents overfitting.
# Output is a raw logit — sigmoid is applied during evaluation.

import torch
import torch.nn as nn


class ChurnMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(256, 128, 64), dropout=0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            prev_dim = hidden_dim

        # final layer outputs a single logit (no activation)
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(1)