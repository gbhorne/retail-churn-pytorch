# src/dataset.py
# PyTorch Dataset wrapper and DataLoader factory.
# WeightedRandomSampler oversamples the minority class during training
# so the model sees a balanced mix of churned/not-churned batches.

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np


class ChurnDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_loader(X, y, batch_size=512, oversample=False, shuffle=True):
    dataset = ChurnDataset(X, y)
    sampler = None

    if oversample:
        # give each sample a weight inverse to its class frequency
        # minority class samples get upweighted so they appear more often
        counts  = np.bincount(y.astype(int))
        weights = 1.0 / counts[y.astype(int)]
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )
        shuffle = False  # mutually exclusive with sampler

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler
    )