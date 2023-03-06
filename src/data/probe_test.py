import torch
import os
import numpy as np
from src.models.create_datamodule import create_datamodule

train_dataset = torch.load(os.path.join("data/processed/cifar10/train_probe_suite.pt"))

combined = train_dataset.combined
idxs = [combined[i][-1] for i in range(len(combined))]

for idx in idxs:
    assert idx in train_dataset.used_indices
    assert idx not in train_dataset.remaining_indices
