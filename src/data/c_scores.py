import os
import pathlib
from os import path
from typing import Callable, Optional

import numpy as np
import torch
from torchvision.datasets import CIFAR10, CIFAR100

from src.data.utils.c_score_downloader import c_score_downloader
from src.data.utils.cifar_transform import cifar_transform


def c_scores(dataset: str) -> np.ndarray:
    
    if "cifar100" in dataset:
        file = pathlib.Path(f"data/external/cifar100_infl_matrix.npz")

        scores = np.load(file, allow_pickle=True)

        labels = scores["tr_labels"]
        mem_values = scores["tr_mem"]

    else:
        file = pathlib.Path(f"data/external/cifar10-cscores-orig-order.npz")

        cscores = np.load(file, allow_pickle=True)

        labels = cscores["labels"]
        scores = cscores["scores"]

        mem_values = 1.0 - scores

    data = CIFAR100 if dataset == "cifar100" else CIFAR10
    data = data(root=f"data/raw/{dataset}", train=True, download=False)
    assert np.all(data.targets == labels), "The labels are not the same."

    return mem_values


def c_scores_dataset(dataset: str, input_filepath: str, output_filepath: str) -> None:
    data = (
        CustomCIFAR100
        if dataset == "cifar100"
        else CustomCIFAR10
        if dataset == "cifar10"
        else None
    )
    assert data is not None

    train_data = data(
        score=c_scores(dataset),
        root=path.join(input_filepath, dataset),
        train=True,
        transform=cifar_transform(train=True),
    )

    test_data = data(
        root=path.join(input_filepath, dataset),
        train=False,
        transform=cifar_transform(train=False),
    )

    output_dir = os.path.join(output_filepath, dataset)
    os.makedirs(output_dir, exist_ok=True)

    torch.save(train_data, os.path.join(output_dir, "train.pt"))
    torch.save(test_data, os.path.join(output_dir, "test.pt"))


class CustomCIFAR10(CIFAR10):
    """CIFAR10 dataset with memory scores."""

    def __init__(
        self,
        root: str,
        train: bool = True,
        score: Optional[list] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=False,
        )

        if score is not None:
            self.score = score

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, score) where target is index of the target class.
        """

        img, target = super().__getitem__(index)
        if not self.train:
            return img, target

        score = self.score[index]
        return img, target, score


class CustomCIFAR100(CIFAR100):
    """CIFAR100 dataset with memory scores."""

    def __init__(
        self,
        root: str,
        train: bool = True,
        score: Optional[list] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=False,
        )

        if score is not None:
            self.score = score

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, score) where target is index of the target class.
        """

        img, target = super().__getitem__(index)
        if not self.train:
            return img, target

        score = self.score[index]
        return img, target, score


if __name__ == "__main__":
    c_scores_dataset("cifar10", "data/raw", "data/processed")
    c_scores_dataset("cifar100", "data/raw", "data/processed")
