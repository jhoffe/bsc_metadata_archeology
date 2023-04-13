import os
import pathlib
from os import path
from typing import Callable, Optional

import numpy as np
import torch
from torchvision.datasets import CIFAR10, CIFAR100

from src.data.utils.c_score_downloader import downloader
from src.data.utils.cifar_transform import cifar_transform


def c_scores(dataset: str, use_cscores: Optional[bool]) -> np.ndarray:
    """Get cifar c-scores."""
    if dataset in "cifar10":
        assert (
            use_cscores is not False
        ), "Memorization scores are not available for CIFAR10."

    if use_cscores is not None:
        if "cifar100" in dataset:
            file = (
                pathlib.Path("data/external/cifar100-cscores-orig-order.npz")
                if use_cscores
                else pathlib.Path("data/external/cifar100_infl_matrix.npz")
            )

            if not file.exists():
                downloader()

            scores = np.load(file, allow_pickle=True)

            labels = scores["tr_labels"]
            mem_values = scores["tr_mem"]

        else:
            file = pathlib.Path("data/external/cifar10-cscores-orig-order.npz")
            if not file.exists():
                downloader()

            cscores = np.load(file, allow_pickle=True)

            labels = cscores["labels"]
            scores = cscores["scores"]

            mem_values = 1.0 - scores

        data = CIFAR100 if dataset == "cifar100" else CIFAR10
        data = data(root=f"data/raw/{dataset}", train=True, download=False)
        assert np.all(data.targets == labels), "The labels are not the same."

        return mem_values

    return None


def c_scores_dataset(
    dataset: str, input_filepath: str, output_filepath: str, use_cscores: Optional[bool]
) -> None:
    data = (
        CustomCIFAR100
        if dataset == "cifar100"
        else CustomCIFAR10
        if dataset == "cifar10"
        else None
    )
    assert data is not None

    train_data = data(
        score=c_scores(dataset, use_cscores),
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

    torch.save(test_data, os.path.join(output_dir, "test.pt"))
    if use_cscores is not None:
        torch.save(
            train_data, os.path.join(output_dir, "train_c_scores.pt")
        ) if use_cscores else torch.save(
            train_data, os.path.join(output_dir, "train_mem_scores.pt")
        )
    else:
        torch.save(train_data, os.path.join(output_dir, "train.pt"))


class CustomCIFAR10(CIFAR10):
    """CIFAR10 dataset with C-scores."""

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

        if self.score is None:
            return img, target

        score = self.score[index]
        return img, target, score


class CustomCIFAR100(CIFAR100):
    """CIFAR100 dataset with C-scores."""

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

        if self.score is None:
            return img, target

        score = self.score[index]
        return img, target, score


if __name__ == "__main__":
    c_scores_dataset("cifar10", "data/raw", "data/processed", use_cscores=True)
    c_scores_dataset("cifar100", "data/raw", "data/processed", use_cscores=True)
