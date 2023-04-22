import os
from os import path
from typing import Optional

import numpy as np
import torch
from torchaudio.datasets import SPEECHCOMMANDS


def c_scores(dataset: str, use_cscores: Optional[bool]) -> np.ndarray:
    """Get cifar c-scores."""
    if use_cscores:
        pass

    return None


def c_scores_dataset(
    dataset: str,
    input_filepath: str,
    output_filepath: str,
    use_c_scores: Optional[bool] = None,
) -> None:
    data = CustomSC if dataset == "speechcommands" else None
    assert data is not None

    train_data = data(
        root=path.join(input_filepath, dataset),
        subset="training",
        score=c_scores(dataset, use_c_scores),
    )

    test_data = data(
        root=path.join(input_filepath, dataset),
        subset="testing",
    )

    validation_data = data(
        root=path.join(input_filepath, dataset),
        subset="validation",
    )

    output_dir = os.path.join(output_filepath, dataset)
    os.makedirs(output_dir, exist_ok=True)

    torch.save(test_data, os.path.join(output_dir, "test.pt"))
    torch.save(validation_data, os.path.join(output_dir, "validation.pt"))
    if use_c_scores is not None:
        torch.save(
            train_data, os.path.join(output_dir, "train_c_scores.pt")
        ) if use_c_scores else torch.save(
            train_data, os.path.join(output_dir, "train_mem_scores.pt")
        )
    else:
        torch.save(train_data, os.path.join(output_dir, "train.pt"))


test = SPEECHCOMMANDS


class CustomSC(SPEECHCOMMANDS):
    """SPEECHCOMMANDS dataset with C-scores."""

    def __init__(
        self,
        root: str,
        subset: str = "training",
        score: Optional[list] = None,
    ) -> None:
        super().__init__(
            root=root,
            subset=subset,
            download=False,
        )

        self.subset = subset
        self.train = self.subset == "train"

        if score is not None:
            self.score = score
        else:
            self.score = None

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
    c_scores_dataset("speechcommands", "data/raw", "data/processed")
