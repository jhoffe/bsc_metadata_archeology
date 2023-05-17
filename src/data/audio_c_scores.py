import os
from os import path
from typing import Dict, Optional

import torch
from torchaudio.datasets import SPEECHCOMMANDS

from src.data.proxy_calculator import ProxyCalculator


def load_proxy_scores(proxy_dataset_path: str) -> Dict[int, float]:
    proxy_calculator = ProxyCalculator(proxy_dataset_path)
    proxy_calculator.load()

    return proxy_calculator.calculate_proxy_scores("p_L")


def c_scores_dataset(
    dataset: str,
    input_filepath: str,
    output_filepath: str,
    proxy_dataset: Optional[str] = None,
) -> None:
    data = CustomSC if dataset == "speechcommands" else None
    assert data is not None

    train_data = data(
        root=path.join(input_filepath, dataset),
        subset="training",
        score=load_proxy_scores(proxy_dataset) if proxy_dataset is not None else None,
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
    if proxy_dataset is not None:
        torch.save(train_data, os.path.join(output_dir, "train_c_scores.pt"))
    else:
        torch.save(train_data, os.path.join(output_dir, "train.pt"))


class CustomSC(SPEECHCOMMANDS):
    """SPEECHCOMMANDS dataset with C-scores."""

    def __init__(
        self,
        root: str,
        subset: str = "training",
        score: Optional[Dict[int, float]] = None,
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
        (
            waveform,
            _sample_rate,
            target,
            _speaker_id,
            _utterance_number,
        ) = super().__getitem__(index)
        if not self.train or self.score is None:
            return waveform, target

        score = self.score[index]
        return waveform, target, score


if __name__ == "__main__":
    c_scores_dataset("speechcommands", "data/raw", "data/processed")
