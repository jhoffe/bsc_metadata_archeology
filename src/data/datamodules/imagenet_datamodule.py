import os
from multiprocessing import cpu_count
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


class ImageNetDataModule(pl.LightningDataModule):
    imagenet_train: Dataset
    imagenet_val: Dataset
    num_workers: int

    def __init__(
        self,
        data_dir: str = "data/processed/imagenet",
        batch_size: int = 128,
        num_workers: Optional[int] = None,
    ):
        """Initializes the data module.

        Args:
            data_dir: str, directory where the CIFAR10 dataset is stored.
            batch_size: int, size of the mini-batch.
            num_workers: int, number of worker to use for data loading
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = cpu_count() if num_workers is None else num_workers

    def setup(self, stage: str) -> None:
        """Loads the CIFAR10 dataset from files.

        Args:
            stage: str, the stage for which the setup is being run (e.g. 'fit', 'test')
        """
        train_dataset = torch.load(os.path.join(self.data_dir, "train_probe_suite.pt"))
        val_dataset = torch.load(os.path.join(self.data_dir, "val.pt"))

        self.imagenet_train = train_dataset
        self.imagenet_val = val_dataset

    def train_dataloader(self) -> DataLoader:
        """Returns the dataloader for the validation set.

        Returns:
            DataLoader, the dataloader for the validation set.
        """
        return DataLoader(
            self.imagenet_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Returns the dataloader for the test set.

        Returns:
            DataLoader, the dataloader for the test set.
        """
        return DataLoader(
            self.imagenet_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Returns the dataloader for the test set.

        Returns:
            DataLoader, the dataloader for the test set.
        """
        return DataLoader(
            self.imagenet_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
