import os
from copy import deepcopy
from multiprocessing import cpu_count
from typing import Optional

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset


class ImageNetDataModule(L.LightningDataModule):
    imagenet_train: Dataset
    imagenet_val: Dataset
    imagenet_probes: Dataset
    num_workers: int
    prefetch_factor: Optional[int]

    def __init__(
        self,
        data_dir: str = "data/processed/imagenet",
        batch_size: int = 128,
        num_workers: Optional[int] = None,
        prefetch_factor: Optional[int] = None,
    ):
        """Initializes the data module.

        Args:
            data_dir: str, directory where the imagenet dataset is stored.
            batch_size: int, size of the mini-batch.
            num_workers: int, number of worker to use for data loading
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = cpu_count() if num_workers is None else num_workers
        self.prefetch_factor = prefetch_factor

    def setup(self, stage: str) -> None:
        """Loads the imagenet dataset from files.

        Args:
            stage: str, the stage for which the setup is being run (e.g. 'fit', 'test')
        """
        train_dataset = torch.load(os.path.join(self.data_dir, "train_probe_suite.pt"))
        val_dataset = torch.load(os.path.join(self.data_dir, "val.pt"))

        probes_dataset = deepcopy(train_dataset)
        probes_dataset.only_probes = True

        self.imagenet_train = train_dataset
        self.imagenet_val = val_dataset
        self.imagenet_probes = probes_dataset

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
            prefetch_factor=self.prefetch_factor,
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
            prefetch_factor=self.prefetch_factor,
        )

    def val_dataloader(self) -> list[DataLoader]:
        """Returns the dataloader for the test set.

        Returns:
            DataLoader, the dataloader for the test set.
        """
        return [
            DataLoader(
                self.imagenet_val,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                prefetch_factor=self.prefetch_factor,
            ),
            DataLoader(
                self.imagenet_probes,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                prefetch_factor=self.prefetch_factor,
            ),
        ]
