import os
from copy import deepcopy
from typing import List

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset


class CIFARDataModule(L.LightningDataModule):
    """Data module for loading the CIFAR dataset.

    Attributes:
        cifar_train: TensorDataset, the train set of CIFAR dataset
        cifar_validation: TensorDataset, the validation set of CIFAR dataset
        cifar_test: TensorDataset, the test set of CIFAR dataset
        num_workers: int, number of worker to use for data loading
    """

    cifar_train: TensorDataset
    cifar_test: TensorDataset
    cifar_probes: Dataset
    num_workers: int

    def __init__(
        self,
        data_dir: str = "data/processed/cifar10",
        batch_size: int = 128,
        num_workers: int = 4,
    ):
        """Initializes the data module.

        Args:
            data_dir: str, directory where the CIFAR dataset is stored.
            batch_size: int, size of the mini-batch.
            num_workers: int, number of worker to use for data loading
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        """Loads the CIFAR10 dataset from files.

        Args:
            stage: str, the stage for which the setup is being run (e.g. 'fit', 'test')
        """
        train_dataset = torch.load(os.path.join(self.data_dir, "train_probe_suite.pt"))
        test_dataset = torch.load(os.path.join(self.data_dir, "test.pt"))

        probes_dataset = deepcopy(train_dataset)
        probes_dataset.only_probes = True

        self.cifar_train = train_dataset
        self.cifar_probes = probes_dataset
        self.cifar_test = test_dataset

    def train_dataloader(self) -> DataLoader:
        """Returns the dataloader for the validation set.

        Returns:
            DataLoader, the dataloader for the validation set.
        """
        return DataLoader(
            self.cifar_train,
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
            self.cifar_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> List[DataLoader]:
        """Returns the dataloader for the test set.

        Returns:
            DataLoader, the dataloader for the test set.
        """
        return [
            DataLoader(
                self.cifar_test,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
            ),
            DataLoader(
                self.cifar_probes,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
            ),
        ]
