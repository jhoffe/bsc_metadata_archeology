import os

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.data.c_scores import CustomCIFAR10, CustomCIFAR100


class CIFAR10DataModule(pl.LightningDataModule):
    """Data module for loading the CIFAR10 dataset.

    Attributes:
        cifar10_train: TensorDataset, the train set of CIFAR10 dataset
        cifar10_validation: TensorDataset, the validation set of CIFAR10 dataset
        cifar10_test: TensorDataset, the test set of CIFAR10 dataset
        num_workers: int, number of worker to use for data loading
    """

    cifar10_train: TensorDataset
    cifar10_test: TensorDataset
    num_workers: int

    def __init__(
        self,
        data_dir: str = "data/processed/cifar10",
        batch_size: int = 128,
        num_workers: int = 4,
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
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        """Loads the CIFAR10 dataset from files.

        Args:
            stage: str, the stage for which the setup is being run (e.g. 'fit', 'test')
        """
        train_dataset = torch.load(os.path.join(self.data_dir, "train.pt"))
        print(train_dataset)
        test_dataset = torch.load(os.path.join(self.data_dir, "test.pt"))

        self.cifar10_train = train_dataset
        self.cifar10_test = test_dataset

    def train_dataloader(self) -> DataLoader:
        """Returns the dataloader for the validation set.

        Returns:
            DataLoader, the dataloader for the validation set.
        """
        return DataLoader(
            self.cifar10_train,
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
            self.cifar10_test,
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
            self.cifar10_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class CIFAR100DataModule(pl.LightningDataModule):
    """Data module for loading the CIFAR10 dataset.

    Attributes:
        cifar10_train: TensorDataset, the train set of CIFAR100 dataset
        cifar10_test: TensorDataset, the test set of CIFAR100 dataset
        num_workers: int, number of worker to use for data loading
    """

    cifar100_train: TensorDataset
    cifar100_test: TensorDataset
    num_workers: int

    def __init__(
        self,
        data_dir: str = "data/processed/cifar100",
        batch_size: int = 128,
        num_workers: int = 4,
    ):
        """Initializes the data module.

        Args:
            data_dir: str, directory where the CIFAR100 dataset is stored.
            batch_size: int, size of the mini-batch.
            num_workers: int, number of worker to use for data loading
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        """Loads the CIFAR100 dataset from files.

        Args:
            stage: str, the stage for which the setup is being run (e.g. 'fit', 'test')
        """
        train_dataset = torch.load(os.path.join(self.data_dir, "train.pt"))
        test_dataset = torch.load(os.path.join(self.data_dir, "test.pt"))

        self.cifar100_train = train_dataset
        self.cifar100_test = test_dataset

    def train_dataloader(self) -> DataLoader:
        """Returns the dataloader for the validation set.

        Returns:
            DataLoader, the dataloader for the validation set.
        """
        return DataLoader(
            self.cifar100_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Returns the dataloader for the test set.

        Returns:
            DataLoader, the dataloader for the test set.
        """
        return DataLoader(
            self.cifar100_test, batch_size=self.batch_size, num_workers=self.num_workers
        )
