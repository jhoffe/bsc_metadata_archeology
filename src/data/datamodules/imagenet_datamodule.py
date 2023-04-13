import os
from copy import deepcopy
from multiprocessing import cpu_count
from typing import List, Optional, Union

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset


class ImageNetDataModule(L.LightningDataModule):
    imagenet_train: Dataset
    imagenet_val: Dataset
    imagenet_probes: Optional[Dataset]
    num_workers: int
    prefetch_factor: Optional[int]

    def __init__(
        self,
        data_dir: str = "data/processed/imagenet",
        batch_size: int = 128,
        num_workers: Optional[int] = None,
        prefetch_factor: Optional[int] = None,
        c_score_type: str = "c-score"
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
        self.c_score_type = c_score_type

    def setup(self, stage: str) -> None:
        """Loads the imagenet dataset from files.

        Args:
            stage: str, the stage for which the setup is being run (e.g. 'fit', 'test')
        """
        if self.c_score_type == "c-score":
            train_name = "train_probe_suite_c_scores.pt"
        elif self.c_score_type == "mem-score":
            train_name = "train_probe_suite_mem_scores.pt"
        elif self.c_score_type == "none":
            train_name = "train.pt"
        else:
            raise ValueError(f"Invalid c_score_type: {self.c_score_type}")

        train_dataset = torch.load(os.path.join(self.data_dir, train_name))
        val_dataset = torch.load(os.path.join(self.data_dir, "val.pt"))

        if self.c_score_type != "none":
            probes_dataset = deepcopy(train_dataset)
            probes_dataset.only_probes = True
            self.imagenet_probes = probes_dataset
        else:
            self.imagenet_probes = None

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

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        """Returns the dataloader for the test set.

        Returns:
            DataLoader, the dataloader for the test set.
        """
        if self.c_score_type == "none":
            return self.test_dataloader()

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
