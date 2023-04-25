import os
from copy import deepcopy
from typing import List, Optional, Union

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.utils import IDXDataset
from src.data.utils.idx_to_label_names import get_idx_to_label_names_speechcommands


class SCDataModule(L.LightningDataModule):
    """Data module for loading the CIFAR dataset.

    Attributes:
        sc_train: TensorDataset, the train set of SPEECHCOMMANDS dataset
        sc_validation: TensorDataset, the validation set of SPEECHCOMMANDS dataset
        sc_test: TensorDataset, the test set of SPEECHCOMMANDS dataset
        num_workers: int, number of worker to use for data loading
    """

    sc_train: Dataset
    sc_validation: Dataset
    sc_test: Dataset
    sc_probes: Optional[Dataset]
    num_workers: int

    def __init__(
        self,
        data_dir: str = "data/processed/speechcommands",
        batch_size: int = 128,
        num_workers: int = 4,
        c_score_type: str = "c-score",
        prefetch_factor: int = 2,
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
        self.c_score_type = c_score_type
        self.idx_to_label_names = get_idx_to_label_names_speechcommands()
        self.label_to_idx = {v: k for k, v in self.idx_to_label_names.items()}
        self.prefetch_factor = prefetch_factor

    def setup(self, stage: str) -> None:
        """Loads the CIFAR10 dataset from files.

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
        test_dataset = torch.load(os.path.join(self.data_dir, "test.pt"))
        validation_dataset = torch.load(os.path.join(self.data_dir, "validation.pt"))

        if self.c_score_type != "none":
            probes_dataset = deepcopy(train_dataset)
            probes_dataset.only_probes = True

            self.sc_probes = probes_dataset
            self.sc_train = train_dataset
        else:
            self.sc_probes = None
            self.sc_train = IDXDataset(train_dataset)

        self.sc_test = IDXDataset(test_dataset)
        self.sc_validation = IDXDataset(validation_dataset)

    def pad_sequence(self, batch):
        """Pads the sequences in the batch.

        Args:
            batch: list, a batch of sequences

        Returns:
            list, the padded batch
        """
        batch = [item.t() for item in batch]
        batch = torch.nn.utils.rnn.pad_sequence(
            batch, batch_first=True, padding_value=0.0
        )
        return batch.permute(0, 2, 1)

    def collate_fn(self, batch):
        # A data tuple has the form:
        # waveform, sample_rate, label, speaker_id, utterance_number, index
        tensors, targets, indices = [], [], []

        # Gather in lists, and encode labels as indices
        for (waveform, label), index in batch:
            tensors.append(waveform)
            targets.append(label)
            indices.append(index)

        # Group the list of tensors into a batched tensor
        tensors = self.pad_sequence(tensors)
        targets = torch.tensor(targets)
        indices = torch.tensor(indices)

        return tensors, targets, indices

    def train_dataloader(self) -> DataLoader:
        """Returns the dataloader for the validation set.

        Returns:
            DataLoader, the dataloader for the validation set.
        """
        return DataLoader(
            self.sc_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=self.collate_fn,
            prefetch_factor=self.prefetch_factor,
        )

    def test_dataloader(self) -> DataLoader:
        """Returns the dataloader for the test set.

        Returns:
            DataLoader, the dataloader for the test set.
        """
        return DataLoader(
            self.sc_test,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=self.collate_fn,
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
            return DataLoader(
                self.sc_validation,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=self.collate_fn,
                num_workers=self.num_workers,
                pin_memory=True,
                prefetch_factor=self.prefetch_factor,
            )

        return [
            DataLoader(
                self.sc_validation,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=self.collate_fn,
                num_workers=self.num_workers,
                pin_memory=True,
                prefetch_factor=self.prefetch_factor,
            ),
            DataLoader(
                self.sc_probes,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=self.collate_fn,
                drop_last=False,
            ),
        ]
