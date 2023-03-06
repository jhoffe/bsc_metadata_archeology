import os
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision.transforms import transforms


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0, generator: torch.Generator = None):
        self.std = std
        self.mean = mean
        self.generator = generator

    def __call__(self, tensor):
        return torch.clip(
            tensor
            + torch.randn(tensor.size(), generator=self.generator) * self.std
            + self.mean,
            0.0,
            1.0,
        )


class ClampRangeTransform(object):
    def __init__(self):
        pass

    def __call__(self, x):
        return torch.clamp(x, 0.0, 1.0)


class ProbeSuiteGenerator(Dataset):
    dataset: Dataset
    remaining_indices: list
    used_indices: list = []
    dataset_len: int
    label_count: int

    typical: list
    atypical: list
    random_outputs: list
    random_inputs_outputs: list
    corrupted: list

    dataset_indices_to_probe_indices: Dict[int, int] = {}

    _combined: Optional[list] = None

    def __init__(
        self,
        dataset: Dataset,
        dataset_len: int,
        label_count: int,
        seed: int = 123,
        num_probes: int = 250,
    ):
        self.dataset = dataset
        assert hasattr(self.dataset, "score")
        self.dataset_len = dataset_len
        self.remaining_indices = list(range(dataset_len))
        self.label_count = label_count
        self.num_probes = num_probes

    def generate(self, generator: torch.Generator):
        self.generate_atypical()
        self.generate_typical()
        self.generate_random_outputs(generator)
        self.generate_random_inputs_outputs(generator)
        self.generate_corrupted(generator)

        self.dataset_indices_to_probe_indices = {
            ds_idx: ps_idx for ps_idx, (_, _, ds_idx) in enumerate(self.combined)
        }

    def generate_typical(self):
        sorted_indices = np.argsort(self.dataset.score)
        subset = self.get_subset(indices=sorted_indices[: self.num_probes])
        self.typical = [(x, y, idx) for (x, y, _), idx in zip(subset, subset.indices)]

    def generate_atypical(self):
        sorted_indices = np.argsort(self.dataset.score)
        subset = self.get_subset(
            indices=sorted_indices[-self.num_probes :]  # noqa: E203
        )  # noqa: E203
        self.atypical = [(x, y, idx) for (x, y, _), idx in zip(subset, subset.indices)]

    def generate_random_outputs(self, generator: torch.Generator):
        subset = self.get_subset(generator=generator)
        self.random_outputs = [
            (
                x,
                torch.multinomial(
                    torch.Tensor([1 if y != i else 0 for i in range(self.label_count)]),
                    1,
                    generator=generator,
                ).item(),
                idx,
            )
            for (x, y, _), idx in zip(subset, subset.indices)
        ]

    def generate_random_inputs_outputs(self, generator: torch.Generator):
        subset = self.get_subset(generator=generator)

        self.random_inputs_outputs = [
            (
                torch.rand_like(x),
                torch.randint(0, self.label_count, (1,), generator=generator).item(),
                idx,
            )
            for (x, y, _), idx in zip(subset, subset.indices)
        ]

    def generate_corrupted(self, generator: torch.Generator):
        subset = self.get_subset(generator=generator)
        corruption_transform = transforms.Compose(
            [AddGaussianNoise(mean=0.0, std=0.25), ClampRangeTransform()]
        )

        self.corrupted = [
            (corruption_transform(x), y, idx)
            for (x, y, _), idx in zip(subset, subset.indices)
        ]

    def get_subset(
        self,
        generator: Optional[torch.Generator] = None,
        indices: Optional[list[int]] = None,
    ) -> Subset:
        if indices is None:
            subset_indices = torch.multinomial(
                torch.ones(len(self.remaining_indices)),
                self.num_probes,
                replacement=False,
                generator=generator,
            )
        else:
            subset_indices = indices

        self.used_indices.extend(subset_indices)
        self.remaining_indices = [
            self.remaining_indices[i]
            for i in range(len(self.remaining_indices))
            if i not in subset_indices
        ]

        return Subset(self.dataset, subset_indices.tolist())

    @property
    def combined(self):
        if self._combined is None:
            self._combined = (
                self.typical
                + self.atypical
                + self.random_outputs
                + self.random_inputs_outputs
                + self.corrupted
            )

        return self._combined

    def __getitem__(self, index):
        if index in self.used_indices:
            return self.combined[self.dataset_indices_to_probe_indices[index]], index

        return self.dataset[index], index

    def __len__(self):
        return self.dataset_len


def make_probe_suites(
    input_filepath: str,
    output_filepath: str,
    dataset: str,
    label_count: int,
    num_probes: int = 500,
):
    data = torch.load(os.path.join(input_filepath, dataset, "train.pt"))
    dataset_length = len(data)

    probe_suite = ProbeSuiteGenerator(
        data,
        dataset_length,
        label_count,
        num_probes=num_probes,
    )
    probe_suite.generate(torch.Generator().manual_seed(123))

    torch.save(
        probe_suite, os.path.join(output_filepath, dataset, "train_probe_suite.pt")
    )
