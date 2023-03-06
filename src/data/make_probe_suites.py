import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torch.utils.data.dataset import T_co
from torchvision.transforms import transforms


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return torch.clip(
            tensor + torch.randn(tensor.size()) * self.std + self.mean, 0.0, 1.0
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
    typical_idx: list

    atypical: list
    atypical_idx: list

    random_outputs: list
    random_outputs_idx: list

    random_inputs_outputs: list
    random_inputs_outputs_idx: list

    corrupted: list
    corrupted_idx: list

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
        self.generator = torch.Generator().manual_seed(seed)

    def generate(self):
        self.generate_atypical()
        self.generate_typical()
        self.generate_random_outputs()
        self.generate_random_inputs_outputs()
        self.generate_corrupted()

    def generate_typical(self):
        sorted_indices = np.argsort(self.dataset.score)
        subset = self.get_subset(sorted_indices[: self.num_probes])
        self.typical = [(x, y) for x, y, _ in subset]

    def generate_atypical(self):
        sorted_indices = np.argsort(self.dataset.score)
        subset = self.get_subset(sorted_indices[-self.num_probes :])  # noqa: E203
        self.atypical = [(x, y) for x, y, _ in subset]

    def generate_random_outputs(self):
        subset = self.get_subset()
        self.random_outputs = [
            (
                x,
                torch.multinomial(
                    torch.Tensor([1 if y != i else 0 for i in range(self.label_count)]),
                    1,
                    generator=self.generator,
                ).item(),
            )
            for x, y, _ in subset
        ]

    def generate_random_inputs_outputs(self):
        subset = self.get_subset()

        self.random_inputs_outputs = [
            (torch.rand_like(x), torch.randint(0, self.label_count, (1,)).item())
            for x, y, _ in subset
        ]

    def generate_corrupted(self):
        subset = self.get_subset()
        corruption_transform = transforms.Compose(
            [AddGaussianNoise(mean=0.0, std=0.25), ClampRangeTransform()]
        )

        self.corrupted = [(corruption_transform(x), y) for x, y, _ in subset]

    def get_subset(self, indices: Optional[list[int]] = None) -> Subset:
        if indices is None:
            subset_indices = torch.multinomial(
                torch.ones(len(self.remaining_indices)),
                self.num_probes,
                replacement=False,
                generator=self.generator,
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

    def __getitem__(self, index) -> T_co:
        return self.combined[index]

    def __len__(self):
        return len(self.combined)


def make_probe_suites(
    input_filepath: str, output_filepath: str, dataset: str, num_probes: int = 250
):
    imagenet_dataset = torch.load(os.path.join(input_filepath, "cifar10", "train.pt"))
    imagenet_dataset_len = len(imagenet_dataset)
    imagenet_label_count = 10

    probe_suite = ProbeSuiteGenerator(
        imagenet_dataset,
        imagenet_dataset_len,
        imagenet_label_count,
        num_probes=num_probes,
    )

    probe_suite.generate()

    print(len(probe_suite.used_indices))
    print(len(imagenet_dataset))
    print(
        len(imagenet_dataset) - len(probe_suite.used_indices),
        len(probe_suite.remaining_indices),
    )


make_probe_suites("data/processed", "data/processed", "cifar10")
