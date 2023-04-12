from typing import Any, Callable, Dict, Generator, List, Sized

import numpy as np
from torch.utils.data import Dataset, Subset

T_generator = Callable[[Any], Any]


def generate_random_outputs_suite():
    pass


def generate_corrupted_suite():
    pass


class ProbeSuiteDataset(Dataset):
    DEFAULT_PRE_SUITES: Dict[str, T_generator] = {
        "random_outputs": generate_random_outputs_suite,
        "corrupted": generate_corrupted_suite,
    }

    def __init__(
        self,
        dataset: Sized[Dataset],
        num_train_probes: int,
        num_val_probes: int,
        default_suites: List[str],
        custom_suites: Dict[str, T_generator],
    ):
        self.dataset = dataset
        # Check that dataset has length defined
        assert hasattr(self.dataset, "__len__")

        self.num_train_probes = num_train_probes
        self.num_val_probes = num_val_probes
        self.default_suites = default_suites
        self.custom_suites = custom_suites

        assert (
            len(set(default_suites).intersection(set(custom_suites.keys()))) == 0
        )  # No overlap

        self.used_indices = set()

    def generate_suite(self, suite: str) -> Generator:
        pass

    @staticmethod
    def subset(dataset: Sized[Dataset], num_probes: int) -> Subset:
        indices = np.random.choice(len(dataset), num_probes, replace=False)

        return Subset(dataset, indices)

    def create(self):
        pass

    def __getitem__(self, index: int):
        pass
