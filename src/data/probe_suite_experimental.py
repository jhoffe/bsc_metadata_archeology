from typing import Any, Callable, Dict, List, Optional, Sized, Tuple

import numpy as np
from torch.utils.data import Dataset


class ProbeSuiteDataset(Dataset):
    dataset: Sized[Dataset]
    probe_generators: Dict[str, Callable[[Any], Any]]
    probe_suite_indices: Dict[str, Tuple[List[int], List[int]]]
    num_train_probes: int
    num_val_probes: int

    def __init__(
        self,
        dataset: Sized[Dataset],
        num_train_probes: int,
        num_val_probes: int,
        probe_generators: Dict[str, Callable[[Any], Any]],
        probe_suite_indices: Optional[Dict[str, List[int]]] = None,
    ):
        self.dataset = dataset

        # Check that dataset has length defined
        assert hasattr(dataset, "__len__")

        self.num_train_probes = num_train_probes
        self.num_val_probes = num_val_probes
        self.probe_generators = probe_generators

        if probe_suite_indices is None:
            self.probe_suite_indices = {}

    def create_subset_indices(self, num_indices: int) -> List[int]:
        all_indices = list(range(len(self.dataset)))
        used_indices = sum(self.probe_suite_indices.values())

        remaining_indices = list(set(all_indices) - set(used_indices))

        return np.random.choice(remaining_indices, num_indices, replace=False)

    def __getitem__(self, index: int):
        pass
