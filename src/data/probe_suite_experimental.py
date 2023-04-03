from typing import Any, Callable, Dict

from torch.utils.data import Dataset


class ProbeSuiteDataset(Dataset):
    dataset: Dataset

    def __init__(
        self, dataset: Dataset, probe_generators: Dict[str, Callable[[Any], Any]]
    ):
        self.dataset = dataset

    def __getitem__(self, index: int):
        pass
