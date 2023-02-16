import os

from torch.utils.data import Dataset, Subset
from torchvision.transforms import transforms
import torch


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


class ProbeSuiteGenerator:
    dataset: Dataset
    remaining_indices: list
    dataset_len: int
    label_count: int

    typical: list
    atypical: list
    random_outputs: list
    random_inputs_outputs: list
    corrupted: list

    def __init__(
        self,
        dataset: Dataset,
        dataset_len: int,
        label_count: int,
        seed: int = 123,
        num_probes: int = 250,
    ):
        self.dataset = dataset
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
        pass

    def generate_atypical(self):
        pass

    def generate_random_outputs(self):
        subset = self.get_subset()
        self.random_outputs = [
            (
                x,
                y,
                torch.multinomial(
                    torch.Tensor([1 if y != i else 0 for i in range(self.label_count)]),
                    1,
                    generator=self.generator,
                ).item(),
            )
            for x, y in subset
        ]

    def generate_random_inputs_outputs(self):
        subset = self.get_subset()

        self.random_inputs_outputs = [
            (torch.rand_like(x), torch.randint(0, self.label_count, (1,)).item())
            for x, y in subset
        ]

    def generate_corrupted(self):
        subset = self.get_subset()
        corruption_transform = transforms.Compose(
            [AddGaussianNoise(mean=0.0, std=0.25), ClampRangeTransform()]
        )

        self.corrupted = [(corruption_transform(x), y) for x, y in subset]

    def get_subset(self) -> Subset:
        subset_indices = torch.multinomial(
            torch.ones(len(self.remaining_indices)),
            self.num_probes,
            replacement=False,
            generator=self.generator,
        )

        self.remaining_indices = [
            self.remaining_indices[i]
            for i in range(len(self.remaining_indices))
            if i not in subset_indices
        ]

        return Subset(self.dataset, subset_indices.tolist())


def make_probe_suites(
    input_filepath: str, output_filepath: str, dataset: str, num_probes: int = 250
):
    imagenet_dataset = torch.load(os.path.join(input_filepath, "imagenet", "train.pt"))
    imagenet_dataset_len = len(imagenet_dataset)
    imagenet_label_count = 1000

    probe_suite = ProbeSuiteGenerator(
        imagenet_dataset,
        imagenet_dataset_len,
        imagenet_label_count,
        num_probes=num_probes,
    )

    probe_suite.generate()

    print(len(probe_suite.corrupted))
    print(len(probe_suite.random_outputs))
    print(len(probe_suite.random_inputs_outputs))


make_probe_suites("data/processed", "data/processed", "imagenet")
