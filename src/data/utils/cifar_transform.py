import os

import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100


def cifar_transform(input_filepath: str, output_filepath: str, dataset: str) -> None:
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    Dataset = (
        CIFAR100 if dataset == "cifar100" else CIFAR10 if dataset == "cifar10" else None
    )
    assert Dataset is not None

    train_data = Dataset(
        root=input_filepath, train=True, download=False, transform=transform_train
    )
    test_data = Dataset(
        root=input_filepath, train=False, download=False, transform=transform_test
    )

    output_dir = os.path.join(output_filepath, dataset)
    os.makedirs(output_dir, exist_ok=True)

    torch.save(train_data, os.path.join(output_dir, "train.pt"))
    torch.save(test_data, os.path.join(output_dir, "test.pt"))
