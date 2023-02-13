import csv
import os
from os import path

import torch
import torchvision.io
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.io import ImageReadMode


class ImagenetValidationDataset(Dataset):
    def __init__(
        self,
        input_filepath: str,
        class_to_idx: dict,
        transform=None,
    ) -> None:
        super().__init__()
        solution_path = path.join(input_filepath, "imagenet/LOC_val_solution.csv")
        self.transform = transform

        with open(solution_path, "r") as f:
            reader = csv.reader(f, delimiter=",")
            self.samples = [
                (
                    os.path.join(
                        input_filepath,
                        "imagenet/ILSVRC/Data/CLS-LOC/val",
                        row[0] + ".JPEG",
                    ),
                    class_to_idx[row[1].split(" ")[0]],
                )
                for i, row in enumerate(reader)
                if i != 0
            ]

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = torchvision.io.read_image(image_path, ImageReadMode.RGB)

        if self.transform is not None:
            image = self.transform(image)
        assert image.shape == torch.Size([3, 224, 224])

        return image, label

    def __len__(self):
        return len(self.samples)


def imagenet_transform(input_filepath: str, output_filepath: str, dataset: str) -> None:
    imagenet_train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    imagenet_val_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    dataset_train = ImageFolder(
        path.join(input_filepath, "imagenet/ILSVRC/Data/CLS-LOC/train"),
        transform=imagenet_train_transform,
    )
    dataset_val = ImagenetValidationDataset(
        input_filepath,
        class_to_idx=dataset_train.class_to_idx,
        transform=imagenet_val_transform,
    )

    os.makedirs(path.join(output_filepath, "imagenet"), exist_ok=True)
    torch.save(dataset_train, path.join(output_filepath, "imagenet/train.pt"))
    torch.save(dataset_val, path.join(output_filepath, "imagenet/val.pt"))
