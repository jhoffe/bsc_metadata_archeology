import os
from os import path

import torch
from torchvision import transforms

from src.data.datasets import ImageNetTrainingDataset, ImagenetValidationDataset
from src.data.imagenet_c_scores import imagenet_c_scores


def imagenet_transform(input_filepath: str, output_filepath: str) -> None:
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

    dataset_train = ImageNetTrainingDataset(
        path.join(input_filepath, "imagenet/ILSVRC/Data/CLS-LOC/train"),
        c_scores=imagenet_c_scores(use_cscores=True),
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
