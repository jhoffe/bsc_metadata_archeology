import os
from os import path
from typing import Optional

import torch
from torchvision import transforms

from src.data.datasets import ImageNetTrainingDataset, ImagenetValidationDataset
from src.data.imagenet_c_scores import imagenet_c_scores


def imagenet_transform(
    input_filepath: str, output_filepath: str, use_c_scores: Optional[bool] = None
) -> None:
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
        c_scores=imagenet_c_scores(use_c_scores),
        transform=imagenet_train_transform,
    )
    dataset_val = ImagenetValidationDataset(
        path.join(input_filepath, "imagenet"),
        class_to_idx=dataset_train.class_to_idx,
        transform=imagenet_val_transform,
    )

    os.makedirs(path.join(output_filepath, "imagenet"), exist_ok=True)
    torch.save(dataset_val, path.join(output_filepath, "imagenet/val.pt"))
    if use_c_scores is not None:
        torch.save(
            dataset_train, path.join(output_filepath, "imagenet/train_c_scores.pt")
        ) if use_c_scores else torch.save(
            dataset_train, path.join(output_filepath, "imagenet/train_mem_scores.pt")
        )
    else:
        torch.save(dataset_train, path.join(output_filepath, "imagenet/train.pt"))
