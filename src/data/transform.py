import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from torchvision.datasets import CIFAR10, CIFAR100
import torch
from torchvision import transforms
import os

@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.argument("dataset", type=str)

def main(input_filepath: str, output_filepath: str, dataset: str):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    Args:
        - input_filepath: Filepath for the raw data set.
        - output_filepath: Filepath for where to save the processed data set.
        - seed: Seed for the data set split to train/test.
    """

    logger = logging.getLogger(__name__)
    logger.info(f"Transforming the {dataset} dataset")

    cifar_transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    cifar_transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    imagenet_train_transform = transforms.Compose(
        [   
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
    )
    
    imagenet_test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ]
    )
    

    Dataset = CIFAR100 if dataset == "cifar100" else CIFAR10 if dataset == "cifar10" else None
    assert Dataset is not None

    if "cifar" in dataset:
        trainset = Dataset(
            root=input_filepath, train=True, download=False, transform=cifar_transform_train
        )
        testset = Dataset(
            root=input_filepath, train=False, download=False, transform=cifar_transform_test
        )

    train_data = trainset
    test_data = testset

    output_dir = os.path.join(output_filepath, dataset)
    os.makedirs(output_dir, exist_ok=True)

    torch.save(train_data, os.path.join(output_dir, "train.pt"))
    torch.save(test_data, os.path.join(output_dir, "test.pt"))


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()