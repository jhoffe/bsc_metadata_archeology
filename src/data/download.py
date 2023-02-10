# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from torchvision.datasets import CIFAR10, CIFAR100

def download_dataset(input_filepath: str, dataset: str) -> None:
    """Downloads the dataset to the chosen filepath."""
    Dataset = CIFAR100 if dataset == "cifar100" else CIFAR10 if dataset == "cifar10" else None
    assert Dataset is not None
    Dataset(input_filepath, download=True)

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('dataset', type=str)
def main(input_filepath: str, dataset: str) -> None:
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Downloading dataset')
    download_dataset(input_filepath, dataset)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
