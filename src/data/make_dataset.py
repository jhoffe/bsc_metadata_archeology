import logging
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv

from src.data.download import download_dataset
from src.data.transform import dataset_transform


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    logger.info("Downloading the CIFAR10 and CIFAR100 datasets.")
    download_dataset(input_filepath, "cifar10")
    download_dataset(input_filepath, "cifar100")

    # logger.info("Transforming the CIFAR10 dataset")
    # dataset_transform(input_filepath, output_filepath, "cifar10")
    # logger.info("Transforming the CIFAR100 dataset")
    # dataset_transform(input_filepath, output_filepath, "cifar100")
    logger.info("Transforming the ImageNet dataset")
    dataset_transform(input_filepath, output_filepath, "imagenet")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
