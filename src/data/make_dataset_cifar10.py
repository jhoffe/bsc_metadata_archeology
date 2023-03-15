import logging
from pathlib import Path

import click
import pytorch_lightning as pl
from dotenv import find_dotenv, load_dotenv

from src.data.c_scores import c_scores_dataset
from src.data.download import download_dataset
from src.data.make_probe_suites import make_probe_suites
from src.data.transform import dataset_transform


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    pl.seed_everything(123)
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    logger.info("Downloading the CIFAR10 and CIFAR100 datasets.")
    download_dataset(input_filepath, "cifar10")
    
    logger.info("Transforming the CIFAR10 dataset")
    c_scores_dataset("cifar10", "data/raw", "data/processed")
    
    # Generating probe suites
    logger.info("Generating probe suites for CIFAR10")
    make_probe_suites("data/processed", "data/processed", "cifar10", label_count=10)

    logger.info("Done!")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
