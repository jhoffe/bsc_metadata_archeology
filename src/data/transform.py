import logging
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv

from src.data.utils.imagenet_transform import imagenet_transform


def dataset_transform(input_filepath: str, output_filepath: str, dataset: str) -> None:
    if "imagenet" in dataset:
        imagenet_transform(input_filepath, output_filepath, dataset)


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.argument("dataset", type=str)
def main(input_filepath: str, output_filepath: str, dataset: str) -> None:
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    Args:
        - input_filepath: Filepath for the raw data set.
        - output_filepath: Filepath for where to save the processed data set.
        - seed: Seed for the data set split to train/test.
    """

    logger = logging.getLogger(__name__)
    logger.info(f"Transforming the {dataset} dataset")

    dataset_transform(input_filepath, output_filepath, dataset)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
