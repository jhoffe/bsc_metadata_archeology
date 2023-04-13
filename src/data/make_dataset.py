import logging
from pathlib import Path

import click
import pytorch_lightning as pl
from dotenv import find_dotenv, load_dotenv

from src.data.cifar_c_scores import c_scores_dataset
from src.data.download import download_dataset
from src.data.make_probe_suites import make_probe_suites
from src.data.utils.imagenet_transform import imagenet_transform


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
    download_dataset(input_filepath, "cifar100")

    logger.info("Transforming the CIFAR10 dataset w. no c-scores")
    c_scores_dataset("cifar10", "data/raw", "data/processed")
    logger.info("Transforming the CIFAR10 dataset w. c-scores")
    c_scores_dataset("cifar10", "data/raw", "data/processed", use_c_scores=True)

    logger.info("Transforming the CIFAR100 dataset w. no c-scores")
    c_scores_dataset("cifar100", "data/raw", "data/processed")
    logger.info("Transforming the CIFAR100 dataset w. c-scores")
    c_scores_dataset("cifar100", "data/raw", "data/processed", use_c_scores=True)
    logger.info("Transforming the CIFAR100 dataset w. mem-scores")
    c_scores_dataset("cifar100", "data/raw", "data/processed", use_c_scores=False)

    logger.info("Transforming the ImageNet dataset w. no c-scores")
    imagenet_transform(input_filepath, output_filepath)
    logger.info("Transforming the ImageNet dataset w. c-scores")
    imagenet_transform(input_filepath, output_filepath, use_c_scores=True)
    logger.info("Transforming the ImageNet dataset w. mem-scores")
    imagenet_transform(input_filepath, output_filepath, use_c_scores=False)
    logger.info("Transformed the ImageNet dataset")

    # Generating probe default_suites
    logger.info("Generating probe suites for CIFAR10 w. C-scores")
    make_probe_suites(
        "data/processed", "data/processed", "cifar10", label_count=10, use_c_scores=True
    )

    logger.info("Generating probe suites for CIFAR100 wo. mem-scores")
    make_probe_suites(
        "data/processed",
        "data/processed",
        "cifar100",
        label_count=100,
        use_c_scores=False,
    )
    logger.info("Generating probe suites for CIFAR100 w. C-scores")
    make_probe_suites(
        "data/processed",
        "data/processed",
        "cifar100",
        label_count=100,
        use_c_scores=True,
    )

    logger.info("Generating probe suites for ImageNet w. C-scores")
    make_probe_suites(
        "data/processed",
        "data/processed",
        "imagenet",
        label_count=1000,
        use_c_scores=True,
    )
    logger.info("Generating probe suites for ImageNet wo. mem-scores")
    make_probe_suites(
        "data/processed",
        "data/processed",
        "imagenet",
        label_count=1000,
        use_c_scores=False,
    )

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
