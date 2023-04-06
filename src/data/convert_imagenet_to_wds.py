import logging
import os
import random

import click
import webdataset as wds
from tqdm import tqdm

from src.data.datasets import ImageNetTrainingDataset, ImagenetValidationDataset


def readfile(fname):
    """Read a binary file from disk."""
    with open(fname, "rb") as stream:
        return stream.read()


@click.command()
@click.argument("train_path", type=click.Path(exists=True))
@click.argument("val_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path(exists=True))
@click.option("--maxsize", default=1e9, help="maximum size of each shard")
@click.option("--maxcount", default=1000, help="maximum number of samples per shard")
@click.option("--split", default="train", help="which split to write")
def write_to_wbs(
    train_path, val_path, output_path, maxsize: int, maxcount: int, split: str
):
    logger = logging.getLogger(__name__)

    logger.info("Reading imagenet training dataset")
    train_ds = ImageNetTrainingDataset(train_path)
    logger.info("Reading imagenet validation dataset")
    val_ds = ImagenetValidationDataset(val_path, class_to_idx=train_ds.class_to_idx)

    if split == "train":
        ds = train_ds
    elif split == "val":
        ds = val_ds
    else:
        raise ValueError(f"Unknown split: {split}")

    all_keys = set()
    logger.info(f"Writing {split} dataset")
    pattern = os.path.join(output_path, f"imagenet-{split}-%06d.tar")

    indices = list(range(len(ds)))
    random.shuffle(indices)

    with wds.ShardWriter(pattern, maxcount=maxcount, maxsize=maxsize) as sink:
        for i in tqdm(indices, desc=f"Writing {split}"):
            fname, cls = ds.samples[i]
            image = readfile(fname)

            fname_key = os.path.splitext(os.path.basename(fname))[0]

            assert fname_key not in all_keys
            all_keys.add(fname_key)

            sample = {"__key__": fname_key, "jpg": image, "cls": cls}

            sink.write(sample)

    logger.info("Finished writing to webdataset")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    logger = logging.getLogger(__name__)
    logger.info("Beginning conversion")
    write_to_wbs()
