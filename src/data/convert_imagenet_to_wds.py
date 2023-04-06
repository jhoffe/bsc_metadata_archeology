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
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path(exists=True))
@click.option("--maxsize", default=1e9, help="maximum size of each shard")
@click.option("--maxcount", default=1000, help="maximum number of samples per shard")
def write_to_wbs(input_path, output_path, maxsize: int, maxcount: int):
    all_keys = set()
    logger = logging.getLogger(__name__)

    logger.log("Reading imagenet training dataset")
    train_ds = ImageNetTrainingDataset(input_path)
    logger.log("Reading imagenet validation dataset")
    val_ds = ImagenetValidationDataset(input_path, class_to_idx=train_ds.class_to_idx)

    for split, ds in [("train", train_ds), ("val", val_ds)]:
        logger.log(f"Writing {split} dataset")
        pattern = os.path.join(output_path, f"imagenet-{split}-%06d.tar")

        indices = list(range(len(train_ds)))
        random.shuffle(indices)

        with wds.ShardWriter(pattern, maxcount=maxcount, maxsize=maxsize) as sink:
            for i in tqdm(indices, desc=f"Writing {split}"):
                fname, cls = train_ds.samples[i]
                image = readfile(fname)

                key = os.path.splitext(os.path.basename(fname))[0]

                assert key not in all_keys
                all_keys.add(key)

                sample = {"__key__": key, "jpg": image, "cls": cls, "idx": i}

                sink.write(sample)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    logger = logging.getLogger(__name__)
    logger.log("Beginning conversion")
    write_to_wbs()
