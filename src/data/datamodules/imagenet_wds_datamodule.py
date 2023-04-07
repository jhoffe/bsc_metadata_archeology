from multiprocessing import cpu_count
from typing import Optional

import braceexpand
import lightning as L
import torch
import webdataset as wds
from torch.utils.data import Dataset
from torchvision.transforms import transforms


def get_target(x):
    return x["target"]


def my_node_splitter(urls):
    """Split urls_ correctly per accelerator node
    :param urls:
    :return: slice of urls_
    """
    rank = torch.distributed.get_rank()
    num_replicas = torch.distributed.get_world_size()

    urls_this = urls[rank::num_replicas]

    return urls_this


class ImageNetWDSDataModule(L.LightningDataModule):
    imagenet_train: Dataset
    imagenet_val: Dataset
    num_workers: int

    def __init__(
        self,
        training_urls: str = "",
        val_urls: str = "",
        batch_size: int = 128,
        num_workers: Optional[int] = None,
    ):
        """Initializes the data module.

        Args:
            data_dir: str, directory where the imagenet dataset is stored.
            batch_size: int, size of the mini-batch.
            num_workers: int, number of worker to use for data loading
        """
        super().__init__()
        self.training_urls = training_urls
        self.val_urls = val_urls
        self.batch_size = batch_size
        self.num_workers = cpu_count() if num_workers is None else num_workers

    def make_transform(self, mode="train"):
        if mode == "train":
            return transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
        elif mode == "val":
            return transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                ]
            )

    def make_loader(self, urls, mode="train"):
        if mode == "train":
            dataset_size = 1281167
            shuffle = 5000
        elif mode == "val":
            dataset_size = 5000
            shuffle = 0

        num_instances = self.num_workers

        transform = self.make_transform(mode=mode)

        dataset = (
            wds.WebDataset(urls, nodesplitter=my_node_splitter)
            .shuffle(shuffle)
            .decode("pil")
            .to_tuple("jpg;png;jpeg json")
            .map_tuple(transform, get_target)
            .batched(self.batch_size, partial=True)
        )

        loader = wds.WebLoader(
            dataset, batch_size=None, shuffle=False, num_workers=self.num_workers
        )

        loader.length = dataset_size // self.batch_size

        num_batches = loader.length // num_instances
        loader.with_epoch(dataset_size)
        loader.with_length(num_batches)

        loader.dataset = dataset

        return loader

    def convert_url_string_to_list(self, urls):
        return braceexpand.braceexpand(urls)

    def train_dataloader(self):
        return self.make_loader(
            self.convert_url_string_to_list(self.training_urls), mode="train"
        )

    def val_dataloader(self):
        return self.make_loader(
            self.convert_url_string_to_list(self.val_urls), mode="val"
        )
