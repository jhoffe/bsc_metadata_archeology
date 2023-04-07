from multiprocessing import cpu_count
from typing import Optional

import lightning as L
import webdataset as wds
from torch.utils.data import Dataset
from torchvision.transforms import transforms


def get_target(x):
    return x["target"]


class WDSWithLen(wds.WebDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def with_len(self, length):
        self.length = length
        return self

    def __len__(self):
        return self.length


class WDSLoaderWithLenAndDataset(wds.WebLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def with_len(self, length):
        self.length = length
        return self

    def with_dataset(self, dataset):
        self.dataset = dataset
        return self

    def __len__(self):
        return self.length


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
        epoch_size = dataset_size // num_instances

        transform = self.make_transform(mode=mode)

        dataset = (
            WDSWithLen(urls)
            .shuffle(shuffle)
            .decode("pil")
            .to_tuple("jpg;png;jpeg json")
            .map_tuple(transform, get_target)
            .batched(self.batch_size, partial=True)
        )

        dataset.with_len(epoch_size)

        loader = wds.WebLoader(
            dataset, batch_size=None, shuffle=False, num_workers=self.num_workers
        )

        loader.with_epoch(dataset_size, epoch_size // self.batch_size)
        loader.with_length(epoch_size)

        return loader

    def train_dataloader(self):
        return self.make_loader(self.training_urls, mode="train")

    def val_dataloader(self):
        return self.make_loader(self.val_urls, mode="val")
