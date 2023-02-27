from omegaconf import DictConfig

from src.data.datamodules import ImageNetDataModule
from src.data.datamodules import CIFAR10DataModule
from src.data.datamodules import CIFAR100DataModule


def create_datamodule(params: DictConfig):
    DATAMODULES = {"imagenet": ImageNetDataModule,
                   "cifar10": CIFAR10DataModule,
                   "cifar100": CIFAR100DataModule}

    data_params = params.dataset
    datamodule_name = data_params["name"]

    if datamodule_name not in DATAMODULES.keys():
        raise ValueError(f"name '{datamodule_name}' not in data modules")

    return DATAMODULES[datamodule_name](
        **{key: value for key, value in data_params.items() if key != "name"}
    )
