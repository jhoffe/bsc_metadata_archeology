from omegaconf import DictConfig

from src.data.datamodules import CIFARDataModule, ImageNetDataModule, SCDataModule


def create_datamodule(params: DictConfig):
    DATAMODULES = {
        "imagenet": ImageNetDataModule,
        "cifar": CIFARDataModule,
        "speechcommands": SCDataModule,
    }

    data_params = params.dataset
    datamodule_name = data_params["name"]

    if datamodule_name not in DATAMODULES.keys():
        raise ValueError(f"name '{datamodule_name}' not in data modules")

    return DATAMODULES[datamodule_name](
        **{key: value for key, value in data_params.items() if key != "name"}
    )
