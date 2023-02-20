import os

import hydra
import pytorch_lightning as pl
import torch
from dotenv import find_dotenv, load_dotenv
from omegaconf import OmegaConf
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from time import gmtime, strftime

from src.data.datamodules import ImageNetDataModule
from src.models.callbacks import LossCurveLogger
from src.models.models import ImageNetResNet50


def create_module_and_data(params: dict):
    MODULES = {"imagenet-resnet50": ImageNetResNet50}

    DATAMODULES = {"imagenet": ImageNetDataModule}

    module_name = params["model"]
    datamodule_name = params["datamodule"]

    if module_name not in MODULES.keys():
        raise ValueError(f"model '{module_name}' not in modules")

    if datamodule_name not in DATAMODULES.keys():
        raise ValueError(f"name '{datamodule_name}' not in data modules")

    return (
        MODULES[module_name](**params["model_params"]),
        DATAMODULES[datamodule_name](**params["data_params"]),
    )


def create_trainer(params: dict):
    time_dir = strftime("%Y%m%d_%H%M", gmtime())

    logger = (
        [
            WandbLogger(
                name=params["run_name"] + "-" + time_dir,
                project="bsc",
                save_dir=f"models/",
                config=params,
            )
        ]
        if params["logger"] == "wandb"
        else []
    )

    precision = params["precision"]

    if params["use_bf16_if_ampere"]:
        precision = "bf16"

    strategy = (
        params["strategy"]
        if params["strategy"] != "ddp"
        else DDPStrategy(find_unused_parameters=False)
    )

    return pl.Trainer(
        accelerator=params["accelerator"],
        devices=params["devices"],
        max_epochs=params["n_epochs"],
        strategy=strategy,
        num_nodes=params["num_nodes"],
        limit_train_batches=params["limit_train_batches"],
        logger=logger,
        precision=precision,
        callbacks=[LossCurveLogger(f"models/losses/{time_dir}", time_dir)],
    )


@hydra.main(
    version_base=None,
    config_path=os.path.join(os.getcwd(), "config/imagenet"),
    config_name="default_config.yaml",
)
def train(config):
    print(f"configuration: \n {OmegaConf.to_yaml(config.training)}")

    hparams = config.training
    pl.seed_everything(hparams["seed"])

    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)

    if hparams["matmul_precision"] is not None:
        torch.set_float32_matmul_precision(hparams["matmul_precision"])

    module, datamodule = create_module_and_data(hparams)
    trainer = create_trainer(hparams)
    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    train()
