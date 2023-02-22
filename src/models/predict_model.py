import os
from time import gmtime, strftime

import hydra
import pytorch_lightning as pl
import torch
from dotenv import find_dotenv, load_dotenv
from omegaconf import OmegaConf
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

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
        MODULES[module_name].load_from_checkpoint(params["checkpoint"]),
        DATAMODULES[datamodule_name](**params["data_params"]),
    )


def create_trainer(params: dict):
    time_dir = strftime("%Y%m%d_%H%M", gmtime())

    log_model = params["log_model"] if "log_model" in params.keys() else True
    logger = (
        [
            WandbLogger(
                name=params["run_name"] + "-" + time_dir,
                project="bsc",
                save_dir="models/",
                config=params,
                log_model=log_model
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

    callbacks = []
    if "log_loss_curves" in params.keys() and params["log_loss_curves"]:
        callbacks.append(LossCurveLogger(f"models/losses/{time_dir}", time_dir))

    log_every_n_steps = params["log_every_n"] if "log_every_n" in params.keys() else 50

    profiler = params["profiler"] if "profiler" in params.keys() else None
    limit_train_batches = params["limit_train_batches"] if "limit_train_batches" in params.keys() else None
    limit_val_batches = params["limit_val_batches"] if "limit_val_batches" in params.keys() else None

    return pl.Trainer(
        accelerator=params["accelerator"],
        devices=params["devices"],
        strategy=strategy,
        num_nodes=params["num_nodes"],
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        logger=logger,
        precision=precision,
        callbacks=callbacks,
        log_every_n_steps=log_every_n_steps,
        profiler=profiler
    )


@hydra.main(
    version_base=None,
    config_path=os.path.join(os.getcwd(), "config/imagenet"),
    config_name="default_config.yaml",
)
def train(config):
    print(f"configuration: \n {OmegaConf.to_yaml(config.testing)}")

    hparams = config.testing
    pl.seed_everything(hparams["seed"])

    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)

    if hparams["matmul_precision"] is not None:
        torch.set_float32_matmul_precision(hparams["matmul_precision"])

    module, datamodule = create_module_and_data(hparams)
    trainer = create_trainer(hparams)
    trainer.test(module, datamodule=datamodule)


if __name__ == "__main__":
    train()
