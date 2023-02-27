from time import gmtime, strftime

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from src.models.callbacks import LossCurveLogger


def create_trainer(params: DictConfig):
    time_dir = strftime("%Y%m%d_%H%M", gmtime())

    trainer_params = params.trainer

    if trainer_params["matmul_precision"] is not None:
        torch.set_float32_matmul_precision(trainer_params["matmul_precision"])

    log_model = (
        trainer_params["log_model"] if "log_model" in trainer_params.keys() else True
    )
    logger = (
        [
            WandbLogger(
                name=params["name"] + "-" + time_dir,
                project="bsc",
                save_dir="models/",
                config=params,
                log_model=log_model,
            )
        ]
        if trainer_params["logger"] == "wandb"
        else []
    )

    precision = trainer_params["precision"]

    if trainer_params["use_bf16_if_ampere"]:
        precision = "bf16"

    strategy = (
        trainer_params["strategy"]
        if trainer_params["strategy"] != "ddp"
        else DDPStrategy(find_unused_parameters=False)
    )

    callbacks = []
    if "log_loss_curves" in trainer_params.keys() and trainer_params["log_loss_curves"]:
        callbacks.append(LossCurveLogger(f"models/losses/{time_dir}", time_dir))

    log_every_n_steps = (
        trainer_params["log_every_n"] if "log_every_n" in trainer_params.keys() else 50
    )

    profiler = (
        trainer_params["profiler"] if "profiler" in trainer_params.keys() else None
    )
    limit_train_batches = (
        trainer_params["limit_train_batches"]
        if "limit_train_batches" in trainer_params.keys()
        else None
    )
    limit_val_batches = (
        trainer_params["limit_val_batches"]
        if "limit_val_batches" in trainer_params.keys()
        else None
    )

    return pl.Trainer(
        accelerator=trainer_params["accelerator"],
        devices=trainer_params["devices"],
        max_epochs=trainer_params["n_epochs"],
        strategy=strategy,
        num_nodes=trainer_params["num_nodes"],
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        logger=logger,
        precision=precision,
        callbacks=callbacks,
        log_every_n_steps=log_every_n_steps,
        profiler=profiler,
    )
