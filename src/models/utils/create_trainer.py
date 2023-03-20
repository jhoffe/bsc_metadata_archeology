from time import gmtime, strftime

import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import DictConfig

from src.models.callbacks import LossCurveLogger


def create_trainer(params: DictConfig):
    time_dir = strftime("%Y%m%d_%H%M", gmtime())

    trainer_params = params.trainer

    if trainer_params["matmul_precision"] is not None:
        torch.set_float32_matmul_precision(trainer_params["matmul_precision"])

    log_model = (
        trainer_params["log_model"] if "log_model" in trainer_params.keys() else True
    )

    run_name = params["name"] + "-" + time_dir

    logger = (
        [
            WandbLogger(
                name=run_name,
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
        callbacks.append(LossCurveLogger(f"models/losses/{run_name}", time_dir))

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

    clip_grad_norm = (
        trainer_params["clip_grad_norm"]
        if "clip_grad_norm" in trainer_params.keys()
        else None
    )

    return L.Trainer(
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
        clip_grad_norm=clip_grad_norm,
    )
