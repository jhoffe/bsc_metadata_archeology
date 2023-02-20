import os
from collections import defaultdict
import socket

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from dotenv import find_dotenv, load_dotenv
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger

from src.data.datamodules import ImageNetDataModule
from src.models.models import ImageNetResNet50
from pytorch_lightning.plugins.environments import ClusterEnvironment
from typing import Optional


class MyClusterEnvironment(ClusterEnvironment):
    def __init__(self) -> None:

        from mpi4py import MPI

        self._comm_world = MPI.COMM_WORLD
        self._comm_local: Optional[MPI.Comm] = None
        self._node_rank: Optional[int] = None
        self._main_address: Optional[str] = None
        self._main_port: Optional[int] = None

    @property
    def creates_processes_externally(self) -> bool:
        return True

    @property
    def main_address(self) -> str:
        if self._main_address is None:
            self._main_address = self._get_main_address()
        return self._main_address

    @property
    def main_port(self) -> int:
        if self._main_port is None:
            self._main_port = self._get_main_port()
        return self._main_port

    @staticmethod
    def detect() -> bool:
        """Returns ``True`` if the `mpi4py` package is installed and MPI returns a world size greater than 1."""

        from mpi4py import MPI

        return MPI.COMM_WORLD.Get_size() > 1


    def world_size(self) -> int:
        return self._comm_world.Get_size()


    def set_world_size(self, size: int) -> None:
        pass

    def global_rank(self) -> int:
        return self._comm_world.Get_rank()


    def set_global_rank(self, rank: int) -> None:
        pass

    def local_rank(self) -> int:
        if self._comm_local is None:
            self._init_comm_local()
        assert self._comm_local is not None
        return self._comm_local.Get_rank()


    def node_rank(self) -> int:
        if self._node_rank is None:
            self._init_comm_local()
        assert self._node_rank is not None
        return self._node_rank


    def _get_main_address(self) -> str:
        return self._comm_world.bcast(socket.gethostname(), root=0)


    def _get_main_port(self) -> int:
        return self._comm_world.bcast(29400, root=0)

    def _init_comm_local(self) -> None:
        hostname = socket.gethostname()
        all_hostnames = self._comm_world.gather(hostname, root=0)
        # sort all the hostnames, and find unique ones
        unique_hosts = np.unique(all_hostnames)
        unique_hosts = self._comm_world.bcast(unique_hosts, root=0)
        # find the integer for this host in the list of hosts:
        self._node_rank = int(np.where(unique_hosts == hostname)[0])
        self._comm_local = self._comm_world.Split(color=self._node_rank)


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
    logger = (
        [WandbLogger(name=params["run_name"], project="bsc", save_dir="models/", config=params)]
        if params["logger"] == "wandb"
        else []
    )

    precision = params["precision"]

    if params["use_bf16_if_ampere"]:
        precision = "bf16"

    return pl.Trainer(
        accelerator=params["accelerator"],
        devices=params["devices"],
        max_epochs=params["n_epochs"],
        strategy=params["strategy"],
        num_nodes=params["num_nodes"],
        limit_train_batches=params["limit_train_batches"],
        logger=logger,
        precision=precision,
        plugins=[MyClusterEnvironment()],
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
