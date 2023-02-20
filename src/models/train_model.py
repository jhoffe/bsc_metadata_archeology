import os
from collections import defaultdict

import hydra
import pytorch_lightning as pl
import torch
from dotenv import find_dotenv, load_dotenv
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger

from src.data.datamodules import ImageNetDataModule
from src.models.models import ImageNetResNet50
from pytorch_lightning.plugins.environments import ClusterEnvironment

MASTER_ADDR = os.environ['MASTER_ADDR']
MASTER_PORT = os.environ['MASTER_PORT']
LOCAL_RANK = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
WORLD_SIZE = int(os.environ['OMPI_COMM_WORLD_SIZE'])
WORLD_RANK = int(os.environ['OMPI_COMM_WORLD_RANK'])
class MyClusterEnvironment(ClusterEnvironment):
    @property
    def creates_processes_externally(self) -> bool:
        """Return True if the cluster is managed (you don't launch processes yourself)"""
        return True

    def world_size(self) -> int:
        return int(os.environ["OMPI_COMM_WORLD_SIZE"])

    def global_rank(self) -> int:
        return int(os.environ["OMPI_COMM_WORLD_RANK"])

    def local_rank(self) -> int:
        return int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])

    def node_rank(self) -> int:
        return int(os.environ["OMPI_COMM_WORLD_NODE_RANK"])

    def main_address(self) -> str:
        return os.environ["MASTER_ADDRESS"]

    def main_port(self) -> int:
        return int(os.environ["MASTER_PORT"])



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
