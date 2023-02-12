import os

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from src.data.datamodules.imagenet_datamodule import ImageNetDataModule
from src.models.models.imagenet import ImageNet


@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(), "config/imagenet"), config_name="default_config.yaml")
def train(config):
    print(f"configuration: \n {OmegaConf.to_yaml(config.training)}")

    hparams = config.training
    pl.seed_everything(hparams["seed"])

    imagenet_module = ImageNet()
    datamodule = ImageNetDataModule(batch_size=hparams["batch_size"], num_workers=hparams["num_workers"])

    trainer = pl.Trainer(
        accelerator=hparams["accelerator"],
        devices=hparams["devices"],
        max_epochs=hparams["n_epochs"],
        strategy=hparams["strategy"],
        num_nodes=hparams["num_nodes"]
    )
    trainer.fit(imagenet_module, datamodule=datamodule)


if __name__ == "__main__":
    train()
