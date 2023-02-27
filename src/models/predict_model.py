import os

import hydra
import pytorch_lightning as pl
from dotenv import find_dotenv, load_dotenv
from omegaconf import DictConfig, OmegaConf

from src.models.create_datamodule import create_datamodule
from src.models.create_module import create_module
from src.models.create_trainer import create_trainer


@hydra.main(
    version_base="1.2",
    config_path=os.path.join(os.getcwd(), "config/"),
    config_name="default_config.yaml",
)
def predict(config: DictConfig):
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")

    pl.seed_everything(config["seed"])

    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)

    module = create_module(config)
    datamodule = create_datamodule(config)
    trainer = create_trainer(config)
    trainer.predict(module, datamodule=datamodule)


if __name__ == "__main__":
    predict()
