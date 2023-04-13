import os

import hydra
from dotenv import find_dotenv, load_dotenv
from omegaconf import DictConfig, OmegaConf

from src.models.utils.create_datamodule import create_datamodule
from src.models.utils.create_module import create_module
from src.models.utils.create_trainer import create_trainer


def train(config: DictConfig):
    module = create_module(config)
    datamodule = create_datamodule(config)
    trainer = create_trainer(config)

    return trainer, module, datamodule


@hydra.main(
    version_base="1.2",
    config_path=os.path.join(os.getcwd(), "config/"),
    config_name="default_config.yaml",
)
def main(config: DictConfig):
    import lightning as L

    print(f"configuration: \n {OmegaConf.to_yaml(config)}")

    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)

    L.seed_everything(config["seed"])

    trainer, module, datamodule = train(config)
    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
