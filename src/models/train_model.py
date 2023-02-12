from src.data.datamodules.imagenet_datamodule import ImageNetDataModule
from src.models.models.imagenet import ImageNet
import pytorch_lightning as pl
import torch


def main():
    imagenet_module = ImageNet()
    datamodule = ImageNetDataModule(batch_size=128, num_workers=8)

    torch.set_float32_matmul_precision("medium")

    trainer = pl.Trainer(accelerator="auto", devices=-1, max_epochs=1)
    trainer.fit(imagenet_module, datamodule=datamodule)


if __name__ == "__main__":
    main()
