from typing import Optional

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from torch.nn import functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.classification import Accuracy
from torchvision.models.resnet import resnet50


class ImageNetResNet50(pl.LightningModule):
    def __init__(
        self,
        max_epochs: int = 100,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
        sync_dist: bool = False,
    ):
        super().__init__()
        self.model = resnet50(weights=None)

        self.max_epochs = max_epochs
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.sync_dist = sync_dist

        self.save_hyperparameters(ignore=["model"])

        self.val_accuracy = Accuracy(task="multiclass", num_classes=1000)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        y = F.one_hot(y, num_classes=1000).to(torch.float32)

        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y, reduction="none")
        mean_loss = loss.mean()

        self.log(
            "train/loss",
            mean_loss,
            on_step=True,
            on_epoch=False,
            sync_dist=self.sync_dist,
        )

        return {"loss": mean_loss, "unreduced_loss": loss, "batch_indices": batch_idx}

    def training_step_end(self, outputs):
        unreduced_losses = outputs["unreduced_loss"]
        batch_idx = outputs["batch_indices"]

        if self.global_rank == 0:
            all = [torch.zeros(unreduced_losses.shape, device=self.device)]*self.trainer.num_devices
            torch.distributed.gather(unreduced_losses, all)

            losses = []

            for losses in all:
                losses.append((batch_idx, losses.detach()))

            return {"losses": losses, "loss": outputs["loss"]}
        else:
            torch.distributed.gather(unreduced_losses)
        return {"losses": [], "loss": outputs["loss"]}

    def training_epoch_end(self, outputs):
        if isinstance(self.logger, WandbLogger) and self.global_rank == 0:
            epoch_losses = []

            print(outputs)

            for output in outputs:
                for batch in output:
                    epoch_losses.append(batch)

            torch.save(epoch_losses, "models/losses.pt")

            artifact = wandb.Artifact("losses", "loss")
            artifact.add_file("models/losses.pt")
            self.logger.experiment.log_artifact(artifact)

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y = F.one_hot(y, num_classes=1000).to(torch.float32)

        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y, reduction="none")

        self.val_accuracy(y_hat, y)
        self.log(
            "val/accuracy",
            self.val_accuracy,
            on_step=False,
            on_epoch=True,
            sync_dist=self.sync_dist,
        )
        self.log(
            "validation/loss",
            loss.mean(),
            on_step=False,
            on_epoch=True,
            sync_dist=self.sync_dist,
        )

    def test_step(self, batch, batch_idx):
        x, y = batch

        y = F.one_hot(y, num_classes=1000).to(torch.float32)

        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y, reduction="none")
        self.log("test/loss", loss.mean(), on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = SGD(
            self.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        scheduler_dict = {
            "scheduler": CosineAnnealingLR(optimizer, T_max=self.max_epochs),
            "interval": "epoch",
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)
