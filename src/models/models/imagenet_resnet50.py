import pytorch_lightning as pl
import torch
from torch.nn import Module
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
    ):
        super().__init__()
        self.model = resnet50(weights=None)

        self.max_epochs = max_epochs
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.save_hyperparameters(ignore=["model"])

        self.val_accuracy = Accuracy(task="multiclass", num_classes=1000)

    @property
    def should_sync_dist(self):
        return self.trainer.strategy == "ddp"

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        y = F.one_hot(y, num_classes=1000).to(torch.float32)

        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        self.log("train/loss", loss, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y = F.one_hot(y, num_classes=1000).to(torch.float32)

        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        self.val_accuracy(y_hat, y)
        self.log(
            "val/accuracy",
            self.val_accuracy,
            on_step=False,
            on_epoch=True,
            sync_dist=self.should_sync_dist,
        )
        self.log("validation/loss", loss, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch

        y = F.one_hot(y, num_classes=1000).to(torch.float32)

        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("test/loss", loss, on_step=False, on_epoch=True)

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
