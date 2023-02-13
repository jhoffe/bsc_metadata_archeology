import pytorch_lightning as pl
import torch
from torch.nn import Module
from torch.nn import functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from torchvision.models.resnet import resnet50

imagenet_model: Module = resnet50(weights=None)


class ImageNet(pl.LightningModule):
    def __init__(self, model: Module = imagenet_model, max_epochs: int = 100):
        super().__init__()
        self.model = model
        self.max_epochs = max_epochs
        self.save_hyperparameters(ignore=["model"])

        self.val_metrics = MetricCollection(
            {
                "val/accuracy": MulticlassAccuracy(num_classes=1000),
                "val/f1_score": MulticlassF1Score(num_classes=1000),
            }
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        y = F.one_hot(y, num_classes=1000).to(torch.float32)

        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        self.log("train/loss", loss, sync_dist=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y = F.one_hot(y, num_classes=1000).to(torch.float32)

        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.val_metrics.update(y_hat, y)

        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, sync_dist=True)
        self.log("validation/loss", loss, sync_dist=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch

        y = F.one_hot(y, num_classes=1000).to(torch.float32)

        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("test/loss", loss, sync_dist=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)

        scheduler_dict = {
            "scheduler": CosineAnnealingLR(optimizer, T_max=self.max_epochs),
            "interval": "epoch",
        }

        return {"optimizer": optimizer, "scheduler": scheduler_dict}
