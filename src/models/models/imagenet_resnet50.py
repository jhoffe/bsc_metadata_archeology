import lightning as L
import torch
from torch.nn import functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.classification import Accuracy
from torchvision.models.resnet import resnet50


class ImageNetResNet50(L.LightningModule):
    def __init__(
        self,
        max_epochs: int = 100,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
        sync_dist_train: bool = False,
        sync_dist_val: bool = False,
    ):
        super().__init__()
        self.model = torch.compile(resnet50(weights=None))

        self.max_epochs = max_epochs
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.sync_dist_train = sync_dist_train
        self.sync_dist_val = sync_dist_val

        self.save_hyperparameters(ignore=["model"])

        self.val_accuracy = Accuracy(task="multiclass", num_classes=1000)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=1000)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        (x, y, _), indices = batch

        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y, reduction="none")
        mean_loss = loss.mean()

        self.log(
            "train/loss",
            mean_loss,
            on_step=True,
            on_epoch=False,
            sync_dist=self.sync_dist_train,
        )

        y_pred = y_hat.argmax(dim=1)
        return {
            "loss": mean_loss,
            "unreduced_loss": loss,
            "indices": indices,
            "y": y,
            "y_hat": y_pred,
        }

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:
            x, y = batch
            indices = None
        else:
            (x, y, _), indices = batch

        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y, reduction="none")
        mean_loss = loss.mean()

        self.val_accuracy(y_hat, y)
        self.log(
            "val/accuracy",
            self.val_accuracy,
            on_step=True,
            on_epoch=True,
            sync_dist=self.sync_dist_val,
        )
        self.log(
            "validation/loss",
            mean_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=self.sync_dist_val,
        )

        y_pred = y_hat.argmax(dim=1)
        return {
            "loss": mean_loss,
            "unreduced_loss": loss,
            "indices": indices,
            "y": y,
            "y_hat": y_pred,
        }

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y, reduction="none")
        self.test_accuracy.update(y_hat, y)
        self.log(
            "test/loss",
            loss.mean(),
            on_step=False,
            on_epoch=True,
            sync_dist=self.sync_dist_val,
        )

    def test_epoch_end(self, outputs) -> None:
        self.log(
            "testing/accuracy",
            self.test_accuracy.compute(),
            sync_dist=self.sync_dist_val,
        )

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
