import lightning as L
from torch.nn import functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.classification import Accuracy

from src.models.utils.create_model import create_resnet50_model


class ResNet50WDS(L.LightningModule):
    def __init__(
        self,
        max_epochs: int = 100,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
        sync_dist_train: bool = False,
        sync_dist_val: bool = False,
        num_classes=1000,
        resize_conv1=False,
        should_compile: bool = True,
    ):
        super().__init__()
        self.model = create_resnet50_model(
            num_classes, resize_conv1=resize_conv1, should_compile=should_compile
        )

        self.max_epochs = max_epochs
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.sync_dist_train = sync_dist_train
        self.sync_dist_val = sync_dist_val

        self.save_hyperparameters(ignore=["model"])

        self.val1_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val5_accuracy = Accuracy(
            task="multiclass", num_classes=num_classes, top_k=5
        )
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)
        loss = F.cross_entropy(logits, y, reduction="none")
        mean_loss = loss.mean()

        self.log(
            "train/loss",
            mean_loss,
            on_step=True,
            on_epoch=False,
            sync_dist=self.sync_dist_train,
        )

        return mean_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch

        logits = self(x)
        loss = F.cross_entropy(logits, y, reduction="none")
        mean_loss = loss.mean()

        self.val1_accuracy(logits, y)
        self.val5_accuracy(logits, y)
        self.log(
            "validation/accuracy_top1",
            self.val1_accuracy,
            on_step=True,
            on_epoch=True,
            sync_dist=self.sync_dist_val,
        )
        self.log(
            "validation/accuracy_top5",
            self.val5_accuracy,
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
