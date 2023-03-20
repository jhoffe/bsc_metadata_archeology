import lightning as L
from torch.nn import functional as F
from torch.optim import AdamW
from torchmetrics.classification import Accuracy

from src.models.utils.create_model import create_vit_model
from src.models.utils.loss_curve import LossCurve


class ViT(L.LightningModule):
    def __init__(
        self,
        lr: float = 0.1,
        sync_dist_train: bool = False,
        sync_dist_val: bool = False,
        num_classes=1000,
        should_compile: bool = True,
        model_version: str = "vit_b_16",
    ):
        super().__init__()
        self.model = create_vit_model(
            num_classes, model_version, should_compile=should_compile
        )

        self.lr = lr
        self.sync_dist_train = sync_dist_train
        self.sync_dist_val = sync_dist_val

        self.save_hyperparameters(ignore=["model"])

        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        (x, y, _), indices = batch

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

        return LossCurve.create(loss, indices, y, logits)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:
            x, y = batch
            indices = None
        else:
            (x, y, _), indices = batch

        logits = self(x)
        loss = F.cross_entropy(logits, y, reduction="none")
        mean_loss = loss.mean()

        self.val_accuracy(logits, y)
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

        return LossCurve.create(loss, indices, y, logits)

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
        return AdamW(self.parameters(), lr=self.lr)
