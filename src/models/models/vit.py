import lightning as L
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchmetrics.classification import Accuracy

from src.models.utils.create_model import create_vit_model
from src.models.utils.loss_curve import LossCurve


class ViT(L.LightningModule):
    def __init__(
        self,
        lr: float = 0.003,
        weight_decay: float = 0.3,
        max_epochs: int = 300,
        label_smoothing: float = 0.11,
        sync_dist_train: bool = False,
        sync_dist_val: bool = False,
        num_classes=1000,
        should_compile: bool = True,
        model_version: str = "vit_b_16",
        warmup_epochs: int = 30,
        lr_warmup_decay: float = 0.033,
    ):
        super().__init__()
        self.model = create_vit_model(
            num_classes, model_version, should_compile=should_compile
        )

        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.label_smoothing = label_smoothing
        self.sync_dist_train = sync_dist_train
        self.sync_dist_val = sync_dist_val
        self.warmup_epochs = warmup_epochs
        self.lr_warmup_decay = lr_warmup_decay

        self.save_hyperparameters(ignore=["model"])

        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        self.criterion = nn.CrossEntropyLoss(
            reduction="none", label_smoothing=self.label_smoothing
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        (x, y, _), indices = batch

        logits = self(x)
        loss = self.criterion(logits, y)
        print(loss)
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
        loss = self.criterion(logits, y)
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

        logits = self(x)
        loss = self.criterion(logits, y)
        self.test_accuracy.update(logits, y)
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
        optimizer = AdamW(self.parameters(), lr=self.lr)

        if self.warmup_epochs > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                total_iters=self.warmup_epochs,
                start_factor=self.lr_warmup_decay,
            )
            scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs)
            scheduler_dict = {
                "scheduler": SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, scheduler],
                    milestones=[self.warmup_epochs],
                ),
                "interval": "epoch",
            }
        else:
            scheduler_dict = {
                "scheduler": CosineAnnealingLR(optimizer, T_max=self.max_epochs),
                "interval": "epoch",
            }

        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
