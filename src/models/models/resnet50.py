import lightning as L
from torch.nn import functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.classification import Accuracy

from src.models.utils.create_model import create_resnet50_model
from src.models.utils.loss_curve import LossCurve
from src.models.utils.proxy_output import ProxyOutput


class ResNet50(L.LightningModule):
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
        use_proxy_logger: bool = False,
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
        self.use_proxy_logger = use_proxy_logger

        self.save_hyperparameters(ignore=["model"])

        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        (x, y), indices = batch

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

        if self.use_proxy_logger:
            return ProxyOutput.create(mean_loss, indices, y, logits)

        return LossCurve.create(loss, indices, y, logits)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:
            x, y = batch
            indices = None
        else:
            (x, y), indices = batch

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

        if self.use_proxy_logger:
            return ProxyOutput.create(mean_loss, indices, y, logits)

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
