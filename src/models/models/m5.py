import lightning as L
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchmetrics.classification import Accuracy

from src.models.utils.create_model import create_m5_model
from src.models.utils.loss_curve import LossCurve
from src.models.utils.proxy_output import ProxyOutput


class M5(L.LightningModule):
    def __init__(
        self,
        max_epochs: int = 100,
        lr: float = 0.01,
        weight_decay: float = 0.0001,
        gamma: float = 0.1,
        step_size: int = 20,
        should_compile: bool = True,
        use_proxy_logger: bool = False,
        num_classes: int = 35,
        sync_dist_train: bool = False,
        sync_dist_val: bool = False,

    ):
        super().__init__()
        self.model = create_m5_model(
            should_compile=should_compile,
        )

        self.max_epochs = max_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.step_size = step_size
        self.use_proxy_logger = use_proxy_logger
        self.sync_dist_train = sync_dist_train
        self.sync_dist_val = sync_dist_val

        self.save_hyperparameters(ignore=["model"])

        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        (x, y), indices = batch

        output = self(x)
        loss = F.nll_loss(output.squeeze(), y)

        self.log(
            "train/loss",
            loss.mean(),
            on_step=True,
            on_epoch=False,
            sync_dist=self.sync_dist_train,
        )

        if self.use_proxy_logger:
            return ProxyOutput.create(loss, indices, y, x)

        return LossCurve.create(loss, indices, y, x)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:
            x, y = batch
            indices = None
        else:
            (x, y), indices = batch

        logits = self(x)
        loss = F.nll_loss(logits.squeeze(), y)

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
            loss.mean(),
            on_step=False,
            on_epoch=True,
            sync_dist=self.sync_dist_val,
        )

        if self.use_proxy_logger:
            return ProxyOutput.create(loss, indices, y, logits)

        return LossCurve.create(loss, indices, y, logits)

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)
        loss = F.nll_loss(y_hat.squeeze(), y)
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
        optimizer = Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        scheduler_dict = {
            "scheduler": StepLR(optimizer, step_size=self.step_size, gamma=self.gamma),
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
