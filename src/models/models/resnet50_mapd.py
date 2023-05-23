from typing import Dict, Any

from torch.nn import functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.classification import Accuracy

from src.models.utils.create_model import create_resnet50_model
from mapd.mapd_module import MAPDModule


class ResNet50MAPD(MAPDModule):
    def __init__(
        self,
        max_epochs: int = 100,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
        num_classes=1000,
        resize_conv1=False,
        should_compile: bool = True,
        use_proxy_logger: bool = False,
        proxies_output_path: str = "models/proxies_e2e",
        probes_output_path: str = "models/probes_e2e",
    ):
        super().__init__()
        self.model = create_resnet50_model(
            num_classes, resize_conv1=resize_conv1, should_compile=should_compile
        )

        self.max_epochs = max_epochs
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.use_proxy_logger = use_proxy_logger
        self.proxies_output_path = proxies_output_path
        self.probes_output_path = probes_output_path

        self.save_hyperparameters(ignore=["model", "proxies_output_path", "probes_output_path"])

        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)


    def mapd_settings(cls) -> Dict[str, Any]:
        return {
            "proxies_output_path": cls.proxies_output_path,
            "probes_output_path": cls.probes_output_path,
        }

    def forward(self, x):
        return self.model(x)

    def batch_loss(self, logits, y):
        return F.cross_entropy(logits, y, reduction="none")

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)
        loss = self.batch_loss(logits, y).mean()
        self.mapd_log(logits, y)

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=False
        )

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch

        logits = self(x)
        loss = self.batch_loss(logits, y).mean()
        self.mapd_log(logits, y)

        self.val_accuracy(logits, y)
        self.log(
            "val/accuracy",
            self.val_accuracy,
            on_step=True,
            on_epoch=True
        )
        self.log(
            "validation/loss",
            loss,
            on_step=False,
            on_epoch=True
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
