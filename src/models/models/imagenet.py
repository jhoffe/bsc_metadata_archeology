import torch
from torchvision.models.resnet import resnet50
from torch.nn import Module, functional as F
from torch.optim import SGD
import pytorch_lightning as pl

imagenet_model: Module = resnet50(weights=None)


class ImageNet(pl.LightningModule):
    def __init__(self, model: Module = imagenet_model):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        y = F.one_hot(y, num_classes=1000).to(torch.float32)

        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y = F.one_hot(y, num_classes=1000).to(torch.float32)

        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("validation/loss", loss)

    def test_step(self, batch, batch_idx):
        x, y = batch

        y = F.one_hot(y, num_classes=1000).to(torch.float32)

        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("test/loss", loss)

    def configure_optimizers(self):
        return SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)