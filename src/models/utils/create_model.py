import torch
import torchvision
from torch import nn

def create_model(dataset: str, batch_size: int) -> nn.Module:
    """Creates a ResNet50 model"""
    num_classes = 10 if dataset == "cifar10" else 100
    model = torchvision.models.resnet50(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(
        3, batch_size, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    )
    model.maxpool = nn.Identity()
    return model