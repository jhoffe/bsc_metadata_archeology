import torch
import torchvision


def create_model(num_classes: int):
    """Creates a ResNet50 model"""
    model = torchvision.models.resnet50(weights=None, num_classes=num_classes)
    model.conv1 = torch.nn.Conv2d(
        3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    )
    model.maxpool = torch.nn.Identity()
    return torch.compile(model)
