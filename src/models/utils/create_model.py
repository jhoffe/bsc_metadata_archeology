import torch
import torchvision


def create_resnet50_model(
    num_classes: int, resize_conv1: bool = False, should_compile: bool = True
):
    """Create a ResNet50 model.

    Args:
        num_classes: Number of classes in the dataset.
        resize_conv1: Whether to resize the first convolutional layer.
        should_compile: Whether to compile the model.

    Returns:
        A ResNet50 model.
    """
    model = torchvision.models.resnet50(weights=None, num_classes=num_classes)

    if resize_conv1:
        model.conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        model.maxpool = torch.nn.Identity()

    return torch.compile(model, disable=not should_compile)


def create_vit_model(num_classes: int, should_compile: bool = True):
    """Create a ViT model.

    Args:
        num_classes: Number of classes in the dataset.
        should_compile: Whether to compile the model.

    Returns:
        A ViT model.
    """
    model = torchvision.models.vit_b_16(weights=None, num_classes=num_classes)

    return torch.compile(model, disable=not should_compile)
