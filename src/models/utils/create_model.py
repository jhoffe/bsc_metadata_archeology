import torch
import torchvision
from src.models.utils.audio_model import M5


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

    return torch.compile(model, disable=not should_compile, mode="max-autotune")


def create_vit_model(
    num_classes: int, version: str = "vit_b_16", should_compile: bool = True
):
    """Create a ViT model.

    Args:
        num_classes: Number of classes in the dataset.
        image_size: Size of the input images.
        version: ViT version.
        should_compile: Whether to compile the model.

    Returns:
        A ViT model.
    """
    if version == "vit_b_16":
        model = torchvision.models.vit_b_16(weights=None, num_classes=num_classes)
    elif version == "vit_b_32":
        model = torchvision.models.vit_b_32(weights=None, num_classes=num_classes)
    elif version == "vit_l_16":
        model = torchvision.models.vit_l_16(weights=None, num_classes=num_classes)
    elif version == "vit_l_32":
        model = torchvision.models.vit_l_32(weights=None, num_classes=num_classes)
    elif version == "vit_h_14":
        model = torchvision.models.vit_h_14(weights=None, num_classes=num_classes)
    else:
        raise ValueError(f"ViT version '{version}' not supported")

    return torch.compile(model, disable=not should_compile, mode="max-autotune")


def create_m5_model(
        n_input: int = 1,
        n_output: int = 35,
        stride: int = 16,
        n_channel: int = 32,
        should_compile: bool = True
):
    model = M5(n_input, n_output, stride, n_channel)
    return torch.compile(model, disable=not should_compile, mode="max-autotune")
