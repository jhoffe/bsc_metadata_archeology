from omegaconf import DictConfig

from src.models.models import CIFARResNet50, ImageNetResNet50


def create_module(params: DictConfig):
    MODULES = {"imagenet-resnet50": ImageNetResNet50, "cifar-resnet50": CIFARResNet50}

    model_params = params.model
    module_name = model_params["name"]

    if module_name not in MODULES.keys():
        raise ValueError(f"model '{module_name}' not in modules")

    return MODULES[module_name](
        **{key: value for key, value in model_params.items() if key != "name"}
    )
