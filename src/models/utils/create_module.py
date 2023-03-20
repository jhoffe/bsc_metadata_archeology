from omegaconf import DictConfig

from src.models.models import ResNet50
from src.models.models.vit import ViT


def create_module(params: DictConfig):
    MODULES = {"resnet50": ResNet50, "vit": ViT}

    model_params = params.model
    module_name = model_params["name"]

    if module_name not in MODULES.keys():
        raise ValueError(f"model '{module_name}' not in modules")

    return MODULES[module_name](
        **{key: value for key, value in model_params.items() if key != "name"}
    )
