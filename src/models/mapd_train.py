import gc
import logging
import os
from time import gmtime, strftime

import click
import lightning as L
import torch
from dotenv import find_dotenv, load_dotenv
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from mapd.classifiers.make_mapd_classifier import make_mapd_classifier
from mapd.classifiers.make_predictions import make_predictions
from mapd.probes.make_probe_suites import make_probe_suites
from mapd.utils.make_dataloaders import make_dataloaders
from mapd.utils.wrap_dataset import wrap_dataset
from mapd.visualization.surface_predictions import display_surface_predictions
from mapd.visualization.visualization_tool import MAPDVisualizationTool
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from src.data.datasets import ImageNetTrainingDataset, ImagenetValidationDataset
from src.data.utils.idx_to_label_names import get_idx_to_label_names_imagenet
from src.models.models import ResNet50MAPD


def get_datasets():
    imagenet_train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    imagenet_val_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    train_dataset = ImageNetTrainingDataset(
        "data/raw/imagenet", transform=imagenet_train_transform
    )
    val_dataset = ImagenetValidationDataset(
        "data/raw/imagenet",
        class_to_idx=train_dataset.class_to_idx,
        transform=imagenet_val_transform,
    )

    return train_dataset, val_dataset


def get_dataloaders(train_dataset, val_dataset, batch_size, num_workers, prefetch_factor):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
    )

    return train_dataloader, val_dataloader


def run_proxies(train_dataloader, val_dataloader, epochs: int, should_compile: bool, wandb_logger: WandbLogger = None):
    logger = logging.getLogger(__name__)

    torch.set_float32_matmul_precision("high")
    logger.info("loading datasets")

    proxy_module = ResNet50MAPD(
        max_epochs=epochs, should_compile=should_compile
    ).as_proxies()
    proxy_trainer = L.Trainer(
        accelerator="gpu",
        max_epochs=epochs,
        logger=[wandb_logger] if wandb_logger else None,
        precision="16-mixed",
        callbacks=[LearningRateMonitor()],
        limit_train_batches=5,
        limit_val_batches=5,
    )

    logger.info("training proxies")
    proxy_trainer.fit(
        proxy_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    del proxy_module
    gc.collect()


def run_probes(train_dataloader, validation_dataloaders, epochs: int, should_compile: bool,
               train_suite: bool, wandb_logger: WandbLogger = None):
    TIME_STR = strftime("%Y%m%d_%H%M", gmtime())
    run_name = (
            "imagenet-probes-e2e" + ("-with-train" if train_suite else "") + "-" + TIME_STR
    )
    logger = logging.getLogger(__name__)

    probes_module = ResNet50MAPD(
        max_epochs=epochs, should_compile=should_compile
    ).as_probes()
    probes_trainer = L.Trainer(
        accelerator="gpu",
        max_epochs=epochs,
        logger=[wandb_logger] if wandb_logger is not None else None,
        precision="16-mixed",
        callbacks=[LearningRateMonitor()],
        limit_train_batches=5,
        limit_val_batches=5,
    )

    logger.info("training probes")
    probes_trainer.fit(
        probes_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=validation_dataloaders,
    )

    del probes_module
    gc.collect()


def plot(train_probes_dataset, train_suite):
    logger = logging.getLogger(__name__)

    output_path = "reports/figures/mapd_e2e"
    os.makedirs(output_path, exist_ok=True)

    plot_tool = MAPDVisualizationTool("models/probes_e2e", train_probes_dataset)
    plot_tool.loss_curve_plot(
        show=False, save=True, save_path=os.path.join(output_path, "loss_curve.png")
    )
    plot_tool.probe_accuracy_plot(
        show=False, save=True, save_path=os.path.join(output_path, "probe_accuracy.png")
    )
    plot_tool.first_learned_plot(
        show=False, save=True, save_path=os.path.join(output_path, "first_learned.png")
    )
    plot_tool.consistently_learned_plot(
        show=False,
        save=True,
        save_path=os.path.join(output_path, "consistently_learned.png"),
    )

    logger.info("training mapd classifier")
    mapd_clf, label_encoder = make_mapd_classifier("models/probes_e2e", train_probes_dataset)

    logger.info("saving mapd classifier")
    # Now we can surface some examples
    probe_predictions = make_predictions("models/probes_e2e", mapd_clf, label_encoder, n_jobs=16)

    # Define the possible labels (for plotting)
    index_to_labels = get_idx_to_label_names_imagenet()

    train_dataset, _ = get_datasets()

    imagenet_val_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    train_dataset.transforms = imagenet_val_transform

    logger.info("saving surface predictions")
    fig = display_surface_predictions(
        probe_predictions,
        train_dataset,
        probe_suite="typical",
        labels=index_to_labels,
        ordered=True,
    )
    fig.savefig(os.path.join(output_path, "surface_typical.png"), dpi=300)

    fig = display_surface_predictions(
        probe_predictions,
        train_dataset,
        probe_suite="atypical",
        labels=index_to_labels,
        ordered=True,
    )
    fig.savefig(os.path.join(output_path, "surface_atypical.png"), dpi=300)

    fig = display_surface_predictions(
        probe_predictions,
        train_dataset,
        probe_suite="random_outputs",
        labels=index_to_labels,
        ordered=True,
    )
    fig.savefig(os.path.join(output_path, "surface_random_outputs.png"), dpi=300)
    fig = display_surface_predictions(
        probe_predictions,
        train_dataset,
        probe_suite="random_inputs_outputs",
        labels=index_to_labels,
        ordered=True,
    )
    fig.savefig(os.path.join(output_path, "surface_random_inputs_outputs.png"), dpi=300)

    if train_suite:
        fig = display_surface_predictions(
            probe_predictions,
            train_dataset,
            probe_suite="train",
            labels=index_to_labels,
            ordered=True,
        )
        fig.savefig(os.path.join(output_path, "surface_train"), dpi=300)


@click.command()
@click.option("--train-suite", default=False, is_flag=True)
@click.option("--compile", default=False, is_flag=True)
def main(train_suite, compile):
    NUM_WORKERS = 16
    BATCH_SIZE = 64
    PREFETCH_FACTOR = 2
    PROXY_EPOCHS = 3
    PROBE_EPOCHS = 3

    logger = logging.getLogger(__name__)

    train_dataset, val_dataset = get_datasets()
    idx_train_dataset = wrap_dataset(train_dataset)
    idx_val_dataset = wrap_dataset(val_dataset)

    TIME_STR = strftime("%Y%m%d_%H%M", gmtime())

    run_name_proxy = (
            "imagenet-proxy-e2e" + ("-with-train" if train_suite else "") + "-" + TIME_STR
    )
    wandb_logger = WandbLogger(name=run_name_proxy, project="bsc", log_model=False, save_dir="models/")

    proxy_train_dataloader, validation_dataloader = get_dataloaders(idx_train_dataset, idx_val_dataset, BATCH_SIZE,
                                                                    NUM_WORKERS, PREFETCH_FACTOR)
    #run_proxies(proxy_train_dataloader, validation_dataloader, PROXY_EPOCHS, compile, wandb_logger=wandb_logger)

    del proxy_train_dataloader, validation_dataloader
    gc.collect()

    logger.info("creating probes")
    train_probes_dataset = make_probe_suites(
        idx_train_dataset, proxy_calculator="models/proxies_e2e", label_count=1000, add_train_suite=train_suite
    )

    probe_train_dataloader, validation_dataloader = get_dataloaders(train_probes_dataset, idx_val_dataset, BATCH_SIZE,
                                                                    NUM_WORKERS, PREFETCH_FACTOR)

    mapd_validation_dataloaders = make_dataloaders(
        [validation_dataloader],
        train_probes_dataset,
        dataloader_kwargs={
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "prefetch_factor": PREFETCH_FACTOR,
        },
    )

    run_probes(
        probe_train_dataloader, mapd_validation_dataloaders, PROBE_EPOCHS, compile, train_suite
    )

    logger.info("plotting")
    plot(train_probes_dataset, train_suite)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)

    main()
