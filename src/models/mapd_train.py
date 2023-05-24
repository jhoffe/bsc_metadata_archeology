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


def run_proxies(num_workers: int, batch_size: int, prefetch_factor: int, epochs: int, train_suite: bool,
                should_compile: bool):
    TIME_STR = strftime("%Y%m%d_%H%M", gmtime())

    run_name = (
            "imagenet-proxy-e2e" + ("-with-train" if train_suite else "") + "-" + TIME_STR
    )

    wandb_logger = [WandbLogger(name=run_name, project="bsc", save_dir="models/", log_model=False)]

    logger = logging.getLogger(__name__)

    torch.set_float32_matmul_precision("high")
    logger.info("loading datasets")
    train_dataset, val_dataset = get_datasets()
    idx_train_dataset = wrap_dataset(train_dataset)
    idx_val_dataset = wrap_dataset(val_dataset)

    proxy_train_dataloader = DataLoader(
        idx_train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
    )
    validation_dataloader = DataLoader(
        idx_val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
    )

    proxy_module = ResNet50MAPD(
        max_epochs=epochs, should_compile=should_compile
    ).as_proxies()
    proxy_trainer = L.Trainer(
        accelerator="gpu",
        max_epochs=epochs,
        logger=wandb_logger,
        precision="16-mixed",
        callbacks=[LearningRateMonitor()],
    )

    logger.info("training proxies")
    proxy_trainer.fit(
        proxy_module,
        train_dataloaders=proxy_train_dataloader,
        val_dataloaders=validation_dataloader,
    )

    del proxy_module
    gc.collect()


def run_probes(num_workers: int, batch_size: int, prefetch_factor: int, epochs: int, should_compile: bool,
               train_suite: bool):
    TIME_STR = strftime("%Y%m%d_%H%M", gmtime())
    run_name = (
            "imagenet-probes-e2e" + ("-with-train" if train_suite else "") + "-" + TIME_STR
    )
    wandb_logger = [WandbLogger(name=run_name, project="bsc", save_dir="models/", log_model=False)]
    logger = logging.getLogger(__name__)

    train_dataset, val_dataset = get_datasets()
    idx_val_dataset = wrap_dataset(val_dataset)
    idx_train_dataset = wrap_dataset(train_dataset)

    validation_dataloader = DataLoader(
        idx_val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
    )

    logger.info("creating probes")
    train_probes_dataset = make_probe_suites(
        idx_train_dataset, proxy_calculator="models/proxies_e2e", label_count=1000, add_train_suite=train_suite
    )

    probe_train_dataloader = DataLoader(
        train_probes_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
    )

    mapd_validation_dataloaders = make_dataloaders(
        [validation_dataloader],
        train_probes_dataset,
        dataloader_kwargs={
            "batch_size": batch_size,
            "num_workers": num_workers,
            "prefetch_factor": prefetch_factor,
        },
    )

    probes_module = ResNet50MAPD(
        max_epochs=epochs, should_compile=should_compile
    ).as_probes()
    probes_trainer = L.Trainer(
        accelerator="gpu",
        max_epochs=epochs,
        logger=wandb_logger,
        precision="16-mixed",
        callbacks=[LearningRateMonitor()],
    )

    logger.info("training probes")
    probes_trainer.fit(
        probes_module,
        train_dataloaders=probe_train_dataloader,
        val_dataloaders=mapd_validation_dataloaders,
    )

    del probes_module
    gc.collect()

    return train_probes_dataset


@click.command()
@click.option("--train-suite", default=False, is_flag=True)
@click.option("--compile", default=False, is_flag=True)
def main(train_suite, compile):
    NUM_WORKERS = 16
    BATCH_SIZE = 256
    PREFETCH_FACTOR = 4
    PROXY_EPOCHS = 100
    PROBE_EPOCHS = 100

    logger = logging.getLogger(__name__)

    output_path = "reports/figures/mapd_e2e"
    os.makedirs(output_path, exist_ok=True)

    #run_proxies(NUM_WORKERS, BATCH_SIZE, PREFETCH_FACTOR, PROXY_EPOCHS, train_suite, compile)

    logger.info("running probes")
    train_probes_dataset = run_probes(
        NUM_WORKERS, BATCH_SIZE, PREFETCH_FACTOR, PROBE_EPOCHS, compile, train_suite
    )

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


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)

    main()
