import logging
import os
from time import gmtime, strftime

import click
import lightning as L
import torch
from dotenv import find_dotenv, load_dotenv
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from mapd.utils.make_dataloaders import make_dataloaders
from mapd.utils.wrap_dataset import wrap_dataset
from mapd.visualization.surface_predictions import display_surface_predictions
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from src.data.datasets import ImageNetTrainingDataset, ImagenetValidationDataset
from src.data.utils.idx_to_label_names import get_idx_to_label_names_imagenet
from src.models.models import ResNet50MAPD


@click.command()
@click.option("--train-suite", default=False, is_flag=True)
@click.option("--compile", default=False, is_flag=True)
def main(train_suite, compile):
    NUM_WORKERS = 16
    BATCH_SIZE = 256
    PREFETCH_FACTOR = 4
    PROXY_EPOCHS = 100
    PROBE_EPOCHS = 100

    TIME_STR = strftime("%Y%m%d_%H%M", gmtime())

    run_name = (
        "imagenet-proxy-e2e" + ("-with-train" if train_suite else "") + "-" + TIME_STR
    )

    wandb_logger = [
        WandbLogger(name=run_name, project="bsc", save_dir="models/", log_model=False)
    ]

    logger = logging.getLogger(__name__)

    torch.set_float32_matmul_precision("high")

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

    logger.info("loading datasets")
    train_dataset = ImageNetTrainingDataset(
        "data/raw/imagenet", transform=imagenet_train_transform
    )
    val_dataset = ImagenetValidationDataset(
        "data/raw/imagenet",
        class_to_idx=train_dataset.class_to_idx,
        transform=imagenet_val_transform,
    )

    idx_train_dataset = wrap_dataset(train_dataset)
    idx_val_dataset = wrap_dataset(val_dataset)

    proxy_train_dataloader = DataLoader(
        idx_train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
        prefetch_factor=PREFETCH_FACTOR,
        pin_memory=True,
    )
    validation_dataloader = DataLoader(
        idx_val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
        prefetch_factor=PREFETCH_FACTOR,
        pin_memory=True,
    )

    proxy_module = ResNet50MAPD(
        max_epochs=PROXY_EPOCHS, should_compile=compile
    ).as_proxies()
    proxy_trainer = L.Trainer(
        accelerator="gpu",
        max_epochs=PROXY_EPOCHS,
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

    logger.info("creating probes")
    train_probes_dataset = proxy_module.make_probe_suites(
        idx_train_dataset, num_labels=1000, add_train_suite=train_suite
    )

    probe_train_dataloader = DataLoader(
        train_probes_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
        prefetch_factor=PREFETCH_FACTOR,
        pin_memory=True,
    )

    mapd_validation_dataloaders = make_dataloaders(
        [validation_dataloader],
        train_probes_dataset,
        dataloader_kwargs={
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "prefetch_factor": PREFETCH_FACTOR,
        },
    )

    probes_module = ResNet50MAPD(
        max_epochs=PROBE_EPOCHS, should_compile=compile
    ).as_probes()
    probes_trainer = L.Trainer(
        accelerator="gpu",
        max_epochs=PROBE_EPOCHS,
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

    logger.info("saving results")

    output_path = os.path.join("reports/figures", run_name)

    os.makedirs(output_path, exist_ok=True)
    plot_tool = probes_module.visualiaztion_tool(train_probes_dataset)
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
    mapd_clf, label_encoder = probes_module.make_mapd_classifier(train_probes_dataset)

    logger.info("saving mapd classifier")
    # Now we can surface some examples
    probe_predictions = probes_module.mapd_predict(mapd_clf, label_encoder, n_jobs=16)

    # Define the possible labels (for plotting)
    index_to_labels = get_idx_to_label_names_imagenet()

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
