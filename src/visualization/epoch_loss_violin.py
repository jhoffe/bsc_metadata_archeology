import os
import random

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from src.data.loss_dataset import LossDataset


def get_indices_from_probe_suite(suite: list) -> list[int]:
    return [idx for _, idx in suite]


def plot_epoch_loss_violin(
    epoch_index: int, df: pd.DataFrame, output_path: str, dataset_name: str
) -> None:
    """Plot a violin plot of the losses for each probe suite at a given epoch."""

    figure_path = os.path.join(output_path, dataset_name)

    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    epoch = df.groupby("epoch").get_group(epoch_index)

    probe_suite = torch.load(f"data/processed/{dataset_name}/train_probe_suite.pt")

    suite_names = {
        "typical": "Typical",
        "atypical": "Atypical",
        "random_outputs": "Random outputs",
        "random_inputs_outputs": "Random inputs and outputs",
        "corrupted": "Corrupted",
    }

    suite_loss_values = []
    suite_loss_names = []
    suite_loss_is_val = []

    for suite_attr, suite_name in suite_names.items():
        suite = getattr(probe_suite, suite_attr)
        indices = get_indices_from_probe_suite(suite)

        samples = epoch[epoch["sample_index"].isin(indices)]

        loss_values = samples["loss"].tolist()

        random.shuffle(loss_values)
        suite_loss_values.extend(loss_values)

        suite_loss_names.extend([suite_name] * len(loss_values))
        suite_loss_is_val.extend(["Train"] * 250)
        suite_loss_is_val.extend(["Validation"] * 250)

    df = pd.DataFrame(
        {
            "Loss values": suite_loss_values,
            "Suite": suite_loss_names,
            "Is validation": suite_loss_is_val,
        }
    )

    fig, ax = plt.subplots()
    sns.violinplot(
        data=df,
        x="Suite",
        y="Loss values",
        ax=ax,
        cut=0,
        showextrema=False,
        showmeans=False,
        showmedians=True,
        palette=sns.color_palette("pastel"),
        linewidth=1,
        inner="point",
        scale="width",
        hue="Is validation",
    )
    ax.set_ylim(0, 14)
    ax.set_title(f"Losses for each probe suite at epoch {epoch_index + 1}")
    plt.xticks(rotation=90)
    fig.savefig(
        os.path.join(figure_path, f"epoch-{epoch_index + 1}_loss_violin.png"),
        bbox_inches="tight",
    )


def main(dataset_path, output_filepath, dataset_name, epoch_indices):
    dataset = LossDataset(dataset_path)
    dataset.load()

    df = dataset.df

    for epoch_idx in epoch_indices:
        plot_epoch_loss_violin(epoch_idx, df, output_filepath, dataset_name)


@click.command()
@click.argument(
    "dataset_path", type=click.Path(exists=True, dir_okay=True, file_okay=False)
)
@click.argument(
    "output_filepath", type=click.Path(exists=True, dir_okay=True, file_okay=False)
)
@click.argument("dataset_name", type=str)
@click.argument("epoch_indices", nargs=-1, type=int)
def main_click(dataset_path, output_filepath, dataset_name, epoch_indices):
    main(dataset_path, output_filepath, dataset_name, epoch_indices)


if __name__ == "__main__":
    main_click()
