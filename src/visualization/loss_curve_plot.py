import os
import random
import warnings

import click
import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.visualization.utils.plot_utils import (
    get_indices_from_probe_suite,
    get_loss_dataset,
    plot_styles,
)

warnings.filterwarnings("ignore")


def loss_curve_plot(
    df: pd.DataFrame, output_path: str, dataset_name: str, rio: bool = False
) -> None:
    """Plot the accuracy for each probe suite pr. epoch"""

    probe_suite = torch.load(f"data/processed/{dataset_name}/train_probe_suite.pt")

    df.drop_duplicates(subset=["epoch", "sample_index", "stage"], inplace=True)

    df["epoch"] = df["epoch"].astype(int)

    max_epoch = df["epoch"].max() + 1
    num_suite_samples = len(probe_suite.combined)
    num_train_samples = len(probe_suite)

    assert df["stage"][df["stage"] == "val"].count() == max_epoch * num_suite_samples
    assert df["stage"][df["stage"] == "train"].count() == max_epoch * num_train_samples

    val_df = df[df["stage"] == "val"]

    suite_names = {
        "typical": "Typical",
        "atypical": "Atypical",
        "random_outputs": "Random Outputs",
        "corrupted": "Corrupted",
    }

    if rio:
        suite_names["random_inputs_outputs"] = "Random Inputs and Outputs"

    # sort by epoch
    val_df.sort_values(by=["epoch"], inplace=True)

    _, _, marker_colors, plot_titles = plot_styles()

    plt.figure(figsize=(10, 6))
    plt.title(f"Loss curves for {plot_titles[dataset_name]}") if not rio else plt.title(
        f"Loss curves for {plot_titles[dataset_name]} w. RIO"
    )
    i = 0

    for suite_attr, suite_name in suite_names.items():
        suite = getattr(probe_suite, suite_attr)
        indices = get_indices_from_probe_suite(suite)
        random.shuffle(indices)

        rand_indices = indices[250:]

        val_df.loc[df["sample_index"].isin(rand_indices), "suite"] = suite_name
        for idx in rand_indices:
            plt.plot(
                val_df["epoch"][val_df["sample_index"] == idx],
                val_df["loss"][val_df["sample_index"] == idx],
                alpha=0.25,
                linewidth=0.1,
                color=marker_colors[i],
                zorder=1,
            )

        # plot aggregated loss for each suite over all samples
        plt.plot(
            val_df.loc[val_df["suite"] == suite_name].groupby(["epoch"]).mean().index,
            val_df.loc[val_df["suite"] == suite_name].groupby(["epoch"]).mean()["loss"],
            label=suite_name,
            linewidth=1,
            color=marker_colors[i],
            zorder=2,
        )

        i += 1

    plt.ylim(0, 10) if dataset_name == "cifar10" else plt.ylim(0, 14)
    plt.xlim(0, 120) if dataset_name == "cifar10" else plt.xlim(0, max_epoch)

    plt.legend(loc="upper right", fontsize="small")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    if rio:
        plt.savefig(
            os.path.join(
                output_path, dataset_name, f"{dataset_name}_loss_curve_rio.png"
            )
        )
    else:
        plt.savefig(
            os.path.join(output_path, dataset_name, f"{dataset_name}_loss_curve.png")
        )


def main(loss_dataset_path, output_filepath, dataset_name, rio=False):
    df = get_loss_dataset(loss_dataset_path)
    loss_curve_plot(df, output_filepath, dataset_name, rio=rio)


@click.command()
@click.argument(
    "loss_dataset_path", type=click.Path(exists=True, dir_okay=True, file_okay=False)
)
@click.argument(
    "output_filepath", type=click.Path(exists=True, dir_okay=True, file_okay=False)
)
@click.argument("dataset_name", type=str)
@click.option("--rio", is_flag=True)
def main_click(loss_dataset_path, output_filepath, dataset_name, rio):
    main(loss_dataset_path, output_filepath, dataset_name, rio=rio)


if __name__ == "__main__":
    main_click()
