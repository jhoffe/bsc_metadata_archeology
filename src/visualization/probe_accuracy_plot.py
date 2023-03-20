import os
import random

import click
import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.visualization.utils.plot_utils import (
    get_indices_from_probe_suite,
    get_loss_dataset,
)


def probe_accuracy_plot(df: pd.DataFrame, output_path: str, dataset_name: str) -> None:
    """Plot the accuracy for each probe suite pr. epoch"""

    probe_suite = torch.load(f"data/processed/{dataset_name}/train_probe_suite.pt")

    df["epoch"] = df["epoch"].astype(int)

    suite_names = {
        "typical": "Typical",
        "atypical": "Atypical",
        "random_outputs": "Random outputs",
        # "random_inputs_outputs": "Random inputs and outputs",
        "corrupted": "Corrupted",
    }

    for suite_attr, suite_name in suite_names.items():
        suite = getattr(probe_suite, suite_attr)
        indices = get_indices_from_probe_suite(suite)
        random.shuffle(indices)
        train_indices = indices[:250]
        val_indices = indices[250:]
        df.loc[df["sample_index"].isin(train_indices), "suite"] = suite_name
        df.loc[df["sample_index"].isin(val_indices), "suite"] = suite_name + " [Val]"

    df["suite"] = df["suite"].fillna("Train")
    suites = sorted(df["suite"].unique())

    df["prediction"] = df["y"] == df["y_hat"]

    val_df = df[df['stage'] == 'val']
    val_df = val_df.groupby(["epoch", "suite"]).agg({"prediction": "mean"})
    val_df.reset_index(inplace=True)
    val_df["prediction"] = val_df["prediction"] * 100

    train_df = df[df['stage'] == 'train']
    train_df = train_df.groupby(["epoch", "suite"]).agg({"prediction": "mean"})
    train_df.reset_index(inplace=True)
    train_df["prediction"] = train_df["prediction"] * 100

    suites = sorted(train_df["suite"].unique())

    # Plot
    line_styles = ["solid", "dashed", "dashdot", "dotted"]
    marker_list = ["o", "*", "X", "P", "p", "D", "v", "^", "h", "1", "2", "3", "4"]
    marker_colors = [
        "tab:gray",
        "tab:green",
        "tab:blue",
        "tab:purple",
        "tab:orange",
        "tab:red",
        "tab:pink",
        "tab:olive",
        "tab:brown",
        "tab:cyan",
    ]

    plot_titles = {
        "cifar10": "CIFAR-10",
        "cifar100": "CIFAR-100",
        "imagenet": "ImageNet",
    }
    plt.figure(figsize=(10, 6))
    plt.title(f"Probe Suite Accuracy for {plot_titles['cifar10']}")
    for i, suite in enumerate(suites):
        if "Train" in suite:
            plt.plot(
            train_df[train_df["suite"] == suite]["epoch"],
            train_df[train_df["suite"] == suite]["prediction"],
            label=suite,
            alpha=0.75,
            linewidth=0.5,
            linestyle=line_styles[i % len(line_styles)],
            marker=marker_list[i % len(marker_list)],
            markersize=3,
            color=marker_colors[i % len(marker_colors)],
        )
        else:
            plt.plot(
                val_df[val_df["suite"] == suite]["epoch"],
                val_df[val_df["suite"] == suite]["prediction"],
                label=suite,
                alpha=0.75,
                linewidth=0.5,
                linestyle=line_styles[i % len(line_styles)],
                marker=marker_list[i % len(marker_list)],
                markersize=3,
                color=marker_colors[i % len(marker_colors)],
            )
    plt.legend(loc="lower right", fontsize="small")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")

    figure_path = os.path.join(output_path, dataset_name)

    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    plt.savefig(os.path.join(figure_path, f"{dataset_name}_probe_suite_accuracy.png"))


def main(dataset_path, output_filepath, dataset_name):
    df = get_loss_dataset(dataset_path)
    probe_accuracy_plot(df, output_filepath, dataset_name)


@click.command()
@click.argument(
    "dataset_path", type=click.Path(exists=True, dir_okay=True, file_okay=False)
)
@click.argument(
    "output_filepath", type=click.Path(exists=True, dir_okay=True, file_okay=False)
)
@click.argument("dataset_name", type=str)
def main_click(dataset_path, output_filepath, dataset_name):
    main(dataset_path, output_filepath, dataset_name)


if __name__ == "__main__":
    main_click()
