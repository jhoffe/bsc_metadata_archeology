import os
import random

import click
import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.visualization.utils.plot_utils import (
    get_indices_from_probe_suite,
    get_loss_dataset,
    plot_dicts,
    plot_styles,
)


def first_learned_plot(df: pd.DataFrame, output_path: str, dataset_name: str) -> None:
    """Plot the accuracy for each probe suite pr. epoch"""

    probe_suite = torch.load(f"data/processed/{dataset_name}/train_probe_suite.pt")

    df.drop_duplicates(subset=["epoch", "sample_index", "stage"], inplace=True)
    df["epoch"] = df["epoch"].astype(int)
    max_epoch = df["epoch"].max() + 1
    num_suite_samples = len(probe_suite.combined)
    num_train_samples = len(probe_suite)

    assert df["stage"][df["stage"] == "val"].count() == max_epoch * num_suite_samples
    assert df["stage"][df["stage"] == "train"].count() == max_epoch * num_train_samples

    suite_names = {
        "typical": "Typical",
        "atypical": "Atypical",
        "random_outputs": "Random outputs",
        "random_inputs_outputs": "Random inputs and outputs",
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

    df["prediction"] = df["y"] == df["y_hat"]

    val_df = df[df["stage"] == "val"]
    train_df = df[df["stage"] == "train"]

    first_learned, learned, suite_names = plot_dicts()

    temp_train = train_df[train_df["suite"] == "Train"]

    for epoch in range(max_epoch):
        epoch_train_group = temp_train.groupby(["epoch"]).get_group(epoch)
        learned["Train"].update(
            epoch_train_group["sample_index"][epoch_train_group["prediction"]].values
        )
        first_learned["Train"].append(len(learned["Train"]) / len(temp_train))
        epoch_val_group = val_df.groupby(["epoch"]).get_group(epoch)
        for suite in suite_names:
            suite_group = epoch_val_group.groupby(["suite"]).get_group(suite)
            learned[suite].update(
                suite_group["sample_index"][suite_group["prediction"]].values
            )
            first_learned[suite].append(len(learned[suite]) / len(suite_group))

    suites = first_learned.keys()

    # Plot
    line_styles, marker_list, marker_colors, plot_titles = plot_styles()

    plt.figure(figsize=(10, 6))
    plt.title(f"Percent First Learned for {plot_titles['cifar10']}")
    for i, suite in enumerate(suites):
        plt.plot(
            first_learned[suite],
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
    plt.ylabel("Fraction of samples learned (%)")

    figure_path = os.path.join(output_path, dataset_name)

    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    plt.savefig(os.path.join(figure_path, f"{dataset_name}_first_learned_accuracy.png"))


def main(loss_dataset_path, output_filepath, dataset_name):
    df = get_loss_dataset(loss_dataset_path)
    first_learned_plot(df, output_filepath, dataset_name)


@click.command()
@click.argument(
    "loss_dataset_path", type=click.Path(exists=True, dir_okay=True, file_okay=False)
)
@click.argument(
    "output_filepath", type=click.Path(exists=True, dir_okay=True, file_okay=False)
)
@click.argument("dataset_name", type=str)
def main_click(loss_dataset_path, output_filepath, dataset_name):
    main(loss_dataset_path, output_filepath, dataset_name)


if __name__ == "__main__":
    main_click()
