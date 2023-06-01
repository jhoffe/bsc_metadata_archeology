import os
import random

import click
import matplotlib.pyplot as plt

from src.visualization.utils.plot_utils import (
    load_probe_suite,
    plot_dicts,
    plot_styles, load_loss_by_epoch,
)


def first_learned_plot(
    name: str, probe_suite_path: str, loss_dataset_path: str, output_path: str
) -> None:
    """Plot the accuracy for each probe suite pr. epoch"""

    probe_suite = load_probe_suite(probe_suite_path=probe_suite_path)
    suite_indices = probe_suite.index_to_suite

    suite_names = {
        "typical": "Typical",
        "atypical": "Atypical",
        "random_outputs": "Random outputs",
        "random_inputs_outputs": "Random inputs and outputs",
        "corrupted": "Corrupted",
    }

    suite_to_indices = {}

    for suite_attr, suite_name in suite_names.items():
        indices = [idx for idx, suite in suite_indices.items() if suite == suite_attr]
        random.shuffle(indices)
        train_indices = indices[:250]
        val_indices = indices[250:]

        suite_to_indices[suite_name] = train_indices
        suite_to_indices[suite_name + " [Val]"] = val_indices

    first_learned, learned, suite_names = plot_dicts()

    epoch = 0
    while True:
        val_df = load_loss_by_epoch(loss_dataset_path, epoch, "val")
        if val_df.empty:
            break

        val_df.drop_duplicates(subset=["sample_index"], inplace=True)
        val_df["prediction"] = (val_df["y"] == val_df["y_hat"]).astype(int)

        # Loop over the different suites and count which has been learned
        for suite, indices in suite_to_indices.items():
            samples = val_df[(val_df["sample_index"].isin(indices))
                             & (val_df["prediction"] == 1)]
            learned[suite].update(samples["sample_index"].values)
            first_learned[suite].append(100*len(learned[suite]) / len(indices))

        train_df = load_loss_by_epoch(loss_dataset_path, epoch, "train")
        train_df["prediction"] = (train_df["y"] == train_df["y_hat"]).astype(int)
        samples = train_df[train_df["prediction"] == 1]
        learned["Train"].update(samples["sample_index"].values)

        first_learned["Train"].append(100*len(learned["Train"]) / len(train_df))
        epoch += 1

    suites = first_learned.keys()
    # Plot
    line_styles, marker_list, marker_colors, plot_titles = plot_styles()

    plt.figure(figsize=(10, 6))
    plt.tight_layout()
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
    plt.legend(loc="lower right", fontsize=12, fancybox=True, framealpha=0.4)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Fraction of samples learned (%)", fontsize=16)

    figure_path = os.path.join(output_path, name)

    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    plt.savefig(
        os.path.join(
            figure_path,
            f"{name}_first_learned_accuracy.png"
        ),
        bbox_inches="tight"
    )


def main(name, probe_suite_path, loss_dataset_path, output_filepath):
    first_learned_plot(name, probe_suite_path, loss_dataset_path, output_filepath)


@click.command()
@click.option("--name", type=str, required=True)
@click.argument(
    "probe_suite_path", type=click.Path(exists=True, dir_okay=False, file_okay=True)
)
@click.argument(
    "loss_dataset_path", type=click.Path(exists=True, dir_okay=True, file_okay=False)
)
@click.argument(
    "output_filepath", type=click.Path(exists=True, dir_okay=True, file_okay=False)
)
def main_click(name, probe_suite_path, loss_dataset_path, output_filepath):
    main(name, probe_suite_path, loss_dataset_path, output_filepath)


if __name__ == "__main__":
    main_click()
