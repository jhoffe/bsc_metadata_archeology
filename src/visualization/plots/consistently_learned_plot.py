import os
import random

import click
import matplotlib.pyplot as plt

from src.visualization.utils.plot_utils import (
    load_probe_suite,
    plot_dicts,
    plot_styles, load_loss_by_epoch,
)


def consistently_learned_plot(
        name: str, probe_suite_path: str, loss_dataset_path: str, output_path: str
) -> None:
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

    consistently_learned, learned, suite_names = plot_dicts()

    epochs = load_loss_by_epoch(loss_dataset_path, stage="val")["epoch"]
    max_epoch = (epochs).astype(int).max()

    train_df = load_loss_by_epoch(loss_dataset_path, epoch=max_epoch, stage="train")
    train_df["prediction"] = (train_df["y"] == train_df["y_hat"]).astype(int)
    learned["Train"] = set(train_df[train_df["prediction"] == 1]["sample_index"].values)

    val_df = load_loss_by_epoch(loss_dataset_path, epoch=max_epoch, stage="val")
    val_df["prediction"] = (val_df["y"] == val_df["y_hat"]).astype(int)
    for suite, indices in suite_to_indices.items():
        samples = val_df[(val_df["sample_index"].isin(indices))
                         & (val_df["prediction"] == 1)]
        learned[suite].update(samples["sample_index"].values)

    suite_size = len(probe_suite.index_to_suite) / len(suite_names)
    train_size = len(train_df)

    for epoch in reversed(range(max_epoch)):
        train_df = load_loss_by_epoch(loss_dataset_path, epoch=epoch, stage="train")
        train_df["prediction"] = (train_df["y"] == train_df["y_hat"]).astype(int)

        train_df = train_df[train_df["sample_index"].isin(learned["Train"])]
        learned["Train"] = set(
            train_df["sample_index"][train_df["prediction"] == 1].values
        )
        consistently_learned["Train"].insert(
            0,
            100 * len(learned["Train"]) / train_size
        )

        val_df = load_loss_by_epoch(loss_dataset_path, epoch=epoch, stage="val")
        val_df["prediction"] = (val_df["y"] == val_df["y_hat"]).astype(int)

        for suite, indices in suite_to_indices.items():
            suite_group = val_df[val_df["sample_index"].isin(indices)]
            suite_group = suite_group[suite_group["sample_index"].isin(learned[suite])]
            learned[suite] = set(
                suite_group["sample_index"][suite_group["prediction"] == 1].values
            )
            consistently_learned[suite].insert(
                0,
                100 * len(learned[suite]) / suite_size
            )

    suites = consistently_learned.keys()

    line_styles, marker_list, marker_colors, plot_titles = plot_styles()

    plt.figure(figsize=(10, 6))
    for i, suite in enumerate(suites):
        plt.plot(
            consistently_learned[suite],
            label=suite,
            alpha=0.75,
            linewidth=0.5,
            linestyle=line_styles[i % len(line_styles)],
            marker=marker_list[i % len(marker_list)],
            markersize=3,
            color=marker_colors[i % len(marker_colors)],
        )
    plt.legend(loc="upper left", fontsize=12, fancybox=True, framealpha=0.4)
    plt.xlabel("Epoch", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel("Percent Samples Learned (%)", fontsize=16)

    figure_path = os.path.join(output_path, name)

    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    plt.savefig(
        os.path.join(
            figure_path,
            f"{name}_consistently_learned_accuracy.png"
        ),
        bbox_inches="tight"
    )


def main(name, probe_suite_path, loss_dataset_path, output_filepath):
    consistently_learned_plot(
        name, probe_suite_path, loss_dataset_path, output_filepath
    )


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
