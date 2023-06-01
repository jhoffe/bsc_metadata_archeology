import os

import click
import numpy as np
import pyarrow.parquet as pq
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import transforms
from tqdm import tqdm

from src.data import get_idx_to_label_names


@click.command()
@click.argument(
    "surfaced_examples_path",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
)
@click.argument(
    "probe_suite_path", type=click.Path(exists=True, dir_okay=False, file_okay=True)
)
@click.argument("output_path", type=click.Path(dir_okay=True, file_okay=False))
@click.argument("classes", type=int, nargs=-1, required=True)
@click.option("--name", type=str, required=True)
def main(surfaced_examples_path, probe_suite_path, output_path, classes, name):
    surfaced_examples = pq.read_table(surfaced_examples_path)
    df = surfaced_examples.to_pandas()

    assert df["sample_index"].is_unique

    print("Loading probe suite...")
    probe_suite = torch.load(probe_suite_path)

    if name == "cifar10" or name == "cifar100":
        probe_suite.dataset.transform = transforms.Compose([transforms.ToTensor()])
    else:
        probe_suite.dataset.transform = transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
        )

    idx_to_label = get_idx_to_label_names(name)

    NROWS, NCOLUMNS = 4, 4

    suites = ["typical", "atypical", "corrupted", "random_outputs"]

    for cls in tqdm(classes, desc="Generating plots for classes"):
        label = idx_to_label[cls]
        shortened_label = label.split(",")[0] if "," in label else label
        save_dir = os.path.join(
            output_path, str(cls) + "-" + shortened_label.replace(" ", "_")
        )
        os.makedirs(save_dir, exist_ok=True)

        fig, axs = plt.subplots(NROWS, NCOLUMNS, figsize=(20, 20))
        # fig.tight_layout()
        fig.suptitle(f"Class: {label}", fontsize=32)

        # Plot 4 samples for each suite in columns
        # such that each column contains 4 pictures from the same class
        # and each column is a different suite
        for i, suite in enumerate(suites):
            suite_sample_indices = df[
                (df["label_name"] == suite) & (df["original_class"] == cls)
            ].sort_values("probs")["sample_index"]

            if len(suite_sample_indices) == 0 or len(suite_sample_indices) < NROWS:
                continue

            # sample_indices = suite_sample_indices.sample(
            #     n=NROWS * NCOLUMNS, random_state=123
            # ).values

            sample_indices = suite_sample_indices.values[:NROWS]
            # sample_indices = np.random.choice(
            #    sample_indices, size=NROWS, replace=False
            # )

            samples = [probe_suite[index] for index in sample_indices]

            for j, ((sample, _), _) in enumerate(samples):
                axs[j, i].imshow(sample.permute(1, 2, 0).numpy())
                axs[j, i].axis("off")

            # Add a suite title
            axs[0, i].set_title(suite, fontsize=32)

        # Reduce distance between images
        fig.subplots_adjust(top=0.94, hspace=0.1, wspace=0.1)
        fig.savefig(
            os.path.join(save_dir, "all.png"),
            bbox_inches="tight",
        )
        plt.close(fig)

        for suite in suites:
            suite_sample_indices = df[
                (df["label_name"] == suite) & (df["original_class"] == cls)
            ].sort_values("probs")["sample_index"]

            if (
                len(suite_sample_indices) == 0
                or len(suite_sample_indices) < NROWS * NCOLUMNS
            ):
                continue

            # sample_indices = suite_sample_indices.sample(
            #     n=NROWS * NCOLUMNS, random_state=123
            # ).values

            sample_indices = suite_sample_indices.values[: NROWS * NCOLUMNS * 4]
            sample_indices = np.random.choice(
                sample_indices, size=NROWS * NCOLUMNS, replace=False
            )

            samples = [probe_suite[index] for index in sample_indices]

            fig, axs = plt.subplots(NROWS, NCOLUMNS, figsize=(20, 20))
            fig.tight_layout()

            for i, (((sample, y), sample_idx), ax) in enumerate(
                zip(samples, axs.flatten())
            ):
                ax.imshow(sample.permute(1, 2, 0))

                ax.set_title(f"{shortened_label} (IDX: {sample_idx})")

            fig.suptitle(f"Predicted suite: {suite}", y=0.98, fontsize=20)
            fig.subplots_adjust(top=0.94, hspace=0.3)

            fig.savefig(
                os.path.join(save_dir, f"probe_suite_{suite}.png"),
                bbox_inches="tight",
            )
            plt.close(fig)


if __name__ == "__main__":
    main()
