import click
import pyarrow.parquet as pq
from matplotlib import pyplot as plt


@click.command()
@click.argument(
    "surfaced_examples_path",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
)
def main(surfaced_examples_path):
    surfaced_examples = pq.ParquetDataset(
        surfaced_examples_path, use_legacy_dataset=False
    )
    df = surfaced_examples.read().to_pandas()

    plt.figure(figsize=(10, 10))

    plt.pie(df["label_name"].value_counts(), labels=None)
    plt.title("Probe suite prediction distribution")

    labels = [
        f"{label} ({count}/{(count / len(df)) * 100:0.2f}%)"
        for label, count in zip(
            df["label_name"].value_counts().index,
            df["label_name"].value_counts().values,
        )
    ]

    plt.legend(labels)
    plt.show()


if __name__ == "__main__":
    main()
