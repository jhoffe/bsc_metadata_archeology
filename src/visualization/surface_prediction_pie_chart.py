import click
import pandas as pd
from matplotlib import pyplot as plt


@click.command()
@click.argument(
    "surfaced_examples_path",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
)
def main(surfaced_examples_path):
    df = pd.read_parquet(surfaced_examples_path)

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
