import click

from src.visualization.first_learned_plot import main as first_learned_plot
from src.visualization.probe_accuracy_plot import main as probe_accuracy_plot


# add click options here
@click.command()
@click.argument(
    "dataset_path", type=click.Path(exists=True, dir_okay=True, file_okay=False)
)
@click.argument(
    "output_filepath", type=click.Path(exists=True, dir_okay=True, file_okay=False)
)
@click.argument("dataset_name", type=str)
def plots(dataset_path, output_filepath, dataset_name):
    probe_accuracy_plot(dataset_path, output_filepath, dataset_name)
    first_learned_plot(dataset_path, output_filepath, dataset_name)


if __name__ == "__main__":
    plots()
