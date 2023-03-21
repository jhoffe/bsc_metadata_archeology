import click

from src.visualization.first_learned_plot import main as first_learned_plot
from src.visualization.probe_accuracy_plot import main as probe_accuracy_plot
from src.visualization.consistently_learned_plot import main as consistently_learned_plot


# add click options here
@click.command()
@click.argument(
    "loss_dataset_path", type=click.Path(exists=True, dir_okay=True, file_okay=False)
)
@click.argument(
    "output_filepath", type=click.Path(exists=True, dir_okay=True, file_okay=False)
)
@click.argument("dataset_name", type=str)
def plots(loss_dataset_path, output_filepath, dataset_name):
    probe_accuracy_plot(loss_dataset_path, output_filepath, dataset_name)
    first_learned_plot(loss_dataset_path, output_filepath, dataset_name)
    consistently_learned_plot(loss_dataset_path, output_filepath, dataset_name)


if __name__ == "__main__":
    plots()
