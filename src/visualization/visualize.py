import click

from src.visualization.plots.consistently_learned_plot import (
    main as consistently_learned_plot,
)
from src.visualization.plots.first_learned_plot import main as first_learned_plot
from src.visualization.plots.loss_curve_plot import main as loss_curve_plot
from src.visualization.plots.probe_accuracy_plot import main as probe_accuracy_plot
from src.visualization.plots.probe_accuracy_val import main as probe_accuracy_val


# add click options here
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
def plots(name, probe_suite_path, loss_dataset_path, output_filepath, rio):
    probe_accuracy_plot(name, probe_suite_path, loss_dataset_path, output_filepath)
    probe_accuracy_val(name, probe_suite_path, loss_dataset_path, output_filepath)
    first_learned_plot(name, probe_suite_path, loss_dataset_path, output_filepath)
    consistently_learned_plot(
        name, probe_suite_path, loss_dataset_path, output_filepath
    )
    loss_curve_plot(name, probe_suite_path, loss_dataset_path, output_filepath, False)
    loss_curve_plot(name, probe_suite_path, loss_dataset_path, output_filepath, True)


if __name__ == "__main__":
    plots()
