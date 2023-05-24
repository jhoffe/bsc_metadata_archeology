import logging

import click
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from src.data.loss_dataset import LossDataset


@click.command()
@click.argument(
    "loss_dataset_path", type=click.Path(exists=True, dir_okay=True, file_okay=False)
)
@click.argument(
    "probe_suite_path", type=click.Path(exists=True, dir_okay=False, file_okay=True)
)
def main(loss_dataset_path, probe_suite_path):
    logger = logging.getLogger(__name__)

    loss_dataset = LossDataset(loss_dataset_path)

    logger.info("Loading loss dataset...")
    loss_dataset.load()
    logger.info("Loading probe suite...")
    loss_dataset.load_probe_suite(probe_suite_path)

    logger.info("Creating train matrix...")
    X_train, y_train = loss_dataset.to_sklearn_train_matrix()

    logger.info("Standardizing...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    logger.info("TSNE...")
    tsne = TSNE(n_components=2, perplexity=50, n_iter=2500, random_state=42, n_jobs=-1)
    X_train = tsne.fit_transform(X_train)

    suite_names = set(loss_dataset.probe_suite.index_to_suite.values())
    logger.info(suite_names)
    idx_to_label = {i: label for i, label in enumerate(suite_names)}

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.tight_layout()

    for i, label in idx_to_label.items():
        ax.scatter(
            X_train[y_train == i, 0], X_train[y_train == i, 1], label=label
        )
    plt.legend()

    fig.savefig("reports/figures/tsne/tsne.png")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
