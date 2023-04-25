import click
import matplotlib.pyplot as plt
from lightning import seed_everything
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from src.data.loss_dataset import LossDataset


@click.command()
@click.argument(
    "probe_suite_path", type=click.Path(exists=True, dir_okay=False, file_okay=True)
)
@click.argument(
    "loss_dataset_path", type=click.Path(exists=True, dir_okay=True, file_okay=False)
)
@click.option("--seed", type=int, default=42)
def main(probe_suite_path, loss_dataset_path, seed):
    seed_everything(seed)
    loss_dataset = LossDataset(loss_dataset_path)

    print("Loading loss dataset...")
    loss_dataset.load()
    print("Loading probe suite...")
    loss_dataset.load_probe_suite(probe_suite_path)

    print("Creating train matrix...")
    X, y = loss_dataset.to_sklearn_train_matrix()

    print("Standardizing...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print("TSNE...")
    tsne = TSNE(
        n_components=3, n_jobs=-1, random_state=seed, perplexity=100, init="pca"
    )
    X = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 10))

    suite_names = set(loss_dataset.probe_suite.index_to_suite.values())
    idx_to_label = {i: label for i, label in enumerate(suite_names)}

    for i, label in idx_to_label.items():
        ax.scatter(
            X[y == i, 0], X[y == i, 1], label=label.replace("_", " ").capitalize()
        )

    plt.legend()
    plt.savefig("reports/figures/tsne_decision_boundary_with_scaling.png")


if __name__ == "__main__":
    main()
