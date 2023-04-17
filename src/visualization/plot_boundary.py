import click
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, is_classifier
from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data.loss_dataset import LossDataset


def train_metadata_model(
    classifier: BaseEstimator, X: np.array, y: np.array
) -> BaseEstimator:
    assert is_classifier(classifier)

    classifier.fit(X, y)

    return classifier


@click.command()
@click.argument(
    "loss_dataset_path", type=click.Path(exists=True, dir_okay=True, file_okay=False)
)
@click.argument(
    "probe_suite_path", type=click.Path(exists=True, dir_okay=False, file_okay=True)
)
def main(loss_dataset_path, probe_suite_path):
    loss_dataset = LossDataset(loss_dataset_path)

    print("Loading loss dataset...")
    loss_dataset.load()
    print("Loading probe suite...")
    loss_dataset.load_probe_suite(probe_suite_path)

    print("Creating train matrix...")
    X_train, y_train = loss_dataset.to_sklearn_train_matrix()

    print("Standardizing...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    print("PCA...")
    pca = PCA(n_components=2)
    X_train = pca.fit_transform(X_train)

    print("Explained variance ratio: ", pca.explained_variance_ratio_)

    # classifier = XGBClassifier(n_estimators=100)
    classifier = SVC()
    print("Training model...")
    classifier = train_metadata_model(classifier, X_train, y_train)

    print("Plotting decision boundary...")
    disp = DecisionBoundaryDisplay.from_estimator(classifier, X_train, alpha=0.5)

    suite_names = set(loss_dataset.probe_suite.index_to_suite.values())
    print(suite_names)
    idx_to_label = {i: label for i, label in enumerate(suite_names)}

    for i, label in idx_to_label.items():
        disp.ax_.scatter(
            X_train[y_train == i, 0], X_train[y_train == i, 1], label=label
        )
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
