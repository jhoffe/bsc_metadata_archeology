import click
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.base import BaseEstimator, is_classifier
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

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
@click.argument("output_path", type=click.Path(dir_okay=True, file_okay=False))
def main(loss_dataset_path, probe_suite_path, output_path):
    loss_dataset = LossDataset(loss_dataset_path)

    print("Loading loss dataset...")
    loss_dataset.load()
    print("Loading probe suite...")
    loss_dataset.load_probe_suite(probe_suite_path)

    print("Creating train matrix...")
    X_train, y_train = loss_dataset.to_sklearn_train_matrix()
    print("Creating predict matrix...")
    (
        X_predict,
        sample_indices,
        index_to_label_name,
    ) = loss_dataset.to_sklearn_predict_matrix()

    rf_classifier = RandomForestClassifier()
    print("Training model...")
    rf_classifier = train_metadata_model(rf_classifier, X_train, y_train)

    chunk_size = 512
    chunk_count = int(np.ceil(len(X_predict) / chunk_size))

    print("Predicting...")
    for i in tqdm(range(chunk_count)):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(X_predict))

        X_chunk = X_predict[start:end]

        y_pred_chunk = rf_classifier.predict(X_chunk)
        y_prob_chunk = rf_classifier.predict_proba(X_chunk)

        table = pa.Table.from_arrays(
            arrays=[
                pa.array(sample_indices[start:end]),
                pa.array(y_pred_chunk),
                pa.array(y_prob_chunk),
                pa.array([index_to_label_name[idx] for idx in y_pred_chunk]),
            ],
            names=["sample_index", "label", "probs", "label_name"],
        )

        pq.write_to_dataset(table=table, root_path=output_path)


if __name__ == "__main__":
    main()
