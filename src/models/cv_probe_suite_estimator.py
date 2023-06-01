import os
import time

import click
import numpy as np
import pandas as pd
from lightning import seed_everything
from sklearn.base import BaseEstimator, is_classifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRFClassifier

from src.data.loss_dataset import LossDataset


def cv_metadata_model(classifier: BaseEstimator, X: np.array, y: np.array) -> np.array:
    assert is_classifier(classifier)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42, stratify=y, shuffle=True
    )

    folds = cross_validate(classifier, X_train, y_train, cv=10, return_estimator=True)

    best_fold = np.argmax(folds["test_score"])

    best_est = folds["estimator"][best_fold]

    gen_score = np.mean(folds["test_score"])
    test_score = best_est.score(X_test, y_test)

    return gen_score, test_score


@click.command()
@click.argument(
    "loss_dataset_path", type=click.Path(exists=True, dir_okay=True, file_okay=False)
)
@click.argument(
    "probe_suite_path", type=click.Path(exists=True, dir_okay=False, file_okay=True)
)
@click.option(
    "-r",
    "--results-dir",
    type=click.Path(dir_okay=True, file_okay=False),
    default="results/tables/",
)
@click.option("--name", type=str, required=True)
@click.option("--seed", type=int, default=42)
def main(loss_dataset_path, probe_suite_path, results_dir, seed, name):
    seed_everything(seed)

    loss_dataset = LossDataset(loss_dataset_path)
    loss_dataset.load()
    loss_dataset.load_probe_suite(probe_suite_path)
    X, y = loss_dataset.to_sklearn_train_matrix()

    # Run all classifiers
    classifiers = {
        "KNN": KNeighborsClassifier(20),
        "SVM": SVC(),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=3000),
        "Random Forest": RandomForestClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "MLP": MLPClassifier((50, 25, 10), max_iter=1000),
        "XGBoost": XGBClassifier(),
        "XGBoost RF": XGBRFClassifier(n_estimators=100),
    }

    results_table = []

    for clf_name, classifier in classifiers.items():
        print(f"Running {clf_name}")
        start = time.time()
        test_score, gen_score = cv_metadata_model(classifier, X, y)
        end = time.time()
        total_time = end - start
        print(f"Gen score={np.mean(gen_score):.3f}")
        print(f"Test score={np.mean(test_score):.3f}")
        print(f"Time={total_time:.2f} seconds")

        results_table.append([clf_name, gen_score, test_score, total_time])

    results_table = pd.DataFrame(
        results_table, columns=["Classifier", "Gen Score", "Test Score", "Time"]
    )
    results_table.sort_values(by="Test Score", ascending=False, inplace=True)
    print(results_table)

    os.makedirs(results_dir, exist_ok=True)
    results_table.to_csv(
        os.path.join(results_dir, f"results_table_{name}.csv"), index=False
    )
    results_table.to_latex(
        os.path.join(results_dir, f"results_table_{name}.tex"),
        index=False,
        float_format="%.3f",
    )


if __name__ == "__main__":
    main()
