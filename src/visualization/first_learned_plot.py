import random

import pandas as pd
import torch

from src.visualization.utils.plot_utils import (
    get_indices_from_probe_suite,
    get_loss_dataset,
)


def first_learned_plot(df: pd.DataFrame, output_path: str, dataset_name: str) -> None:
    probe_suite = torch.load(f"data/processed/{dataset_name}/train_probe_suite.pt")

    df["epoch"] = df["epoch"].astype(int)

    suite_names = {
        "typical": "Typical",
        "atypical": "Atypical",
        "random_outputs": "Random outputs",
        # "random_inputs_outputs": "Random inputs and outputs",
        "corrupted": "Corrupted",
    }

    for suite_attr, suite_name in suite_names.items():
        suite = getattr(probe_suite, suite_attr)
        indices = get_indices_from_probe_suite(suite)
        random.shuffle(indices)
        train_indices = indices[:250]
        val_indices = indices[250:]
        df.loc[df["sample_index"].isin(train_indices), "suite"] = suite_name
        df.loc[df["sample_index"].isin(val_indices), "suite"] = suite_name + " [Val]"

    df["suite"] = df["suite"].fillna("Train")
    # suites = sorted(df["suite"].unique())


def main():
    df = get_loss_dataset("data/processed/cifar10/losses.csv")
    first_learned_plot(df, "data/processed/cifar10/losses.csv", "cifar10")

if __name__ == "__main__":
    df = get_loss_dataset("models/losses/cifar10-20230320_1038")
    

