import os
import random

import click
import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.data.loss_dataset import LossDataset


def get_loss_dataset(dataset_path: str) -> pd.DataFrame:
    dataset = LossDataset(dataset_path)
    dataset.load()

    df = dataset.df
    return df


def get_indices_from_probe_suite(suite: list) -> list[int]:
    return [idx for _, idx in suite]



dataset_name = "cifar10"

probe_suite = torch.load(f"data/processed/{dataset_name}/train_probe_suite.pt")
df = get_loss_dataset("models/losses/cifar10-1x-gpua100-20230311_1752")
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
suites = sorted(df["suite"].unique())
df["prediction"] = df["y"] == df["y_hat"]
final = df.groupby(["epoch", "suite"]).agg({"prediction": "mean"})
final.reset_index(inplace=True)
final["prediction"] = final["prediction"] * 100
