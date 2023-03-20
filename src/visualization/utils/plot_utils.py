# imports
import pandas as pd
from src.data.loss_dataset import LossDataset


def get_loss_dataset(dataset_path: str) -> pd.DataFrame:
    dataset = LossDataset(dataset_path)
    dataset.load()

    df = dataset.df
    return df


def get_indices_from_probe_suite(suite: list) -> list[int]:
    return [idx for _, idx in suite]
