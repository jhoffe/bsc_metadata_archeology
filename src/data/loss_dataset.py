from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from src.data.make_probe_suites import ProbeSuiteGenerator


class LossDataset:
    dataset_path: str
    dataset: pq.ParquetDataset
    df: Optional[pd.DataFrame]
    probe_suite: Optional[ProbeSuiteGenerator]

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.dataset = pq.ParquetDataset(dataset_path)

    def load(self) -> "LossDataset":
        self.df = self.dataset.read().to_pandas()

        return self

    def load_probe_suite(self, probe_suite_path: str) -> "LossDataset":
        self.probe_suite = torch.load(probe_suite_path)

        return self

    def to_sklearn_train_matrix(self) -> tuple[np.array, np.array]:
        assert self.df is not None
        assert self.probe_suite is not None

        (
            index_to_probe_suite_type_dict,
            index_to_label_name,
        ) = self.probe_suite.index_to_probe_suite_type_dict()

        df = self.df[self.df["sample_index"].isin(self.probe_suite.used_indices)].copy()

        # Convert to int from categorical
        df["epoch"] = df["epoch"].astype(int)

        df.sort_values(["sample_index", "epoch"], inplace=True)
        sample_groups = df.groupby("sample_index", sort=False)

        losses = sample_groups["loss"].agg(list)
        losses_indices = losses.index

        X = np.asarray(losses.to_list())
        y = np.array([index_to_probe_suite_type_dict[idx] for idx in losses_indices])

        return X, y

    def to_sklearn_predict_matrix(self):
        assert self.df is not None
        assert self.probe_suite is not None

        (
            index_to_probe_suite_type_dict,
            index_to_label_name,
        ) = self.probe_suite.index_to_probe_suite_type_dict()

        df = self.df[
            ~self.df["sample_index"].isin(self.probe_suite.used_indices)
        ].copy()

        # Convert to int from categorrical
        df["epoch"] = df["epoch"].astype(int)

        df.sort_values(["sample_index", "epoch"], inplace=True)
        sample_groups = df.groupby("sample_index", sort=False)

        losses = sample_groups["loss"].agg(list)
        sample_indices = losses.index

        X = np.asarray(losses.to_list())

        return X, sample_indices, index_to_label_name
