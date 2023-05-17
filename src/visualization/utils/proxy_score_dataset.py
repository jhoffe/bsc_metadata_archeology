import pyarrow.parquet as pq
import os
import pandas as pd


class ProxyDataset:
    def __init__(self, proxy_dataset_path: os.PathLike):
        self.proxy_df = None
        self.proxy_dataset_path = proxy_dataset_path
        self.proxy_dataset = pq.ParquetDataset(proxy_dataset_path)

    def load(self, columns: list = None):
        self.proxy_df = self.proxy_dataset.read(columns=columns).to_pandas()

        return self.proxy_df


def calculate_proxy_score_per_epoch(proxy_df: pd.DataFrame, proxy_name: str):
    return proxy_df.groupby(["sample_indices"]).agg({proxy_name: "cumsum"})
