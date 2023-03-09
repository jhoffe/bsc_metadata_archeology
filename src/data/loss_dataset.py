from typing import Optional

import pandas as pd
import pyarrow.parquet as pq


class LossDataset:
    dataset_path: str
    dataset: pq.ParquetDataset
    df: Optional[pd.DataFrame]

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.dataset = pq.ParquetDataset(dataset_path)

    def load(self) -> "LossDataset":
        self.df = self.dataset.read().to_pandas()

        return self
