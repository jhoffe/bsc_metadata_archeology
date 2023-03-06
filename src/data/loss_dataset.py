import pandas as pd
import pyarrow.parquet as pq
import torch


def loss_dataset(dataset: str) -> pd.DataFrame:
    losses = pq.ParquetDataset(
        f"data/processed/{dataset}/losses", use_legacy_dataset=False
    )
    return losses.read().to_pandas()


if __name__ == "__main__":
    df = loss_dataset("cifar10")

    print(df.head())

    print("Epochs=", df["epoch"])

    all_pos = list(range(0, 50000))
    unique_pos = df["sample_index"].unique().tolist()
    print(len(unique_pos))
    probe_suite = torch.load("data/processed/cifar10/train_probe_suite.pt")
