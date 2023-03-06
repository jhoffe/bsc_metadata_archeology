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

    print("Epochs=", df["epoch"])

    all_pos = list(range(0, 50000))
    unique_pos = df["sample_index"].unique().tolist()
    probe_suite = torch.load("data/processed/cifar10/train_probe_suite.pt")

    all_used = []
    all_remaining = []

    missing = []

    for pos in all_pos:
        if pos not in unique_pos:
            print("Missing position=", pos)
            print("Missing in used: ", pos in probe_suite.used_indices)
            print("Missing in remaining: ", pos in probe_suite.remaining_indices)
            all_used.append(pos in probe_suite.used_indices)
            all_remaining.append(pos in probe_suite.remaining_indices)
            missing.append(pos)

    print("All remaining: ", all(all_remaining))
    print("All used: ", any(all_used))
