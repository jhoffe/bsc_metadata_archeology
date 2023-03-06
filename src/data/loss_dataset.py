import pyarrow.parquet as pq


def loss_dataset(dataset: str) -> None:
    losses = pq.ParquetDataset('data/processed/losses_'+dataset, use_legacy_dataset=False)

    df = losses.read().to_pandas()

    return df

    

if __name__ == "__main__":

    loss_dataset('cifar10', 'ost')