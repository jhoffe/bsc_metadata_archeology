import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from scipy.stats import spearmanr
import click

from src.visualization.utils.proxy_score_dataset import (
    ProxyDataset,
    calculate_proxy_score_per_epoch,
)
from src.visualization.utils.plot_utils import plot_styles


def proxy_correlation_plot(
    ds: os.PathLike, orig_dataset: str, include_loss: bool = True
):
    proxy_dataset = ProxyDataset(ds)
    proxy_df = proxy_dataset.load()

    proxy_df["epoch"] = proxy_df["epoch"].astype(int)
    proxy_df.sort_values(by=["epoch", "sample_indices"], inplace=True)

    proxy_df["p_L_cs"] = calculate_proxy_score_per_epoch(proxy_df, "p_L")
    proxy_df["p_max_cs"] = calculate_proxy_score_per_epoch(proxy_df, "p_max")
    proxy_df["H_cs"] = calculate_proxy_score_per_epoch(proxy_df, "H")
    proxy_df["loss_cs"] = calculate_proxy_score_per_epoch(proxy_df, "loss")

    dataset = torch.load(f"data/processed/{orig_dataset}/train_c_scores.pt")

    idx_to_c_score = {idx: 1 - c_score for idx, (_, _, c_score) in enumerate(dataset)}

    def calculate_spearmanr(proxy_df: pd.DataFrame, epoch: int, proxy_name: str):
        proxy_epoch = proxy_df[proxy_df["epoch"] == epoch]

        proxy_scores = proxy_epoch[proxy_name].values
        c_scores = np.array(
            [idx_to_c_score[idx] for idx in proxy_epoch["sample_indices"].values]
        )

        return spearmanr(proxy_scores, c_scores).correlation

    proxies = (
        ["p_L_cs", "p_max_cs", "H_cs", "loss_cs"]
        if include_loss
        else ["p_L_cs", "p_max_cs", "H_cs"]
    )

    epoch_range = proxy_df["epoch"].unique()

    fig, ax = plt.subplots(figsize=(3, 4))
    _, _, _, plot_titles = plot_styles()
    for proxy in proxies:
        print("Working on proxy: ", proxy)
        corrs = np.array(
            [abs(calculate_spearmanr(proxy_df, epoch, proxy)) for epoch in epoch_range]
        )
        ax.plot(epoch_range, corrs, label=proxy)


    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Spearman correlation")
    plt.title(
        f"Rank Correlation Plot \n {plot_titles[orig_dataset]}"
    )
    # add grid
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    save_path = (
        f"reports/figures/{orig_dataset}/proxy_correlation_plot_loss.png"
        if include_loss
        else f"reports/figures/{orig_dataset}/proxy_correlation_plot_no_loss.png"
    )
    plt.savefig(save_path)


def main(ds: os.PathLike, orig_dataset: str, include_loss: bool = True):
    proxy_correlation_plot(ds, orig_dataset, include_loss)


@click.command()
@click.argument("ds", type=click.Path(exists=True))
@click.argument("orig_dataset", type=str)
@click.option("--include-loss", is_flag=True)
def main_click(ds: os.PathLike, orig_dataset: str, include_loss: bool):
    main(ds, orig_dataset, include_loss)


if __name__ == "__main__":
    main_click()
