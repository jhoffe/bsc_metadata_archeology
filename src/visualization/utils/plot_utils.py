# imports
import pandas as pd
from src.data.loss_dataset import LossDataset


def get_loss_dataset(dataset_path: str) -> pd.DataFrame:
    """Load the loss dataset"""
    dataset = LossDataset(dataset_path)
    dataset.load()

    df = dataset.df
    
    return df


def get_indices_from_probe_suite(suite: list) -> list[int]:
    return [idx for _, idx in suite]


def plot_styles():
    line_styles = ["solid", "dashed", "dashdot", "dotted"]

    marker_list = ["o", "*", "X", "P", "p", "D", "v", "^", "h", "1", "2", "3", "4"]

    marker_colors = [
        "tab:gray",
        "tab:green",
        "tab:blue",
        "tab:purple",
        "tab:orange",
        "tab:red",
        "tab:pink",
        "tab:olive",
        "tab:brown",
        "tab:cyan",
    ]

    plot_titles = {
        "cifar10": "CIFAR-10",
        "cifar100": "CIFAR-100",
        "imagenet": "ImageNet",
    }

    return line_styles, marker_list, marker_colors, plot_titles


def plot_dicts():
    ratios = {
        "Typical": [],
        "Atypical": [],
        "Random outputs": [],
        "Random inputs and outputs": [],
        "Corrupted": [],
        "Typical [Val]": [],
        "Atypical [Val]": [],
        "Random outputs [Val]": [],
        "Random inputs and outputs [Val]": [],
        "Corrupted [Val]": [],
        "Train": [],
    }

    learned = {
        "Typical": set([]),
        "Atypical": set([]),
        "Random outputs": set([]),
        "Random inputs and outputs": set([]),
        "Corrupted": set([]),
        "Typical [Val]": set([]),
        "Atypical [Val]": set([]),
        "Random outputs [Val]": set([]),
        "Random inputs and outputs [Val]": set([]),
        "Corrupted [Val]": set([]),
        "Train": set([]),
    }

    suite_names = [
        "Typical",
        "Atypical",
        "Random outputs",
        "Random inputs and outputs",
        "Corrupted",
        "Typical [Val]",
        "Atypical [Val]",
        "Random outputs [Val]",
        "Random inputs and outputs [Val]",
        "Corrupted [Val]",
    ]

    return ratios, learned, suite_names
