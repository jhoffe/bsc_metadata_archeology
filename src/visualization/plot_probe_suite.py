import random

import matplotlib.pyplot as plt
import torch

from src.data.idx_to_label_names import get_idx_to_label_names

random.seed(42)
probe_suite = torch.load("data/processed/imagenet/train_probe_suite.pt")

idx_to_label = get_idx_to_label_names()

suite_names = {
    "typical": "Typical",
    "atypical": "Atypical",
    "random_outputs": "Random outputs",
    # "random_inputs_outputs": "Random inputs and outputs",
    "corrupted": "Corrupted",
}

NROWS = 4
NCOLUMNS = 4

for suite_attr, suite_name in suite_names.items():
    suite = getattr(probe_suite, suite_attr)

    fig, axs = plt.subplots(NROWS, NCOLUMNS, figsize=(20, 20))
    fig.tight_layout()

    samples = random.sample(suite, NROWS * NCOLUMNS)

    for i, ((sample, _), ax) in enumerate(zip(samples, axs.flatten())):
        ax.imshow(sample[0].permute(1, 2, 0))
        label = idx_to_label[sample[1]]

        shortened_label = label.split(",")[0] if "," in label else label

        ax.set_title(f"Class: {shortened_label}\n Memorization Score={sample[2]:0.2f}")

    fig.suptitle(f"Probe suite: {suite_name} (n={len(suite)})", y=0.98, fontsize=20)
    fig.subplots_adjust(top=0.94, hspace=0.3)

    plt.savefig(f"reports/figures/imagenet/probe_suites/probe_suite_{suite_name}.png")
