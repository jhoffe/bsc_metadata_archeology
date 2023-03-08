import pathlib

import numpy as np

from src.data.utils.c_score_downloader import c_score_downloader


def imagenet_c_scores():
    """Get imagenet-c scores."""
    file = pathlib.Path("data/external/imagenet-cscores-with-filename.npz")

    if not file.exists():
        c_score_downloader()

    cscores = np.load(file, allow_pickle=True)

    return {
        "labels": cscores["labels"],
        "scores": cscores["scores"],
        "filenames": cscores["filenames"],
    }
