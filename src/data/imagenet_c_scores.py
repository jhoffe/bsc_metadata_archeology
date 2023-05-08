import pathlib
from typing import Optional

import numpy as np

from src.data.utils.c_score_downloader import downloader


def imagenet_c_scores(use_cscores: Optional[bool] = None):
    """Get imagenet c-scores."""
    if use_cscores is None:
        return None

    if not use_cscores:
        file = pathlib.Path("data/external/imagenet_index.npz")

        if not file.exists():
            downloader()

        scores = np.load(file, allow_pickle=True)

        return {
            "labels": scores["tr_labels"],
            "scores": scores["tr_mem"],
            "filenames": [f.decode("utf-8") for f in scores["tr_filenames"]],
        }

    elif use_cscores:
        file = pathlib.Path("data/external/imagenet-cscores-with-filename.npz")

        if not file.exists():
            downloader()

        scores = np.load(file, allow_pickle=True)

        return {
            "labels": scores["labels"],
            "scores": 1.0 - scores["scores"],
            "filenames": [f.decode("utf-8") for f in scores["filenames"]],
        }

    return None


if __name__ == "__main__":
    c_scores = imagenet_c_scores()
