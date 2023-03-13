import pathlib

import numpy as np

from src.data.utils.c_score_downloader import mem_score_downloader


def imagenet_c_scores():
    """Get imagenet-c scores."""
    # file = pathlib.Path("data/external/imagenet-cscores-with-filename.npz")

    # if not file.exists():
    #    c_score_downloader()

    file = pathlib.Path("data/external/imagenet_index.npz")

    if not file.exists():
        mem_score_downloader()

    scores = np.load(file, allow_pickle=True)

    return {
        "labels": scores["tr_labels"],
        "scores": scores["tr_mem"],
        "filenames": [f.decode("utf-8") for f in scores["tr_filenames"]],
    }


if __name__ == "__main__":
    c_scores = imagenet_c_scores()
