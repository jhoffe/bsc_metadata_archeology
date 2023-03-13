import requests


def c_score_downloader():
    """Takes a list of files and downloads these into the data/external folder."""
    cscores = [
        "cifar10-cscores-orig-order.npz",
        "cifar100-cscores-orig-order.npz",
        "imagenet-cscores-with-filename.npz",
    ]
    for file in cscores:
        URL = f"https://pluskid.github.io/structural-regularity/cscores/{file}"
        r = requests.get(URL)
        open(f"data/external/{file}", "wb").write(r.content)


def mem_score_downloader():
    """Takes a list of files and downloads these into the data/external folder."""
    cscores = [
        "cifar100_infl_matrix.npz",
        "imagenet_index.npz"
    ]
    for file in cscores:
        URL = f"https://pluskid.github.io/influence-memorization/data/{file}"
        r = requests.get(URL)
        open(f"data/external/{file}", "wb").write(r.content)


if __name__ == "__main__":
    c_score_downloader()
    mem_score_downloader()
