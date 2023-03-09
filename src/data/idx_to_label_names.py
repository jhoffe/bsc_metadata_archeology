import requests


def get_idx_to_label_names() -> dict[int, str]:
    URL = "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt"  # noqa: E501
    response = requests.get(URL)

    lines = response.content.decode("utf-8")

    idx2labelname = eval(lines)

    return idx2labelname
