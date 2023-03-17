import requests


def get_idx_to_label_names(dataset: str) -> dict[int, str]:
    if dataset == "cifar10":
        return get_idx_to_label_names_cifar10()

    if dataset == "cifar100":
        return get_idx_to_label_names_cifar100()

    if dataset == "imagenet":
        return get_idx_to_label_names_imagenet()


def get_idx_to_label_names_imagenet() -> dict[int, str]:
    URL = "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt"  # noqa: E501
    response = requests.get(URL)

    lines = response.content.decode("utf-8")

    idx2labelname = eval(lines)

    return idx2labelname


def get_idx_to_label_names_cifar100() -> dict[int, str]:
    classes = [
        "beaver",
        "dolphin",
        "otter",
        "seal",
        "whale",
        "aquarium fish",
        "flatfish",
        "ray",
        "shark",
        "trout",
        "orchids",
        "poppies",
        "roses",
        "sunflowers",
        "tulips",
        "bottles",
        "bowls",
        "cans",
        "cups",
        "plates",
        "apples",
        "mushrooms",
        "oranges",
        "pears",
        "sweet peppers",
        "clock",
        "computer keyboard",
        "lamp",
        "telephone",
        "television",
        "bed",
        "chair",
        "couch",
        "table",
        "wardrobe",
        "bee",
        "beetle",
        "butterfly",
        "caterpillar",
        "cockroach",
        "bear",
        "leopard",
        "lion",
        "tiger",
        "wolf",
        "bridge",
        "castle",
        "house",
        "road",
        "skyscraper",
        "cloud",
        "forest",
        "mountain",
        "plain",
        "sea",
        "camel",
        "cattle",
        "chimpanzee",
        "elephant",
        "kangaroo",
        "fox",
        "porcupine",
        "possum",
        "raccoon",
        "skunk",
        "crab",
        "lobster",
        "snail",
        "spider",
        "worm",
        "baby",
        "boy",
        "girl",
        "man",
        "woman",
        "crocodile",
        "dinosaur",
        "lizard",
        "snake",
        "turtle",
        "hamster",
        "mouse",
        "rabbit",
        "shrew",
        "squirrel",
        "maple",
        "oak",
        "palm",
        "pine",
        "willow",
        "bicycle",
        "bus",
        "motorcycle",
        "pickup truck",
        "train",
        "lawn-mower",
        "rocket",
        "streetcar",
        "tank",
        "tractor",
    ]
    classes.sort()

    label2name = {k: v for k, v in enumerate(classes)}
    return label2name


def get_idx_to_label_names_cifar10() -> dict[int, str]:
    classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    classes.sort()
    label2name = {k: v for k, v in enumerate(classes)}
    return label2name
