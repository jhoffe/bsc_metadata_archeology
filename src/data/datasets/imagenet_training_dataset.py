import os

from torchvision.datasets import ImageFolder
import pickle


class ImageNetTrainingDataset(ImageFolder):
    def __init__(self, root, c_scores=None, transform=None, target_transform=None):
        cache_path = "cache.pkl"

        if os.path.exists(cache_path):
            self.load_cache(cache_path)
        else:
            super().__init__(root, transform, target_transform)
            self.create_cache(cache_path)

        if c_scores is not None:
            path_to_scores = {
                path: score
                for path, score in zip(c_scores["filenames"], c_scores["scores"])
            }
            self.score = [
                path_to_scores[os.path.basename(path)] for path, _ in self.imgs
            ]
        else:
            self.score = None

    def load_cache(self, cache_path: str):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)

        self.class_to_idx = cache["class_to_idx"]
        self.classes = cache["classes"]
        self.samples = cache["samples"]
        self.imgs = cache["imgs"]
        self.transform = cache["transform"]
        self.target_transform = cache["target_transform"]
        self.root = cache["root"]
        self.loader = cache["loader"]

    def create_cache(self, cache_path: str):
        cache = {
            "class_to_idx": self.class_to_idx,
            "classes": self.classes,
            "samples": self.samples,
            "imgs": self.imgs,
            "transform": self.transform,
            "target_transform": self.target_transform,
            "root": self.root,
            "loader": self.loader
        }

        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.score is None:
            return sample, target

        return sample, target, self.score[index]
