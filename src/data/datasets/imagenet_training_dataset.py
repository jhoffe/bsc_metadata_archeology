import os

from torchvision.datasets import ImageFolder


class ImageNetTrainingDataset(ImageFolder):
    def __init__(self, root, c_scores, transform=None, target_transform=None):
        super().__init__(root, transform, target_transform)

        path_to_scores = {
            path: score
            for path, score in zip(c_scores["filenames"], c_scores["scores"])
        }
        self.score = [path_to_scores[os.path.basename(path)] for path, _ in self.imgs]

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, self.score[index]
