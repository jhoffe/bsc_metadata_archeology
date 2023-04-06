import csv
from os import path

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode


class ImagenetValidationDataset(Dataset):
    def __init__(
        self,
        input_filepath: str,
        class_to_idx: dict,
        transform=None,
    ) -> None:
        super().__init__()
        solution_path = path.join(input_filepath, "LOC_val_solution.csv")
        self.transform = transform

        with open(solution_path, "r") as f:
            reader = csv.reader(f, delimiter=",")
            self.samples = [
                (
                    path.join(
                        input_filepath,
                        "ILSVRC/Data/CLS-LOC/val",
                        row[0] + ".JPEG",
                    ),
                    class_to_idx[row[1].split(" ")[0]],
                )
                for i, row in enumerate(reader)
                if i != 0
            ]

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = torchvision.io.read_image(image_path, ImageReadMode.RGB)

        if self.transform is not None:
            image = self.transform(image)
        assert image.shape == torch.Size([3, 224, 224])

        return image, label

    def __len__(self):
        return len(self.samples)
