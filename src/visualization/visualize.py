import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image

train_set = torch.load("data/processed/imagenet/train.pt")
validation_set = torch.load("data/processed/imagenet/val.pt")

np.random.seed(42)
indices = np.random.randint(len(validation_set), size=6)
samples = [validation_set.samples[i] for i in indices]

fig, axs = plt.subplots(3, 2)

class_to_label = {}
idx_to_class = {idx: class_name for class_name, idx in train_set.class_to_idx.items()}

with open("data/raw/imagenet/LOC_synset_mapping.txt", "r") as f:
    lines = f.readlines()

    for line in lines:
        class_name, label = line.split(" ", 1)
        class_to_label[class_name] = label.strip()

for i, (path, label) in enumerate(samples):
    image = np.asarray(Image.open(path))

    row = i // 2
    column = i % 2

    axs[row][column].imshow(image)
    title = class_to_label[idx_to_class[label]]
    axs[row][column].set_title(title)

plt.show()
