import matplotlib.pyplot as plt
import torch

from src.data.idx_to_label_names import get_idx_to_label_names_cifar100

# cifar10 = torch.load("data/processed/cifar10/train_probe_suite.pt")
cifar100 = torch.load("data/processed/cifar100/train_probe_suite.pt")

idx_to_label = get_idx_to_label_names_cifar100()

# print(cifar10.random_outputs[0])
random_outputs = cifar100.corrupted
print(random_outputs[0][0])

testid = 478
# Plot the image
plt.imshow(random_outputs[testid][0][0].permute(1, 2, 0))
# Title
label = idx_to_label[random_outputs[testid][0][1]]
plt.title(f"Random Output w. label {label}")
plt.show()
