import torch
import numpy as np
import matplotlib.pyplot as plt

data = torch.load('data/processed/cifar10/train.pt')
scores = data.score

sorted = np.argsort(scores)
typical = sorted[:250]
atypical = sorted[-250:]

plt.figure(figsize=(10, 10))
plt.imshow(data[typical[0]][0].permute(1, 2, 0))
plt.show()
