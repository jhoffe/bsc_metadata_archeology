import torch

cifar10 = torch.load("data/processed/cifar10/train_probe_suite.pt")
cifar100 = torch.load("data/processed/cifar100/train_probe_suite.pt")

print(cifar10.random_outputs[0])
print(cifar100.random_outputs[0])
