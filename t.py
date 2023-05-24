import torch

ps = torch.load("data/processed/cifar100/train_probe_suite_c_scores.pt")

index_to_suite = {}

for idx, suite in list(ps.index_to_suite.items()):
    if suite == "typical":
        index_to_suite[idx] = "atypical"
        continue
    if suite == "atypical":
        index_to_suite[idx] = "typical"
        continue
    index_to_suite[idx] = suite

ps.index_to_suite = index_to_suite

torch.save(ps, "data/processed/cifar100/train_probe_suite_c_scores.pt")