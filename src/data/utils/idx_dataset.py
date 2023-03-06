from torch.utils.data import Dataset


class IDXDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __getitem__(self, index: int):
        return self.dataset[index], index

    def __len__(self):
        return len(self.dataset)
