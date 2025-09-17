from torch.utils.data import Dataset


class SimpleStackDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self._length = min(len(d) for d in datasets)

    def __getitem__(self, index):
        return tuple(dataset[index] for dataset in self.datasets)

    def __len__(self):
        return self._length
