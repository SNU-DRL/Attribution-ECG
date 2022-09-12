import torch
from torch.utils.data import Dataset


class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        return X, y
        
    def __len__(self):
        return len(self.X)