import gzip
import pickle

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.utils import preprocess


class ECG_DataModule:
    def __init__(self, dataset_path: str, batch_size: int, seed: int):
        """
        Args:
            dataset_path (str): add an explanation here...
            seed (int): random seed
        """
        self.dataset_path = dataset_path
        self.seed = seed
        self.batch_size = batch_size

        X, labels = pickle.load(gzip.GzipFile(self.dataset_path, "rb"))
        X = preprocess(X)
        X = np.expand_dims(X, [1, 2])  # shape: (12000, 1, 2049)
        y = np.array([l["btype"] for l in labels])  # Extract btype label (beat label)
        y_raw = np.array([l["btype_raw"] for l in labels], dtype=object)

        X_train, X_test, y_train, y_test, y_raw_train, y_raw_test = train_test_split(
            X, y, y_raw, train_size=6000, test_size=6000, stratify=y, random_state=seed
        )

        self.train_set = ECG_Dataset(X_train, y_train, y_raw_train)
        print(f"Loaded dataset for training: {len(self.train_set)}")
        self.test_set = ECG_Dataset(X_test, y_test, y_raw_test)
        print(f"Loaded dataset for test: {len(self.test_set)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_set, pin_memory=True, batch_size=self.batch_size, shuffle=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, pin_memory=True, batch_size=self.batch_size, shuffle=False
        )


class ECG_Dataset(Dataset):
    def __init__(self, X, y, y_raw, prob=None):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.y_raw = y_raw
        self.prob = prob

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        return idx, X, y
