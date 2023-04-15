import gzip
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

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

@torch.no_grad()
def get_attr_data(dataloader, model, prob_threshold, device):
    """
    return dictionary of data for evaluating attribution methods (samples with correct prediction with high prob.)
    """

    model.eval()
    model.to(device)

    attr_x = []
    attr_y = []
    attr_y_raw = []
    attr_prob = []

    for idx_batch, data_batch in enumerate(pbar := tqdm(dataloader)):
        idx, x, y = data_batch
        x = x.to(device)
        y_hat = model(x)
        probs = F.softmax(y_hat, dim=1)

        idx = idx.detach().numpy()
        x = x.detach().cpu().numpy()
        y = y.detach().numpy()
        probs = probs.detach().cpu().numpy()

        # 1) Remove label 0
        # 2) Select samples with prob > threshold
        for i in range(len(idx)):
            label = y[i]
            prob = probs[i]
            if label > 0 and prob[label] > prob_threshold:
                attr_x.append(x[i])
                attr_y.append(label)
                attr_y_raw.append(dataloader.dataset.y_raw[idx[i]])
                attr_prob.append(prob[label])

    data_dict = {
        "x": attr_x,
        "y": attr_y,
        "y_raw": attr_y_raw,
        "prob": attr_prob,
        "length": len(attr_prob)
    }

    return data_dict
