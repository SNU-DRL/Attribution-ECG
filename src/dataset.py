import gzip
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.utils import preprocess, get_beat_spans


class ECG_DataModule:
    def __init__(self, dataset: str, dataset_path: str, batch_size: int, seed: int):
        """
        Args:
            dataset (str): mitdb OR svdb OR incartdb OR icentia11k
            dataset_path (str): path to dataset file
            batch_size (int): batch size for dataloader
            seed (int): random seed
        """
        self.dataset = dataset
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.seed = seed

        if dataset == "icentia11k":
            x, labels = pickle.load(gzip.GzipFile(self.dataset_path, "rb"))
            x = preprocess(x)
            x = np.expand_dims(x, axis=(1, 2))  # x.shape: (12000, 2049) -> (12000, 1, 1, 2049)
            y = np.array([l["btype"] for l in labels])  # Extract btype label (beat label)
            y_raw = [l["btype_raw"] for l in labels]

            x_train, x_test, y_train, y_test, y_raw_train, y_raw_test = train_test_split(
                x, y, y_raw, train_size=6000, test_size=6000, stratify=y, random_state=self.seed
            )

        elif dataset in ["mitdb", "svdb", "incartdb"]:
            data_dict = pickle.load(gzip.GzipFile(self.dataset_path, "rb"))
            train_set, test_set = data_dict["train"], data_dict["test"]
            x_train = np.expand_dims(preprocess(train_set["X"]), axis=(1,2))
            y_train, y_raw_train = np.array([Y['y'] for Y in train_set["Y"]]), [Y['y_raw'] for Y in train_set["Y"]]
            x_test = np.expand_dims(preprocess(test_set["X"]), axis=(1,2))
            y_test, y_raw_test = np.array([Y['y'] for Y in test_set["Y"]]), [Y['y_raw'] for Y in test_set["Y"]]

        self.train_set = ECG_Dataset(x_train, y_train, y_raw_train)
        print(f"Loaded dataset for training: {len(self.train_set)}")
        self.test_set = ECG_Dataset(x_test, y_test, y_raw_test)
        print(f"Loaded dataset for test: {len(self.test_set)}")

        self.num_classes = np.unique(np.concatenate([y_train, y_test])).size

    def train_dataloader(self):
        return DataLoader(
            self.train_set, pin_memory=True, batch_size=self.batch_size, shuffle=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, pin_memory=True, batch_size=self.batch_size, shuffle=False
        )


class ECG_Dataset(Dataset):
    def __init__(self, x, y, y_raw, prob=None):
        self.x = x
        self.y = y
        self.y_raw = y_raw
        self.prob = prob

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return idx, x, y

@torch.no_grad()
def get_eval_attr_data(dataset, dataloader, model, prob_threshold, device):
    """
    returns dictionary of data for evaluating feature attribution methods.
    (samples with correct prediction with high prob.)
    """

    model.eval()
    model.to(device)

    x_list = []
    y_list = []
    y_raw_list = []
    beat_spans_list = []
    prob_list = []

    print("Prepare dataset for evaluating feature attribution methods...")
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
                x_list.append(x[i])
                y_list.append(label)
                y_raw = dataloader.dataset.y_raw[idx[i]]
                y_raw_list.append(y_raw)
                beat_spans = get_beat_spans(y_raw, x[i].shape[-1], dataset)
                beat_spans_list.append(beat_spans)
                prob_list.append(prob[label])

    data_dict = {
        "x": x_list,
        "y": y_list,
        "y_raw": y_raw_list,
        "beat_spans": beat_spans_list,
        "prob": prob_list,
        "length": len(prob_list)
    }

    return data_dict
