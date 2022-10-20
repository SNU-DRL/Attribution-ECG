import argparse
import copy
import gzip
import json
import os
import pickle
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.models
from src.dataset import SimpleDataset
from src.preprocess import preprocess

parser = argparse.ArgumentParser(description='Attribution ECG')
parser.add_argument('--dataset_path', default='dataset/12000_btype_new.pkl', type=str, help='path to dataset')
parser.add_argument('--model_path', default='models', type=str)
parser.add_argument("--gpu", default=None, type=str, help="gpu id to use")
parser.add_argument("--seed", default=0, type=int, help="random seed")

parser.add_argument("--model", default='resnet18_7', type=str)

parser.add_argument("--lr", default=5e-4, type=float)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--n_epochs", default=50, type=int)
parser.add_argument("--weight_decay", default=1e-7, type=float)


def main():
    args = parser.parse_args()
    setup(args)

    if not os.path.isdir(args.model_path):
        os.makedirs(args.model_path)

    X, labels = pickle.load(gzip.GzipFile(args.dataset_path, 'rb'))
    X = preprocess(X)           # sample wise standardization
    X = np.expand_dims(X, [1, 2])    # shape: (12000, 1, 2049)
    y = np.array([l['btype'] for l in labels]) # Extract btype label (beat label) 

    test_acc_list = []
    for seed in range(5):
        test_acc = train(X, y, seed, args)
        test_acc_list.append(test_acc.item())

    results_dict = {
        'test_mean': np.mean(test_acc_list).item(),
        'test_std': np.std(test_acc_list).item()
    }

    results_json = os.path.join(args.model_path, 'results.json')
    with open(results_json, 'w') as f:
        json.dump(results_dict, f, indent=4)
    

def train(X, y, seed, args):
    device = torch.device("cuda")

    """
    Split train, test
    """
    X_train_ds, X_test_ds, y_train_ds, y_test_ds = train_test_split(
        X, y, train_size=6000, test_size=6000, stratify=y, random_state=seed
    )
    # print("Train: ", Counter(y_train))
    # print("Test: ", Counter(y_test))
    
    """
    Model
    """
    model = vars(src.models)[args.model](in_channels=1, num_classes=3).to(device)

    """
    Dataloader
    """
    train_ds = SimpleDataset(X_train_ds, y_train_ds)
    test_ds = SimpleDataset(X_test_ds, y_test_ds)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, num_workers=8)

    """
    Optimizer
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    """
    Train
    """
    best_test_acc = 0
    best_model = copy.deepcopy(model)
    pbar_0 = tqdm(range(args.n_epochs), position=0)
    for ep in pbar_0:
        pbar_1 = tqdm(train_dl, total=len(train_dl), position=1)
        pbar_2 = tqdm(test_dl, total=len(test_dl), position=3)

        model.train()
        y_train_true_list = []
        y_train_pred_list = []
        for train_batch in pbar_1: 
            X_train, y_train_true = train_batch[0].to(device), train_batch[1].to(device)
            
            optimizer.zero_grad()
            y_train_pred = model(X_train)
            loss = criterion(y_train_pred, y_train_true)
            loss.backward()
            optimizer.step()

            y_train_pred_onehot = torch.argmax(y_train_pred, -1)
            y_train_true_list.append(y_train_true)
            y_train_pred_list.append(y_train_pred_onehot)

            train_acc = (torch.cat(y_train_true_list) == torch.cat(y_train_pred_list)).float().mean()
            
            pbar_1.set_description(
                f"Train: [{ep + 1:03d}] "
                f"Acc: {train_acc:.4f} "
            )

    with torch.no_grad():
        model.eval()
        y_test_true_list = []
        y_test_pred_list = []
        for test_batch in pbar_2: 
            X_test, y_test_true = test_batch[0].to(device), test_batch[1].to(device)

            y_test_pred = model(X_test)
            y_test_pred_onehot = torch.argmax(y_test_pred, -1)
            
            y_test_true_list.append(y_test_true)
            y_test_pred_list.append(y_test_pred_onehot)

            test_acc = (torch.cat(y_test_true_list) == torch.cat(y_test_pred_list)).float().mean()
            
            pbar_2.set_description(
                f" Test: [{ep + 1:03d}] "
                f"Acc: {test_acc:.4f} "
            )
    torch.save(model, os.path.join(args.model_path, f'model_{seed}.pt'))

    return test_acc

def setup(args):
    setup_gpu(args)
    setup_seed(args)


def setup_gpu(args):
    if args.gpu is not None:
        args.gpu = args.gpu # to remove bracket
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def setup_seed(args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        torch.backends.deterministic = True
        torch.backends.cudnn.benchmark = False
    

if __name__ == "__main__":
    main()