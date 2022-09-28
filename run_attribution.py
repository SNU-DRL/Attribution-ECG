import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
import copy
import random
from collections import Counter
from tqdm import tqdm
import pickle
import gzip
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import src.models
from src.attribution import to_np, compute_attr_x
from src.dataset import SimpleDataset
from src.preprocess import preprocess


parser = argparse.ArgumentParser(description='Attribution ECG')
parser.add_argument('--dataset_path', default='dataset/12000_btype_new.pkl', type=str, help='path to dataset')
parser.add_argument('--results_path', default='results/', type=str)
parser.add_argument("--gpu", default=None, type=str, help="gpu id to use")
parser.add_argument("--seed", default=0, type=int, help="random seed")

parser.add_argument("--model", default='resnet18_7', type=str)

parser.add_argument("--input_idx", default=0, type=int, help="Index of a test input")
parser.add_argument("--attr_method", default='saliency', type=str, help="Attribution method, if all => all methods")
parser.add_argument("--n_samples", default=128, type=int, help="number of samples used for lime or shap")
parser.add_argument("--feature_mask_window", default=16, type=int, help="window of feature mask for Feature Ablation method")

def main():
    args = parser.parse_args()
    setup(args)

    ## Load data
    X, labels = pickle.load(gzip.GzipFile(args.dataset_path, 'rb'))
    X = preprocess(X)           # sample wise standardization
    X_exp = np.expand_dims(X, [1, 2])    # shape: (12000, 1, 2049)
    y = np.array([l['btype'] for l in labels]) # Extract btype label (beat label)
    y_btype_raw = np.array([l['btype_raw'] for l in labels])

    X_train_ds, X_test_ds, y_train_ds, y_test_ds, y_train_btype_raw_ds, y_test_btype_raw_ds = train_test_split(
        X_exp, y, y_btype_raw, train_size=0.5, stratify=y, random_state=args.seed
    )

    ## Load model
    model = torch.load("results/model_" + str(args.seed) + ".pt").cuda().eval()

    ## Prepare input tensor
    input_tensor = torch.Tensor(X_test_ds[args.input_idx]).unsqueeze(0).cuda()
    
    ## Compute attribution for an input
    attr_x = compute_attr_x(model, args.attr_method, input_tensor, args.n_samples, args.feature_mask_window)
    torch.save(attr_x, args.results_path + args.attr_method + "_" + str(args.input_idx) + ".pt")

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