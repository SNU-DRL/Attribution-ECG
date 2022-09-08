import argparse
import gzip
import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn

parser = argparse.ArgumentParser(description='Attribution ECG')
parser.add_argument('--data_dir', type=str, help='path to dataset')
parser.add_argument("--gpu", default=None, type=str, help="gpu id to use")
parser.add_argument("--seed", default=0, type=int, help="random seed")


def main():
    args = parser.parse_args()
    setup(args)


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