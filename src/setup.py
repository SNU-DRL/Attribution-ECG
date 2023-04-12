import os
import random

import numpy as np
import torch


def set_random_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True  # This option shuould be set for reproducibility, but this hurts performance
    torch.backends.cudnn.benchmark = True


def setup(args):
    if args.seed is not None:
        set_random_seed(args.seed)
    if args.gpu_num is not None and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        return torch.device("cuda")
    else:
        return torch.device("cpu")
