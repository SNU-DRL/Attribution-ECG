import os
import random

import numpy as np
import torch

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def set_random_seed(SEED):
    print(f"-- Set random seed: {SEED}")
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
        print(f"-- Use gpu: {args.gpu_num}")
        return torch.device(f"cuda:{args.gpu_num}")
    else:
        print(f"-- Use cpu")
        return torch.device("cpu")
