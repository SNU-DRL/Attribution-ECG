import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.dataset import ECG_DataModule
from src.setup import setup


def main(args):
    # device
    device = setup(args)

    # dataloader
    data_module = ECG_DataModule(args.dataset, args.dataset_path, batch_size=32, seed=args.seed)
    test_loader = data_module.test_dataloader()

    # model
    model = torch.load(args.model_path, map_location=device)
    
    model.eval()
    model.to(device)
    
    id_list = []
    x_list = []
    y_list = []
    prob_list = []

    print("Prepare dataset for evaluating feature attribution methods...")
    for idx_batch, data_batch in enumerate(pbar := tqdm(test_loader)):
        idx, x, y = data_batch
        x = x.to(device)
        y_hat = model(x)
        probs = F.sigmoid(y_hat)

        idx = idx.detach().numpy()
        x = x.detach().cpu().numpy()
        y = y.detach().numpy()
        probs = probs.detach().cpu().numpy()

        id_list.extend(idx)
        x_list.extend(x)
        y_list.extend(y)
        prob_list.extend(probs)

    y_array = np.array(y_list)
    prob_array = np.array(prob_list)
    
    num_classes = y_list[0].shape[-1]
    for class_idx in range(num_classes):
        print(f"Plotting distribution: class {class_idx}")
        save_dir = f"{args.result_dir}/{class_idx}"
        os.makedirs(save_dir, exist_ok=True)
        
        class_labels = y_array[:, class_idx]
        class_probs = prob_array[:, class_idx]
        probs_0 = class_probs[class_labels == 0]
        probs_1 = class_probs[class_labels == 1]

        kwargs = dict(alpha=0.5, bins=50)
        plt.hist(probs_0, **kwargs, color='g', label='0')
        plt.hist(probs_1, **kwargs, color='b', label='1')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/dist.png")
        plt.close()
        
        kwargs = dict(alpha=0.5, bins=50, density=True, stacked=True)
        plt.hist(probs_0, **kwargs, color='g', label='0')
        plt.hist(probs_1, **kwargs, color='b', label='1')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/dist_density.png")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Computing feature attribution"
    )

    # Dataset
    parser.add_argument(
        "--dataset", default="icentia11k", type=str, choices=["mitdb", "svdb", "incartdb", "icentia11k", "ptbxl"]
    )
    parser.add_argument(
        "--dataset_path", default="./dataset/data/icentia11k.pkl", type=str
    )

    # Model
    parser.add_argument("--model_path", default="./result_train/model_last.pt", type=str)

    # Settings
    parser.add_argument(
        "--gpu_num", default=None, type=str, help="gpu number to use (default: use cpu)"
    )
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    
    # Result
    parser.add_argument("--result_dir", default="./result_attr", type=str)

    args = parser.parse_args()
    os.makedirs(args.result_dir, exist_ok=True)

    # Save arguments
    with open(os.path.join(args.result_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    print(json.dumps(vars(args), indent=4))

    main(args)
