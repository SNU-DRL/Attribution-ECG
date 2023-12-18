import argparse
import json
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn import metrics

from src.dataset import ECG_DataModule
from src.setup import setup

np.random.seed(42)

def compute_threshold(labels, probs):
    # Find a threshold that maximizes Youden's J statistics
    fpr, tpr, thresholds = metrics.roc_curve(labels, probs)
    J = tpr-fpr
    threshold_idx = np.argmax(J)
    threshold = thresholds[threshold_idx]
    return threshold

def main(args):
    # # device
    # device = setup(args)

    # # dataloader
    # data_module = ECG_DataModule(args.dataset, args.dataset_path, batch_size=32, seed=args.seed)
    # test_loader = data_module.test_dataloader()

    # # model
    # model = torch.load(args.model_path, map_location=device)
    
    # model.eval()
    # model.to(device)
    
    # id_list = []
    # x_list = []
    # y_list = []
    # prob_list = []

    # print("Prepare dataset for evaluating feature attribution methods...")
    # for idx_batch, data_batch in enumerate(pbar := tqdm(test_loader)):
    #     idx, x, y = data_batch
    #     x = x.to(device)
    #     y_hat = model(x)
    #     probs = F.sigmoid(y_hat)

    #     idx = idx.detach().numpy()
    #     x = x.detach().cpu().numpy()
    #     y = y.detach().numpy()
    #     probs = probs.detach().cpu().numpy()

    #     id_list.extend(idx)
    #     x_list.extend(x)
    #     y_list.extend(y)
    #     prob_list.extend(probs)

    # y_array = np.array(y_list)
    # prob_array = np.array(prob_list)
    
    # with open(f"{args.result_dir}/y_array.pickle", "wb") as f:
    #     pickle.dump(y_array, f)
        
    # with open(f"{args.result_dir}/prob_array.pickle", "wb") as f:
    #     pickle.dump(prob_array, f)
    
    #############    
    with open(f"{args.result_dir}/y_array.pickle", "rb") as f:
       y_array =  pickle.load(f)
    with open(f"{args.result_dir}/prob_array.pickle", "rb") as f:
       prob_array =  pickle.load(f)
    #############
    
    num_classes = y_array.shape[-1]
    for class_idx in range(num_classes):
        print(f"Plotting distribution: class {class_idx}")
        save_dir = f"{args.result_dir}/{class_idx}"
        os.makedirs(save_dir, exist_ok=True)
        
        with open(f"{save_dir}/log.txt", "w") as f:
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
            
            fpr, tpr, _ = metrics.roc_curve(class_labels, class_probs)
            auc = metrics.roc_auc_score(class_labels, class_probs)
            plt.plot(fpr,tpr,label=f"AUROC={auc:.3f}")
            plt.legend(loc=4)
            plt.savefig(f"{save_dir}/roc_curve.png")
            plt.close()
            
            threshold = compute_threshold(class_labels, class_probs)
            f.write(f"Threshold (maximizes Youden's J statistics): {threshold:.3f}\n")
            f.write(f"AUROC: {auc:.3f}\n")

            sample_idx_over_threshold = np.argwhere(class_probs >= threshold)
            over_threshold_y = class_labels[sample_idx_over_threshold]           # threshold 이상 samples들의 y labels
            over_threshold_prob = class_probs[sample_idx_over_threshold]     # threshold 이상 samples들의 probs
            
            num_samples = len(class_labels)
            num_over_threshold = len(over_threshold_y)
            over_threshold_ratio = num_over_threshold / num_samples
            try:
                num_over_threshold_label1 = int(sum(over_threshold_y).item())
            except AttributeError:
                f.write("No samples with label 1 over threshold\n")
            else:
                over_threshold_accuracy = num_over_threshold_label1 / num_over_threshold
                f.write(f"samples over {threshold:.3f}: {over_threshold_ratio*100:.3f}% ({num_over_threshold}/{num_samples})\n")
                f.write(f"accuracy of samples over {threshold:.3f}: {over_threshold_accuracy*100:.3f}% ({num_over_threshold_label1}/{num_over_threshold})\n")
                
            num_samples_1 = int(np.sum(class_labels))
            top_ten_percent_rank = int(num_samples_1 * 0.1)
            if top_ten_percent_rank < 10:
                f.write("Not enough samples... set top_ten_percent_rank to 10\n")
                top_ten_percent_rank = 10
            _probs_1 = np.copy(probs_1)
            _probs_1.sort()
            top_ten_percent_threshold = _probs_1[-top_ten_percent_rank]
            f.write(f"Number of label 1 samples: {num_samples_1}\n")
            f.write(f"Number of label 1 samples with prob top 10 percent: {top_ten_percent_rank}\n")
            f.write(f"Top 10 percent threshold: {top_ten_percent_threshold:.3f}\n")
            
            top_ten_percent_indices = np.nonzero((class_probs >= top_ten_percent_threshold) & (class_labels == 1))[0]
            random_ten_indices = np.random.choice(top_ten_percent_indices, size=10, replace=False)
            
            with open(f"{save_dir}/selected_indices.csv", "w") as g:
                g.write(",".join(map(str, random_ten_indices.tolist())))


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
    parser.add_argument("--result_dir", default="./results_ptbxl/dist_12leads", type=str)

    args = parser.parse_args()
    os.makedirs(args.result_dir, exist_ok=True)

    # Save arguments
    with open(os.path.join(args.result_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    print(json.dumps(vars(args), indent=4))

    main(args)
