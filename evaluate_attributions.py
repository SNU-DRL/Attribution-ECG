import argparse
import gzip
import json
import os
import pickle
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.attribution import compute_attr_x, to_np
from src.eval_attribution import evaluate_attr_x
from src.preprocess import preprocess

parser = argparse.ArgumentParser(description='Attribution ECG')
parser.add_argument('--dataset_path', default='dataset/12000_btype_new.pkl', type=str, help='path to dataset')
parser.add_argument('--model_path', default='models', type=str, help='path to models')
parser.add_argument('--results_path', default='results_eval', type=str)

parser.add_argument("--gpu", default=None, type=str, help="gpu id to use")
parser.add_argument("--seed", default=0, type=int, help="random seed")

parser.add_argument("--prob_thres", default=0.9, type=float)
parser.add_argument("--absolute", action='store_true')
parser.add_argument("--attr_bs", default=10, type=int)

parser.add_argument("--n_samples", default=200, type=int, help="number of samples used for lime or shap")
parser.add_argument("--seed_to_eval", default=None, type=int)

parser.add_argument('--attr_method', default='all')


def main():
    args = parser.parse_args()
    setup(args)

    if not os.path.isdir(args.results_path):
        os.makedirs(args.results_path, exist_ok=True)

    X, labels = pickle.load(gzip.GzipFile(args.dataset_path, 'rb'))
    X = preprocess(X) 
    X = np.expand_dims(X, [1, 2])
    y = np.array([l['btype'] for l in labels])
    y_raw = np.array([l['btype_raw'] for l in labels], dtype=object)

    if args.seed_to_eval is None:
        for seed in range(5):
            evaluate_attribution(X, y, y_raw, seed, args)
    else:
        evaluate_attribution(X, y, y_raw, args.seed_to_eval, args)

def evaluate_attribution(X, y, y_raw, seed, args):
    """
    Load model
    """
    model = torch.load(f"{args.model_path}/model_{seed}.pt").cuda().eval()
  
    """
    Get the dataset above thres
    """
    X_new, y_new, y_raw_new, prob_list = filter_dataset(model, X, y, y_raw, seed, args.prob_thres)

    """
    Get min, max prob for MoRF, LeRF curve
    """
    attr_methods_list = [
        'saliency', 
        'integrated_gradients', 
        'input_gradient', 
        'guided_backporp',
        'lrp', 
        'lime', 
        'kernel_shap', 
        'deep_lift',
        'deep_lift_shap',
        'gradcam',
        'guided_gradcam',
        'random_baseline'
    ]
    if args.attr_method == 'all':
        attr_methods = attr_methods_list
    else:
        attr_methods = [args.attr_method]

    bs = args.attr_bs
    num_batch = len(X_new) // bs
    for method in attr_methods:
        """
        Get attributions
        """
        attr_x_all = []
        if len(X_new) % bs:
            pbar_method = tqdm(range(num_batch + 1))
        else:
            pbar_method = tqdm(range(num_batch))
        for bn in pbar_method:
            X_batch = torch.from_numpy(X_new[bn * bs: (bn + 1) * bs]).cuda()
            y_batch = y_new[bn * bs: (bn + 1) * bs]
            y_raw_batch = y_raw_new[bn * bs: (bn + 1) * bs]
            if method == 'random_baseline':
                attr_x = np.random.randn(*X_batch.shape)
            else:
                attr_x = to_np(compute_attr_x(model, method, X_batch, absolute=args.absolute))
            attr_x_all.append(attr_x)
            
            pbar_method.set_description(
                f"Method: {method}"
            )
        attr_x_all = np.concatenate(attr_x_all)
        """
        Logging files
        """
        filter_method = f'thres_{args.prob_thres}'
        results_method_dir = os.path.join(args.results_path, filter_method, f'abs_{args.absolute}', f'seed_{seed}', method)
        results_method_jsonl = os.path.join(results_method_dir, f'attr_eval_per_sample.jsonl')
        results_method_json = os.path.join(results_method_dir, f'attr_eval_all.json')

        if not os.path.isdir(results_method_dir):
            os.makedirs(results_method_dir)

        
        # To empty file
        f = open(results_method_jsonl, "w")
        f.close()
        f = open(results_method_jsonl, "a")

        eval_x_all = []
        pbar_eval = tqdm(range(len(X_new)))
        """
        Evaluate attributions
        """
        for i in pbar_eval:
            sample_x = X_new[i].squeeze()
            sample_attr_x = attr_x_all[i].squeeze()
            sample_y = y_new[i]
            sample_y_raw = y_raw_new[i]
            loc, pnt, deg = evaluate_attr_x(sample_x, model, sample_attr_x, sample_y, sample_y_raw)

            sample_eval_result = {
                'y': sample_y.item(),
                'loc': loc,
                'pnt': pnt,
                'deg': deg
            }
            eval_x_all.append(sample_eval_result)
            f.write(json.dumps(sample_eval_result) + '\n')

            pbar_eval.set_description(
                f"Eval: {method}"
            )
            
        f.close()

        n_samples = len(eval_x_all)

        """
        Localization
        """
        iou_all = [s['loc']['iou'] for s in eval_x_all]
        iou = {
            'mean': np.mean(iou_all),
            'std': np.std(iou_all)
        }
        
        """
        Pointing game
        """
        pnt_acc = np.mean([s['pnt']['correct'] for s in eval_x_all])
        
        evaluation_results = {
            'n_samples': n_samples,
            'loc': {
                'iou': iou
            },
            'pnt': pnt_acc,
        }

        for deg_method in ['mean', 'linear', 'gaussian_plus']:
            """
            Degradation
            """
            true_labels = np.array([s['y'] for s in eval_x_all])
            LeRFs = np.array([s['deg'][deg_method]['LeRF'] for s in eval_x_all])
            MoRFs = np.array([s['deg'][deg_method]['MoRF'] for s in eval_x_all])

            """
            Label wise normalization
            """
            normalized_LeRFs = np.zeros_like(LeRFs)
            normalized_MoRFs = np.zeros_like(MoRFs)
            for l in np.unique(true_labels):
                label_idx = np.arange(len(true_labels))[true_labels == l]
                
                LeRF_to_normalize = LeRFs[label_idx]
                MoRF_to_normalize = MoRFs[label_idx]

                LeRF_init, LeRF_last = LeRF_to_normalize[:, 0].mean(), LeRF_to_normalize[:, -1].mean()
                MoRF_init, MoRF_last = MoRF_to_normalize[:, 0].mean(), MoRF_to_normalize[:, -1].mean()

                normalized_LeRF = (LeRF_to_normalize - LeRF_last) / (LeRF_init- LeRF_last)
                normalized_MoRF = (MoRF_to_normalize - MoRF_last) / (MoRF_init- MoRF_last)

                normalized_LeRFs[label_idx] = normalized_LeRF
                normalized_MoRFs[label_idx] = normalized_MoRF

            LeRF = np.mean(normalized_LeRFs, axis=0)
            MoRF = np.mean(normalized_MoRFs, axis=0)
            area = np.sum(LeRF - MoRF)
            
            evaluation_results[deg_method] = {
                'area': area,
                'LeRF': LeRF.tolist(),
                'MoRF': MoRF.tolist(),
            }

            results_method_plt = os.path.join(results_method_dir, f'attr_eval_curve_{deg_method}.png')
            plt.figure(figsize=(7, 7))
            plt.title(f"M: {method}, replace: {deg_method}, Area: {area:.4f}, N: {n_samples}, thres: {args.prob_thres}, is_abs: {args.absolute}")
            plt.plot(LeRF, label='LeRF')
            plt.plot(MoRF, label='MoRF')
            plt.legend()
            plt.savefig(results_method_plt, bbox_inches='tight')
            plt.close()
        
        with open(results_method_json, 'w') as f:
            json.dump(evaluation_results, f, indent=4)


@torch.no_grad()
def filter_dataset(model, X, y, y_raw, seed, prob_thres):
    """
    Load dataset
    """
    _, X_test_ds, _, y_test_ds, _, y_raw_test_ds = train_test_split(
        X, y, y_raw, train_size=6000, test_size=6000, stratify=y, random_state=seed
    )

    """
    Remove normal
    """
    class_idx = list(filter(lambda i: y_test_ds[i] != 0, np.arange(len(y_test_ds))))
    X_wo_normal = X_test_ds[class_idx]
    y_wo_normal = y_test_ds[class_idx]
    y_raw_wo_normal = y_raw_test_ds[class_idx]

    """
    Get probability
    """
    bs = 1024
    num_batch = len(class_idx) // bs

    X_above_thres = []
    y_above_thres = []
    y_raw_above_thres = []
    prob_above_thres = []

    for bn in range(num_batch + 1):
        X_batch = torch.from_numpy(X_wo_normal[bn * bs: (bn + 1) * bs]).cuda()
        y_batch = y_wo_normal[bn * bs: (bn + 1) * bs]
        y_raw_batch = y_raw_wo_normal[bn * bs: (bn + 1) * bs]
        
        y_softmax = to_np(F.softmax(model(X_batch), dim=1))
        y_prob = y_softmax[np.arange(len(y_softmax)), y_batch]

        is_above_thres = y_prob > prob_thres
        idx_filter = np.arange(len(y_prob))[is_above_thres]
        
        X_above_thres.append(to_np(X_batch)[idx_filter])
        y_above_thres.append(y_batch[idx_filter])
        y_raw_above_thres.append(y_raw_batch[idx_filter])
        prob_above_thres.append(y_prob[idx_filter])

    X_above_thres = np.concatenate(X_above_thres)
    y_above_thres = np.concatenate(y_above_thres)
    y_raw_above_thres = np.concatenate(y_raw_above_thres)
    prob_above_thres = np.concatenate(prob_above_thres)

    return X_above_thres, y_above_thres, y_raw_above_thres, prob_above_thres


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