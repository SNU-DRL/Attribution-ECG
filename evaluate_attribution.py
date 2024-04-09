import argparse
import csv
import gzip
import json
import os
import pickle

import numpy as np
import torch

from src.metrics import EVALUATION_METRICS, evaluate_attribution
from src.setup import setup


def main(args):
    # device
    device = setup(args)

    # model
    model = torch.load(args.model_path, map_location=device)

    # load eval_attr_data & feature attribution
    eval_attr_data = pickle.load(gzip.GzipFile(f"{args.attr_dir}/eval_attr_data.pkl", "rb"))
    attr_list = pickle.load(gzip.GzipFile(f"{args.attr_dir}/attr_list.pkl", "rb"))

    # evaluate feature attribution methods
    metric_kwargs = {
        "abs": args.absolute,
    }
    if args.eval_metric in ["region_perturbation"]:
        metric_kwargs.update({
            "patch_size": args.patch_size,
            "order": args.perturb_order
        })
    if args.eval_metric in ["faithfulness_correlation"]:
        metric_kwargs.update({"subset_size": int(eval_attr_data['x'][0].size * args.subset_ratio)})
    metric_name = '_'.join([args.eval_metric, args.perturb_order]) if args.eval_metric in ["region_perturbation"] else args.eval_metric
    
    # save arguments
    args_file = f"{args.result_dir}/args_absolute_{metric_name}.json" if args.absolute else f"{args.result_dir}/args_{metric_name}.json"
    args_dict = vars(args)
    args_dict.update(metric_kwargs)
    with open(args_file, "w") as f:
        json.dump(args_dict, f, indent=4)
    
    # calculate metric scores
    metric_scores = evaluate_attribution(args.eval_metric, eval_attr_data, attr_list, model, device, metric_kwargs)
    metric_result = np.nanmean(metric_scores)
    
    # save results
    result_file = f"{args.result_dir}/results_absolute.csv" if args.absolute else f"{args.result_dir}/results.csv"
    result_row = [metric_name, metric_result]
    with open(result_file, "a") as f:
        writer = csv.writer(f)
        writer.writerow(result_row)
        
    # save scores
    scores_file = f"{args.result_dir}/scores_absolute_{metric_name}.csv" if args.absolute else f"{args.result_dir}/scores_{metric_name}.csv"
    with open(scores_file, "w") as f:
        csv_writer = csv.writer(f, delimiter="\n")
        csv_writer.writerow(metric_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluating feature attribution methods"
    )

    # Attribution result dir
    parser.add_argument(
        "--attr_dir", default="./result_attr", type=str
    )
    
    # Model
    parser.add_argument("--model_path", default="./result_train/model_last.pt", type=str)

    # Evaluation metric
    parser.add_argument(
        "--eval_metric", default="attribution_localization", type=str, choices=EVALUATION_METRICS.keys()
    )
    parser.add_argument("--absolute", action="store_true")
    
    ### For --eval_metric == region_perturbation
    parser.add_argument(
        "--patch_size", default=16, type=int, help="size of a patch for region perturbation",
    )
    parser.add_argument(
        "--perturb_order", default="morf", type=str, choices=["morf", "lerf"], help="order of applying region perturbation",
    )
    ###

    ### For --eval_metric == faithfulness_correlation
    parser.add_argument(
        "--subset_ratio", default=0.1, type=float, help="ratio of a subset for faithfulness correlation",
    )
    ###
    
    # Settings
    parser.add_argument(
        "--gpu_num", default=None, type=str, help="gpu number to use (default: use cpu)"
    )
    parser.add_argument("--seed", default=0, type=int, help="random seed")

    # Result
    parser.add_argument("--result_dir", default="./result_eval", type=str)
    
    args = parser.parse_args()
    os.makedirs(args.result_dir, exist_ok=True)

    # Load arguments
    with open(os.path.join(args.attr_dir, "args.json"), "r") as f:
        args_dict = json.load(f)
    args_dict.update(vars(args))
    args = argparse.Namespace(**args_dict)
    
    main(args)
