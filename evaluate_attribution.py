import argparse
import csv
import gzip
import json
import os
import pickle

import torch

from src.metrics import EVALUATION_METRICS, evaluate_attribution
from src.setup import setup
from src.utils import replace_by_zero


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
    additional_metric_kwargs = {}
    if args.eval_metric in ["region_perturbation"]:
        additional_metric_kwargs.update({"patch_size": args.patch_size})
        additional_metric_kwargs.update({"order": args.perturb_order})
        additional_metric_kwargs.update({"perturb_func": replace_by_zero})
    metric_kwargs.update(additional_metric_kwargs)
    
    metric_result = evaluate_attribution(args.eval_metric, eval_attr_data, attr_list, model, device, metric_kwargs)
    result_file = f"{args.result_dir}/results_absolute.csv" if args.absolute else f"{args.result_dir}/results.csv"
    result_row_name = args.eval_metric + '_' + '_'.join([key+'_'+str(value) for key, value in additional_metric_kwargs.items()]) if additional_metric_kwargs else args.eval_metric
    result_row = [result_row_name, metric_result]
    with open(result_file, "a") as f:
        writer = csv.writer(f)
        writer.writerow(result_row)

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
        "--patch_size", default=16, type=int, help="size of a patch size for region perturbation",
    )
    parser.add_argument(
        "--perturb_order", default="morf", type=str, choices=["morf", "lerf"], help="order of applying region perturbation",
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
