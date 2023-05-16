import argparse
import json
import os
import pickle
import gzip

import pandas as pd
import torch

from src.metrics import EVALUATION_METRICS, evaluate_attribution
from src.setup import setup
from src.utils import visualize


def main(args):
    # device
    device = setup(args)

    # model
    model = torch.load(args.model_path, map_location=device)

    # load eval_attr_data & feature attribution
    eval_attr_data = pickle.load(gzip.GzipFile(f"{args.attr_dir}/eval_attr_data.pkl", "rb"))
    attr_list = pickle.load(gzip.GzipFile(f"{args.attr_dir}/attr_list.pkl", "rb"))

    if args.visualize:
        visualize(args.dataset, eval_attr_data, attr_list, args.vis_dir, args.n_samples_vis)

    # evaluate feature attribution methods
    # metric_result = evaluate_attribution(args.eval_metric, eval_attr_data, attr_list, model, device, args.absolute)
    for absolute in [True, False]:
        result_filename = f"{args.result_dir}/{args.attribution}_absolute.csv" if absolute else f"{args.result_dir}/{args.attribution}.csv"
        eval_result_dict = {}
        for eval_metric in EVALUATION_METRICS.keys():
            metric_result = evaluate_attribution(eval_metric, eval_attr_data, attr_list, model, device, absolute)
            eval_result_dict[eval_metric] = metric_result
        eval_result_series = pd.Series(eval_result_dict)
        eval_result_series.to_csv(result_filename, header=False)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluating feature attribution methods"
    )

    # Attribution result dir
    parser.add_argument(
        "--attr_dir", default="./result_attr", type=str
    )
    
    # Model
    parser.add_argument("--model_path", default="./result/model_last.pt", type=str)

    # Evaluation metric
    # parser.add_argument(
    #     "--eval_metric", default="attribution_localization", type=str, choices=EVALUATION_METRICS.keys()
    # )
    # parser.add_argument("--absolute", action="store_true")
    
    # Settings
    parser.add_argument(
        "--gpu_num", default=None, type=str, help="gpu number to use (default: use cpu)"
    )
    parser.add_argument("--seed", default=0, type=int, help="random seed")

    # Result
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument(
        "--n_samples_vis",
        default=20,
        type=int,
        help="number of samples for visualization",
    )
    parser.add_argument("--result_dir", default="./result_eval", type=str)
    
    args = parser.parse_args()
    os.makedirs(args.result_dir, exist_ok=True)

    # Load arguments
    with open(os.path.join(args.attr_dir, "args.json"), "r") as f:
        args_dict = json.load(f)
    args_dict.update(vars(args))
    args = argparse.Namespace(**args_dict)
    
    args.attribution = os.path.split(args.attr_dir)[-1]
    args.vis_dir = f"{args.result_dir}/vis"
    
    main(args)
