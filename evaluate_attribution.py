import argparse
import json
import os

import pandas as pd
import torch

from src.attribution import ATTRIBUTION_METHODS
from src.dataset import ECG_DataModule, get_eval_attr_data
from src.evaluator import Evaluator
from src.setup import setup


def main(args):
    # device
    device = setup(args)

    # dataloader
    data_module = ECG_DataModule(args.dataset, args.dataset_path, batch_size=32, seed=args.seed)
    test_loader = data_module.test_dataloader()

    # model
    model = torch.load(args.model_path, map_location=device)

    # initalize evaluator for evaluating feature attribution methods
    eval_attr_data = get_eval_attr_data(args.dataset, test_loader, model, args.prob_threshold, device)
    evaluator = Evaluator(model, eval_attr_data, device, args.result_dir)

    # compute feature attribution
    attr_list = evaluator.compute_attribution(
        args.attr_method, args.absolute, args.n_samples
    )

    if args.visualize:
        evaluator.visualize(args.dataset, attr_list)

    # evaluate feature attribution methods
    loc_score_mean, loc_score_std = evaluator.get_localization_score(attr_list)
    pnt_accuracy = evaluator.get_pointing_game_accuracy(attr_list)
    deg_score = evaluator.get_degradation_score(attr_list, "mean", args.deg_window_size)

    # save results
    results = pd.Series(
        {
            "loc_score_mean": loc_score_mean,
            "loc_score_std": loc_score_std,
            "pnt_accuracy": pnt_accuracy,
            "deg_score": deg_score,
        }
    )
    results.to_csv(f"{args.result_dir}/result.csv", header=["value"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluating feature attribution methods"
    )

    # Dataset
    parser.add_argument(
        "--dataset", default="icentia11k", type=str, choices=["icentia11k", "mit-bih"]
    )
    parser.add_argument(
        "--dataset_path", default="./data/12000_btype_new.pkl", type=str
    )

    # Model
    parser.add_argument("--model_path", default="./result/model_last.pt", type=str)

    # Feature attribution method
    parser.add_argument(
        "--prob_threshold",
        default=0.75,
        type=float,
        help="select samples with higher prediction prob.",
    )
    parser.add_argument(
        "--attr_method", default="gradcam", type=str, choices=ATTRIBUTION_METHODS.keys()
    )
    parser.add_argument("--absolute", action="store_true")
    parser.add_argument(
        "--n_samples",
        default=200,
        type=int,
        help="number of samples used for lime / shap",
    )

    # Evaluation metrics for feature attribution methods
    parser.add_argument(
        "--deg_window_size",
        default=16,
        type=int,
        help="window size for degradation score",
    )

    # Settings
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument(
        "--gpu_num", default=None, type=str, help="gpu number to use (default: use cpu)"
    )
    parser.add_argument("--seed", default=0, type=int, help="random seed")

    # Result
    parser.add_argument("--result_dir", default="./result_eval", type=str)

    args = parser.parse_args()
    os.makedirs(args.result_dir, exist_ok=True)

    # Save arguments
    with open(os.path.join(args.result_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    print(json.dumps(vars(args), indent=4))

    main(args)
