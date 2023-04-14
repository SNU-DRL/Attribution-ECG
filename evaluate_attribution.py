import argparse
import os

import pandas as pd
import torch

from src.dataset import ECG_DataModule, get_attr_loader
from src.evaluator import Evaluator
from src.setup import setup

ATTRIBUTION_METHODS = [
    "saliency",
    "integrated_gradients",
    "input_gradient",
    "guided_backporp",
    "lrp",
    "lime",
    "kernel_shap",
    "deep_lift",
    "deep_lift_shap",
    "gradcam",
    "guided_gradcam",
    "random_baseline",
]


def main(args):
    # device
    device = setup(args)

    # dataloader
    data_module = ECG_DataModule(args.dataset_path, batch_size=128, seed=args.seed)
    test_loader = data_module.test_dataloader()

    # model
    model = torch.load(args.model_path)

    # initalize attribution evaluator
    attr_loader = get_attr_loader(test_loader, model, args.prob_threshold, device)
    evaluator = Evaluator(model, attr_loader, device, args.result_dir)

    # compute attribution
    attr_list = evaluator.compute_attribution(args.attr_method, args.absolute)

    # evaluate feature attribution methods
    loc_score_mean, loc_score_std = evaluator.get_localization_score(attr_list)
    pnt_score = evaluator.get_pointing_game_score(attr_list)
    deg_score = evaluator.get_degradation_score(attr_list, "linear", args.deg_window_size)

    # save results
    results = pd.Series({
            "loc_score_mean": loc_score_mean,
            "loc_score_std": loc_score_std,
            "pnt_score": pnt_score,
            "deg_score": deg_score,
    })
    results.to_csv(f"{args.result_dir}/result.csv", header=["value"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attribution ECG")

    # Dataset
    parser.add_argument(
        "--dataset_path", default="./data/12000_btype_new.pkl", type=str
    )

    # Model
    parser.add_argument("--model_path", default="./result/model_last.pt", type=str)
    parser.add_argument(
        "--prob_threshold",
        default=0.9,
        type=float,
        help="select samples with higher prediction prob.",
    )

    # Attribution method
    parser.add_argument(
        "--attr_method", default="gradcam", type=str, choices=ATTRIBUTION_METHODS
    )
    parser.add_argument("--absolute", action="store_true")
    parser.add_argument(
        "--n_samples",
        default=200,
        type=int,
        help="number of samples used for lime or shap",
    )
    parser.add_argument("--deg_window_size", default=16, type=int)

    # Settings
    parser.add_argument("--gpu_num", default=None, type=str)
    parser.add_argument("--seed", default=0, type=int, help="random seed")

    # Result
    parser.add_argument("--result_dir", default="./result_eval", type=str)

    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)

    main(args)
