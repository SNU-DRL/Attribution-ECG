import argparse
import os

import torch

from src.evaluator import Evaluator
from src.dataset import ECG_DataModule
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
    evaluator = Evaluator(
        model, test_loader, args.prob_threshold, device, args.result_dir
    )
    if args.attr_method == "all":
        for attr_method in ATTRIBUTION_METHODS:
            evaluator.eval(attr_method, args.absolute)
    else:
        evaluator.eval(args.attr_method, args.absolute)


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
    parser.add_argument("--attr_method", default="saliency", type=str)
    parser.add_argument("--absolute", action="store_true")
    parser.add_argument(
        "--n_samples",
        default=200,
        type=int,
        help="number of samples used for lime or shap",
    )

    # Settings
    parser.add_argument("--gpu_num", default=None, type=str)
    parser.add_argument("--seed", default=0, type=int, help="random seed")

    # Result
    parser.add_argument("--result_dir", default="./result_eval", type=str)

    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)

    main(args)
