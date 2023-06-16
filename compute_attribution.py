import argparse
import gzip
import json
import os
import pickle

import torch
from tqdm import tqdm

from src.attribution import ATTRIBUTION_METHODS, Attribution
from src.dataset import ECG_DataModule, get_eval_attr_data
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

    # compute feature attribution
    model.eval()
    model.to(device)
    
    attribution = Attribution(model, args.attr_method, args.n_samples, args.feature_mask_size, eval_attr_data["x"][0].shape[-1], device)

    attr_list = []
    for idx in tqdm(range(eval_attr_data["length"])):
        x, y = eval_attr_data["x"][idx], int(eval_attr_data["y"][idx])
        x = torch.as_tensor(x, device=device).unsqueeze(0)
        attr_x = attribution.apply(x, y)
        attr_list.append(attr_x)

    # save eval_attr_data & feature attribution
    with gzip.open(f"{args.result_dir}/eval_attr_data.pkl", "wb") as f:
        pickle.dump(eval_attr_data, f)
    with gzip.open(f"{args.result_dir}/attr_list.pkl", "wb") as f:
        pickle.dump(attr_list, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Computing feature attribution"
    )

    # Dataset
    parser.add_argument(
        "--dataset", default="icentia11k", type=str, choices=["icentia11k", "mit-bih", "st-petersburg", "mit-bih_svdb"]
    )
    parser.add_argument(
        "--dataset_path", default="./dataset/data/icentia11k.pkl", type=str
    )

    # Model
    parser.add_argument("--model_path", default="./result/model_last.pt", type=str)

    # Feature attribution method
    parser.add_argument(
        "--prob_threshold",
        default=0.9,
        type=float,
        help="select samples with higher prediction prob.",
    )
    parser.add_argument(
        "--attr_method", default="gradcam", type=str, choices=ATTRIBUTION_METHODS.keys()
    )
    parser.add_argument(
        "--n_samples",
        default=500,
        type=int,
        help="number of samples used for lime / kernel_shap",
    )
    parser.add_argument(
        "--feature_mask_size",
        default=16,
        type=int,
        help="size of a feature mask used for lime / kernel_shap",
    )

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
