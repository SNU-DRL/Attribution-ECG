import argparse
import gzip
import json
import os
import pickle

import torch
from tqdm import tqdm
import torch.nn.functional as F

from src.attribution import ATTRIBUTION_METHODS, Attribution
from src.dataset import ECG_DataModule#, get_eval_attr_data, get_eval_attr_data_multi_label
from src.setup import setup


def main(args):
    # device
    device = setup(args)

    # dataloader
    data_module = ECG_DataModule(args.dataset, args.dataset_path, batch_size=32, seed=args.seed)
    test_loader = data_module.test_dataloader()

    # model
    model = torch.load(args.model_path, map_location=device)
    
    # get indices list
    with open(f"{args.dist_dir}/{args.target_label}/selected_indices.csv") as f:
        indices_text = f.readlines()[0]
        indices = list(map(int, indices_text.split(",")))

    # initalize evaluator for evaluating feature attribution methods
    if args.multi_label:
        eval_attr_data = get_eval_attr_data_multi_label(args.dataset, test_loader, model, args.target_label, device, indices)
    # else:
    #     eval_attr_data = get_eval_attr_data(args.dataset, test_loader, model, args.prob_threshold, device)

    # compute feature attribution
    model.eval()
    model.to(device)
    
    attribution = Attribution(model, args.attr_method, args.n_samples, args.feature_mask_size, eval_attr_data["x"][0].shape[-1], eval_attr_data["x"][0].shape[-2], device)

    attr_list = []
    for idx in tqdm(range(eval_attr_data["length"])):
        if args.multi_label:
            x = eval_attr_data["x"][idx]
            x = torch.as_tensor(x, device=device).unsqueeze(0)
            attr_x = attribution.apply(x, args.target_label)
        else:
            x, y = eval_attr_data["x"][idx], int(eval_attr_data["y"][idx])
            x = torch.as_tensor(x, device=device).unsqueeze(0)
            attr_x = attribution.apply(x, y)
        attr_list.append(attr_x)

    # save eval_attr_data & feature attribution
    with gzip.open(f"{args.result_dir}/eval_attr_data.pkl", "wb") as f:
        pickle.dump(eval_attr_data, f)
    with gzip.open(f"{args.result_dir}/attr_list.pkl", "wb") as f:
        pickle.dump(attr_list, f)



@torch.no_grad()
def get_eval_attr_data_multi_label(dataset, dataloader, model, target_label, device, indices):
    """
    returns dictionary of data for evaluating feature attribution methods.
    (samples with correct prediction with high prob.)
    """

    model.eval()
    model.to(device)

    id_list = []
    x_list = []
    y_list = []
    y_raw_list = []
    beat_spans_list = []
    prob_list = []

    for idx in indices:
        x, y = torch.tensor(dataloader.dataset.x[idx]), torch.tensor(dataloader.dataset.y[idx])
        x = x.unsqueeze(0).to(device)
        y_hat = model(x)
        probs = F.sigmoid(y_hat)

        x = x.detach().cpu().numpy()
        y = y.detach().numpy()
        probs = probs.detach().cpu().numpy()

        # 1) Select samples of target label
        # 2) Select samples with prob > threshold
    
        id_list.append(idx)
        x_list.append(x[0])
        y_list.append(y)
        y_raw_list.append(None)
        beat_spans_list.append(None)
        prob_list.append(probs[0][target_label])

    data_dict = {
        "id": id_list,
        "x": x_list,
        "y": y_list,
        "y_raw": y_raw_list,
        "beat_spans": beat_spans_list,
        "prob": prob_list,
        "length": len(prob_list)
    }

    return data_dict

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
    parser.add_argument("--dist_dir", default="./results_ptbxl/dist_12leads", type=str)

    # Model
    parser.add_argument("--model_path", default="./result_train/model_last.pt", type=str)

    # Feature attribution method
    # parser.add_argument(
    #     "--prob_threshold",
    #     default=0.9,
    #     type=float,
    #     help="select samples with high prediction prob.",
    # )
    parser.add_argument(
        "--attr_method", default="gradcam", type=str, choices=ATTRIBUTION_METHODS.keys()
    )
    parser.add_argument(
        "--n_samples",
        default=500,
        type=int,
        help="number of samples used for lime / kernel_shap / deep_shap",
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
    
    # Multi-label
    parser.add_argument("--multi_label", action="store_true")
    parser.add_argument("--target_label", default=0, type=int, help="target label for multi-label classification")

    # Result
    parser.add_argument("--result_dir", default="./result_attr", type=str)

    args = parser.parse_args()
    os.makedirs(args.result_dir, exist_ok=True)

    # Save arguments
    with open(os.path.join(args.result_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    print(json.dumps(vars(args), indent=4))

    main(args)
