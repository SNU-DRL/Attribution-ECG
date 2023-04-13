import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.attribution import compute_attr_x, evaluate_attr_x
from src.dataset import ECG_Dataset


class Evaluator:
    def __init__(self, model, loader, prob_threshold, batch_size, device):
        self.model = model
        self.loader = loader
        self.prob_threshold = prob_threshold
        self.batch_size = batch_size
        self.device = device

        self.model.eval()
        self.model.to(self.device)

        self.attr_loader = self.build_attr_loader()

    def eval(self, attr_method, absolute):
        attr_list = []
        print(f"Attribution method: {attr_method}, absolute: {absolute}")
        for idx_batch, data_batch in enumerate(pbar := tqdm(self.attr_loader)):
            idx, x, y = data_batch
            x = x.to(self.device)
            if attr_method == "random_baseline":
                attr_x = np.random.randn(*x.shape)
            else:
                attr_x = compute_attr_x(self.model, attr_method, x, absolute=absolute)
                attr_x = attr_x.detach().cpu().numpy()
            attr_list.append(attr_x)

        attr_all = np.concatenate(attr_list) # shape: (2820, 1, 1, 2049)

        """
        Logging files
        """
        filter_method = f"thres_{args.prob_thres}"
        results_method_dir = os.path.join(
            args.results_path,
            filter_method,
            f"abs_{args.absolute}",
            f"seed_{seed}",
            method,
        )
        results_method_jsonl = os.path.join(
            results_method_dir, f"attr_eval_per_sample.jsonl"
        )
        results_method_json = os.path.join(results_method_dir, f"attr_eval_all.json")

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
            loc, pnt, deg = evaluate_attr_x(
                sample_x, model, sample_attr_x, sample_y, sample_y_raw
            )

            sample_eval_result = {
                "y": sample_y.item(),
                "loc": loc,
                "pnt": pnt,
                "deg": deg,
            }
            eval_x_all.append(sample_eval_result)
            f.write(json.dumps(sample_eval_result) + "\n")

            pbar_eval.set_description(f"Eval: {method}")

        f.close()

        n_samples = len(eval_x_all)

        """
        Localization
        """
        iou_all = [s["loc"]["iou"] for s in eval_x_all]
        iou = {"mean": np.mean(iou_all), "std": np.std(iou_all)}

        """
        Pointing game
        """
        pnt_acc = np.mean([s["pnt"]["correct"] for s in eval_x_all])

        evaluation_results = {
            "n_samples": n_samples,
            "loc": {"iou": iou},
            "pnt": pnt_acc,
        }

        for deg_method in ["mean", "linear", "gaussian_plus"]:
            """
            Degradation
            """
            true_labels = np.array([s["y"] for s in eval_x_all])
            LeRFs = np.array([s["deg"][deg_method]["LeRF"] for s in eval_x_all])
            MoRFs = np.array([s["deg"][deg_method]["MoRF"] for s in eval_x_all])

            """
            Label wise normalization
            """
            normalized_LeRFs = np.zeros_like(LeRFs)
            normalized_MoRFs = np.zeros_like(MoRFs)
            for l in np.unique(true_labels):
                label_idx = np.arange(len(true_labels))[true_labels == l]

                LeRF_to_normalize = LeRFs[label_idx]
                MoRF_to_normalize = MoRFs[label_idx]

                LeRF_init, LeRF_last = (
                    LeRF_to_normalize[:, 0].mean(),
                    LeRF_to_normalize[:, -1].mean(),
                )
                MoRF_init, MoRF_last = (
                    MoRF_to_normalize[:, 0].mean(),
                    MoRF_to_normalize[:, -1].mean(),
                )

                normalized_LeRF = (LeRF_to_normalize - LeRF_last) / (
                    LeRF_init - LeRF_last
                )
                normalized_MoRF = (MoRF_to_normalize - MoRF_last) / (
                    MoRF_init - MoRF_last
                )

                normalized_LeRFs[label_idx] = normalized_LeRF
                normalized_MoRFs[label_idx] = normalized_MoRF

            LeRF = np.mean(normalized_LeRFs, axis=0)
            MoRF = np.mean(normalized_MoRFs, axis=0)
            area = np.sum(LeRF - MoRF)

            evaluation_results[deg_method] = {
                "area": area,
                "LeRF": LeRF.tolist(),
                "MoRF": MoRF.tolist(),
            }

            results_method_plt = os.path.join(
                results_method_dir, f"attr_eval_curve_{deg_method}.png"
            )
            plt.figure(figsize=(7, 7))
            plt.title(
                f"M: {method}, replace: {deg_method}, Area: {area:.4f}, N: {n_samples}, thres: {args.prob_thres}, is_abs: {args.absolute}"
            )
            plt.plot(LeRF, label="LeRF")
            plt.plot(MoRF, label="MoRF")
            plt.legend()
            plt.savefig(results_method_plt, bbox_inches="tight")
            plt.close()

        with open(results_method_json, "w") as f:
            json.dump(evaluation_results, f, indent=4)

    @torch.no_grad()
    def build_attr_loader(self):
        attr_x = []
        attr_y = []
        attr_y_raw = []
        attr_prob = []

        for idx_batch, data_batch in enumerate(pbar := tqdm(self.loader)):
            idx, x, y = data_batch
            x = x.to(self.device)
            y_hat = self.model(x)
            probs = F.softmax(y_hat, dim=1)

            idx = idx.detach().numpy()
            x = x.detach().cpu().numpy()
            y = y.detach().numpy()
            probs = probs.detach().cpu().numpy()

            # 1) Remove label 0
            # 2) Select samples with prob > threshold
            for i in range(len(idx)):
                label = y[i]
                prob = probs[i]
                if label > 0 and prob[label] > self.prob_threshold:
                    attr_x.append(x[i])
                    attr_y.append(label)
                    attr_y_raw.append(self.loader.dataset.y_raw[idx[i]])
                    attr_prob.append(prob[label])

        attr_dataset = ECG_Dataset(np.array(attr_x), np.array(attr_y), attr_y_raw, attr_prob)
        return DataLoader(attr_dataset, pin_memory=True, batch_size=self.batch_size, shuffle=False)
