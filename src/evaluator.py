import os
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.attribution import get_attribution, evaluate_attribution
from src.dataset import ECG_Dataset


class Evaluator:
    def __init__(self, model, dataloader, prob_threshold, device, result_dir):
        self.model = model
        self.prob_threshold = prob_threshold
        self.device = device
        self.result_dir = result_dir
        self.attr_loader = self.build_attr_loader(dataloader)

        self.model.eval()
        self.model.to(self.device)

    def eval(self, attr_method, absolute):
        eval_result_list = []
        print(f"Attribution method: {attr_method}, absolute: {absolute}")
        for idx_batch, data_batch in enumerate(
            pbar := tqdm(self.attr_loader)
        ):  # batch size set to 1
            idx, x, y = data_batch
            x = x.to(self.device)
            if attr_method == "random_baseline":
                attr_x = np.random.randn(*x.shape)
            else:
                attr_x = get_attribution(self.model, attr_method, x, absolute=absolute)
                attr_x = attr_x.detach().cpu().numpy()

            x = x.detach().cpu().squeeze().numpy()
            y = int(y.detach().cpu().squeeze().numpy())
            y_raw = self.attr_loader.dataset.y_raw[idx.item()]

            loc, pnt, deg = evaluate_attribution(x, self.model, attr_x, y, y_raw)

            sample_eval_result = {
                "y": y,
                "loc": loc,
                "pnt": pnt,
                "deg": deg,
            }

            eval_result_list.append(sample_eval_result)

        eval_result_dir = os.path.join(
            self.result_dir, f"thres_{self.prob_threshold}_{attr_method}_abs_{absolute}"
        )

        os.makedirs(eval_result_dir, exist_ok=True)

        """
        Localization
        """
        iou_all = [s["loc"]["iou"] for s in eval_result_list]
        iou = {"mean": np.mean(iou_all), "std": np.std(iou_all)}

        """
        Pointing game
        """
        pnt_acc = np.mean([s["pnt"]["correct"] for s in eval_result_list])

        evaluation_results = {
            "n_samples": len(eval_result_list),
            "loc": {"iou": iou},
            "pnt": pnt_acc,
        }

        for deg_method in ["mean", "linear", "gaussian_plus"]:
            """
            Degradation
            """
            true_labels = np.array([s["y"] for s in eval_result_list])
            LeRFs = np.array([s["deg"][deg_method]["LeRF"] for s in eval_result_list])
            MoRFs = np.array([s["deg"][deg_method]["MoRF"] for s in eval_result_list])

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

            plt.figure(figsize=(7, 7))
            plt.title(
                f"M: {attr_method}, replace: {deg_method}, Area: {area:.4f}, N: {len(eval_result_list)}, thres: {self.prob_threshold}, is_abs: {absolute}"
            )
            plt.plot(LeRF, label="LeRF")
            plt.plot(MoRF, label="MoRF")
            plt.legend()
            plt.savefig(
                f"{eval_result_dir}/attr_eval_curve_{deg_method}.png",
                bbox_inches="tight",
            )
            plt.close()

        with open(f"{eval_result_dir}/eval_result.json", "w") as f:
            json.dump(evaluation_results, f, indent=4)

    @torch.no_grad()
    def build_attr_loader(self, dataloader):
        """
        Build dataloader for evaluating attribution methods (samples with correct prediction with high prob.)
        """
        attr_x = []
        attr_y = []
        attr_y_raw = []
        attr_prob = []

        for idx_batch, data_batch in enumerate(pbar := tqdm(dataloader)):
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
                    attr_y_raw.append(dataloader.dataset.y_raw[idx[i]])
                    attr_prob.append(prob[label])

        attr_dataset = ECG_Dataset(
            np.array(attr_x), np.array(attr_y), attr_y_raw, attr_prob
        )
        return DataLoader(attr_dataset, pin_memory=True, batch_size=1, shuffle=False)
