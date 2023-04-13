import json
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.attribution import (apply_attr_method, degradation_score,
                             localization_score, pointing_game)
from src.utils import get_boundaries_by_label


class Evaluator:
    def __init__(self, model, dataloader, device, result_dir):
        self.model = model
        self.device = device
        self.result_dir = result_dir
        self.dataloader = dataloader

        self.model.eval()
        self.model.to(self.device)

    def compute_attribution(self, attr_method, absolute):
        print(f"Attribution method: {attr_method}, absolute: {absolute}")
        attr_list = []
        for idx_batch, data_batch in enumerate(
            pbar := tqdm(self.dataloader)
        ):  # batch size set to 1
            idx, x, y = data_batch
            x = x.to(self.device)
            if attr_method == "random_baseline":
                attr_x = np.random.randn(*x.shape)
            else:
                attr_x = apply_attr_method(
                    self.model, attr_method, x, absolute=absolute
                )
                attr_x = attr_x.detach().cpu().numpy()

            attr_list.append(attr_x)

        return attr_list

    def get_localization_score(self, attr_list):
        score_list = []
        for idx_batch, data_batch in enumerate(
            pbar := tqdm(self.dataloader)
        ):  # batch size set to 1
            idx, x, y = data_batch

            x = x.detach().cpu().squeeze().numpy()
            attr_x = np.squeeze(attr_list[idx])
            y = int(y.detach().cpu().squeeze().numpy())
            y_raw = self.dataloader.dataset.y_raw[idx.item()]
            boundaries_per_label = get_boundaries_by_label(y_raw)

            score = localization_score(attr_x, y, boundaries_per_label)
            score_list.append(score)
        return score_list

    def get_pointing_game_score(self, attr_list):
        pointing_game_results = []
        for idx_batch, data_batch in enumerate(
            pbar := tqdm(self.dataloader)
        ):  # batch size set to 1
            idx, x, y = data_batch

            x = x.detach().cpu().squeeze().numpy()
            attr_x = np.squeeze(attr_list[idx])
            y = int(y.detach().cpu().squeeze().numpy())
            y_raw = self.dataloader.dataset.y_raw[idx.item()]
            boundaries_per_label = get_boundaries_by_label(y_raw)

            correct = pointing_game(attr_x, y, boundaries_per_label)
            pointing_game_results.append(correct)

        return np.mean(pointing_game_results)

    def get_degradation_score(self, attr_list, deg_method):
        score_list_lerf, score_list_morf = [], []

        for idx_batch, data_batch in enumerate(
            pbar := tqdm(self.dataloader)
        ):  # batch size set to 1
            idx, x, y = data_batch

            x = x.detach().cpu().squeeze().numpy()
            attr_x = np.squeeze(attr_list[idx])
            y = int(y.detach().cpu().squeeze().numpy())

            lerf_probs, morf_probs = degradation_score(
                attr_x, y, x, self.model, deg_method
            )
            score_list_lerf.append(lerf_probs)
            score_list_morf.append(morf_probs)

        return score_list_lerf, score_list_morf

    def eval(self, attr_method, absolute):
        eval_result_list = []
        print(f"Attribution method: {attr_method}, absolute: {absolute}")
        for idx_batch, data_batch in enumerate(
            pbar := tqdm(self.dataloader)
        ):  # batch size set to 1
            idx, x, y = data_batch
            x = x.to(self.device)
            if attr_method == "random_baseline":
                attr_x = np.random.randn(*x.shape)
            else:
                attr_x = apply_attr_method(
                    self.model, attr_method, x, absolute=absolute
                )
                attr_x = attr_x.detach().cpu().numpy()

            x = x.detach().cpu().squeeze().numpy()
            y = int(y.detach().cpu().squeeze().numpy())
            y_raw = self.dataloader.dataset.y_raw[idx.item()]

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
