import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.attribution import (Attribution, degradation_score,
                             localization_score, pointing_game)
from src.utils import get_beat_spans, plot_attribution


class Evaluator:
    def __init__(self, model, data_dict, device, result_dir):
        self.model = model
        self.device = device
        self.result_dir = result_dir
        self.data_dict = data_dict

        self.model.eval()
        self.model.to(self.device)

    def compute_attribution(self, attr_method, absolute=False, n_samples=200):
        """
        Compute feature attribution of each data by using given attribution method

        Args:
            attr_method (str): attribution method
            absolute (bool): use absolute value (default=False)
        """
        print(
            f"Calculating feature attribution - attribution method: {attr_method}, absolute: {absolute}"
        )

        attribution = Attribution(self.model, attr_method, absolute, n_samples)

        attr_list = []
        for idx in tqdm(range(self.data_dict["length"])):
            x, y = self.data_dict["x"][idx], int(self.data_dict["y"][idx])
            x = torch.as_tensor(x, device=self.device).unsqueeze(0)
            attr_x = attribution.apply(x, y)
            attr_list.append(attr_x)

        return attr_list

    def visualize(self, attr_list):
        vis_dir = f"{self.result_dir}/vis"
        os.makedirs(vis_dir, exist_ok=True)
        for idx in tqdm(range(self.data_dict["length"])):
            x, y, y_raw, prob = self.data_dict["x"][idx], int(self.data_dict["y"][idx]), self.data_dict["y_raw"][idx], self.data_dict["prob"][idx]
            attr_x = attr_list[idx]
            vis_path = f"{vis_dir}/label{y}_prob{prob:.6f}_id{idx}.png"
            plot_attribution(x, y, y_raw, prob, attr_x, vis_path)

    def get_localization_score(self, attr_list):
        print("Calculating localization score...")
        loc_score_list = []

        for idx in tqdm(range(self.data_dict["length"])):
            y, y_raw = int(self.data_dict["y"][idx]), self.data_dict["y_raw"][idx]
            attr_x = attr_list[idx].squeeze()
            beat_spans = get_beat_spans(y_raw, len(attr_x))

            loc_score = localization_score(attr_x, y, beat_spans)
            loc_score_list.append(loc_score)

        return np.mean(loc_score_list), np.std(loc_score_list)

    def get_pointing_game_accuracy(self, attr_list):
        print("Calculating pointing game accuracy...")
        pnt_result_list = []

        for idx in tqdm(range(self.data_dict["length"])):
            y, y_raw = int(self.data_dict["y"][idx]), self.data_dict["y_raw"][idx]
            attr_x = attr_list[idx].squeeze()
            beat_spans = get_beat_spans(y_raw, len(attr_x))

            is_correct = pointing_game(attr_x, y, beat_spans)
            pnt_result_list.append(is_correct)

        return np.mean(pnt_result_list)

    def get_degradation_score(self, attr_list, perturbation, window_size):
        print("Calculating degradation score...")
        y_list, LeRF_probs, MoRF_probs = [], [], []

        for idx in tqdm(range(self.data_dict["length"])):
            x, y = self.data_dict["x"][idx].squeeze(), int(self.data_dict["y"][idx])
            attr_x = attr_list[idx].squeeze()

            LeRF_prob, MoRF_prob = degradation_score(
                self.model, x, attr_x, y, self.device, perturbation, window_size
            )

            y_list.append(y)
            LeRF_probs.append(LeRF_prob)
            MoRF_probs.append(MoRF_prob)

        LeRFs, MoRFs = np.array(LeRF_probs), np.array(MoRF_probs)

        LeRFs_normalized = (LeRFs - LeRFs[:, -1].mean()) / (
            LeRFs[:, 0].mean() - LeRFs[:, -1].mean()
        )
        MoRFs_normalized = (MoRFs - MoRFs[:, -1].mean()) / (
            MoRFs[:, 0].mean() - MoRFs[:, -1].mean()
        )

        LeRF = np.mean(LeRFs_normalized, axis=0)
        MoRF = np.mean(MoRFs_normalized, axis=0)
        area = np.sum(LeRF - MoRF) / (LeRF.shape[0] - 1)

        self._plot_deg_curve(perturbation, LeRF, MoRF, area, len(attr_list))

        return area

    def _plot_deg_curve(self, perturbation, LeRF, MoRF, area, num_samples):
        plt.figure(figsize=(7, 7))
        plt.title(f"Perturbation: {perturbation}, Area: {area:.4f}, N: {num_samples}")
        plt.plot(LeRF, label="LeRF")
        plt.plot(MoRF, label="MoRF")
        plt.legend()
        plt.savefig(
            f"{self.result_dir}/deg_curve_{perturbation}.png",
            bbox_inches="tight",
        )
        plt.close()
