import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.attribution import (Attribution, degradation_score,
                             localization_score, pointing_game)
from src.utils import plot_attribution

matplotlib.rcParams['font.family'] = ['Arial']
matplotlib.rcParams['font.size'] = 14

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

    def visualize(self, dataset, attr_list):
        vis_dir = f"{self.result_dir}/vis"
        os.makedirs(vis_dir, exist_ok=True)
        for idx in tqdm(range(self.data_dict["length"])):
            x, y, beat_spans, prob = self.data_dict["x"][idx], int(self.data_dict["y"][idx]), self.data_dict["beat_spans"][idx], self.data_dict["prob"][idx]
            attr_x = attr_list[idx]
            vis_path = f"{vis_dir}/label{y}_prob{prob:.6f}_id{idx}.png"
            plot_attribution(x, y, beat_spans, prob, attr_x, dataset, vis_path)

    def get_localization_score(self, attr_list):
        print("Calculating localization score...")
        loc_score_list = []

        for idx in tqdm(range(self.data_dict["length"])):
            y, beat_spans = int(self.data_dict["y"][idx]), self.data_dict["beat_spans"][idx]
            attr_x = attr_list[idx].squeeze()

            loc_score = localization_score(attr_x, y, beat_spans)
            loc_score_list.append(loc_score)

        return np.mean(loc_score_list), np.std(loc_score_list)

    def get_pointing_game_accuracy(self, attr_list):
        print("Calculating pointing game accuracy...")
        pnt_result_list = []

        for idx in tqdm(range(self.data_dict["length"])):
            y, beat_spans = int(self.data_dict["y"][idx]), self.data_dict["beat_spans"][idx]
            attr_x = attr_list[idx].squeeze()

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

        self._plot_deg_curve(perturbation, window_size, LeRF, MoRF, area, len(attr_list))

        return area

    def _plot_deg_curve(self, perturbation, window_size, LeRF, MoRF, score, num_samples):
        fig, ax = plt.subplots(figsize=(10, 10))
        x = np.arange(len(LeRF)) / (len(LeRF)-1) * 100
        ax.plot(x, LeRF, label="LeRF")
        ax.plot(x, MoRF, label="MoRF", linestyle="--")
        ax.set_xlim([-5,105])
        ax.fill_between(x, LeRF, MoRF, color="lightgrey", label="score")
        ax.set_xlabel("degradation of x [%]", fontsize=18)
        ax.set_ylabel("normalized score", fontsize=18)
        ax.grid()
        ax.legend(fontsize=18)
        fig.tight_layout()

        plt.savefig(
            f"{self.result_dir}/deg_curve_{perturbation}_{window_size}.png",
            bbox_inches="tight",
        )
        plt.close()

        # save additional infomation
        with open(f"{self.result_dir}/deg_curve_{perturbation}_{window_size}.txt", "w") as f:
            f.write(f"perturbation,{perturbation}\nwindow size,{window_size}\nscore,{score}\nN,{num_samples}")