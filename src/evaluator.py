import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.attribution import (apply_attr_method, degradation_score,
                             localization_score, pointing_game)
from src.utils import get_boundaries_by_label


class Evaluator:
    def __init__(self, model, data_dict, device, result_dir):
        self.model = model
        self.device = device
        self.result_dir = result_dir
        self.data_dict = data_dict

        self.model.eval()
        self.model.to(self.device)

    def compute_attribution(self, attr_method, absolute):
        print(f"Attribution method: {attr_method}, absolute: {absolute}")
        attr_list = []
        for idx in tqdm(range(self.data_dict["length"])):
            x = self.data_dict["x"][idx]
            x = torch.as_tensor(x, device=self.device).unsqueeze(0)
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

        for idx in tqdm(range(self.data_dict["length"])):
            y, y_raw = self.data_dict["y"][idx], self.data_dict["y_raw"][idx]
            attr_x = attr_list[idx].squeeze()
            boundaries_per_label = get_boundaries_by_label(y_raw)

            score = localization_score(attr_x, y, boundaries_per_label)
            score_list.append(score)

        return np.mean(score_list), np.std(score_list)

    def get_pointing_game_score(self, attr_list):
        pointing_game_results = []

        for idx in tqdm(range(self.data_dict["length"])):
            y, y_raw = self.data_dict["y"][idx], self.data_dict["y_raw"][idx]
            attr_x = attr_list[idx].squeeze()
            boundaries_per_label = get_boundaries_by_label(y_raw)

            correct = pointing_game(attr_x, y, boundaries_per_label)
            pointing_game_results.append(correct)

        return np.mean(pointing_game_results)

    def get_degradation_score(self, attr_list, deg_method, window_size):
        y_list, lerf_probs_list, morf_probs_list = [], [], []

        for idx in tqdm(range(self.data_dict["length"])):
            x, y = self.data_dict["x"][idx].squeeze(), self.data_dict["y"][idx]
            attr_x = attr_list[idx].squeeze()

            lerf_probs, morf_probs = degradation_score(
                attr_x, y, x, self.model, self.device, deg_method, window_size
            )

            y_list.append(y)
            lerf_probs_list.append(lerf_probs)
            morf_probs_list.append(morf_probs)

        LeRFs, MoRFs = np.array(lerf_probs_list), np.array(morf_probs_list)

        LeRFs_normalized = (LeRFs - LeRFs[:, -1].mean()) / (
            LeRFs[:, 0].mean() - LeRFs[:, -1].mean()
        )
        MoRFs_normalized = (MoRFs - MoRFs[:, -1].mean()) / (
            MoRFs[:, 0].mean() - MoRFs[:, -1].mean()
        )

        LeRF = np.mean(LeRFs_normalized, axis=0)
        MoRF = np.mean(MoRFs_normalized, axis=0)
        area = np.sum(LeRF - MoRF) / (LeRF.shape[0] - 1)

        self._plot_deg_curve(deg_method, LeRF, MoRF, area, len(attr_list))

        return area

    def _plot_deg_curve(self, deg_method, LeRF, MoRF, area, num_samples):
        plt.figure(figsize=(7, 7))
        plt.title(f"Replace: {deg_method}, Area: {area:.4f}, N: {num_samples}")
        plt.plot(LeRF, label="LeRF")
        plt.plot(MoRF, label="MoRF")
        plt.legend()
        plt.savefig(
            f"{self.result_dir}/deg_curve_{deg_method}.png",
            bbox_inches="tight",
        )
        plt.close()
