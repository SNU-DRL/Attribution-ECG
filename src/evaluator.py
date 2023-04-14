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

        return np.mean(score_list), np.std(score_list)

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
        y_list, lerf_probs_list, morf_probs_list = [], [], []

        for idx_batch, data_batch in enumerate(
            pbar := tqdm(self.dataloader)
        ):  # batch size set to 1
            idx, x, y = data_batch

            x = x.detach().cpu().squeeze().numpy()
            attr_x = np.squeeze(attr_list[idx])
            y = int(y.detach().cpu().squeeze().numpy())

            lerf_probs, morf_probs = degradation_score(
                attr_x, y, x, self.model, self.device, deg_method
            )

            y_list.append(y)
            lerf_probs_list.append(lerf_probs)
            morf_probs_list.append(morf_probs)

        true_labels, LeRFs, MoRFs = (
            np.array(y_list),
            np.array(lerf_probs_list),
            np.array(morf_probs_list),
        )

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

            normalized_LeRF = (LeRF_to_normalize - LeRF_last) / (LeRF_init - LeRF_last)
            normalized_MoRF = (MoRF_to_normalize - MoRF_last) / (MoRF_init - MoRF_last)

            normalized_LeRFs[label_idx] = normalized_LeRF
            normalized_MoRFs[label_idx] = normalized_MoRF

        LeRF = np.mean(normalized_LeRFs, axis=0)
        MoRF = np.mean(normalized_MoRFs, axis=0)
        area = np.sum(LeRF - MoRF) / 128 # Set this value to param

        plt.figure(figsize=(7, 7))
        plt.title(
            f"Replace: {deg_method}, Area: {area:.4f}, N: {len(attr_list)}"
        )
        plt.plot(LeRF, label="LeRF")
        plt.plot(MoRF, label="MoRF")
        plt.legend()
        plt.savefig(
            f"{self.result_dir}/deg_curve_{deg_method}.png",
            bbox_inches="tight",
        )
        plt.close()

        return area
