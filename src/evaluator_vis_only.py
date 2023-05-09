import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import random

from src.attribution import Attribution
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

    def compute_and_visualize_attribution(self, dataset, attr_method, absolute=False, n_samples=200, n_samples_vis=20):
        """
        Compute feature attribution of each data by using given attribution method

        Args:
            attr_method (str): attribution method
            absolute (bool): use absolute value (default=False)
        """
        print(
            f"Calculating feature attribution - attribution method: {attr_method}, absolute: {absolute}"
        )
        vis_dir = f"{self.result_dir}/vis"
        os.makedirs(vis_dir, exist_ok=True)
        attribution = Attribution(self.model, attr_method, absolute, n_samples)
        
        sample_indices = random.sample(range(self.data_dict["length"]), n_samples_vis)
        
        for idx in tqdm(sample_indices):
            x, y, beat_spans, prob = self.data_dict["x"][idx], int(self.data_dict["y"][idx]), self.data_dict["beat_spans"][idx], self.data_dict["prob"][idx]
            x_tensor = torch.as_tensor(x, device=self.device).unsqueeze(0)
            attr_x = attribution.apply(x_tensor, y)
            vis_path = f"{vis_dir}/label{y}_prob{prob:.6f}_id{idx}.png"
            plot_attribution(x, y, beat_spans, prob, attr_x, dataset, vis_path)
