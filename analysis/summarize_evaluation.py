"""
Calculate mean and standard deviation of evaluation result metrics with the same settings and different seeds
"""
import os
import sys
from glob import glob

import pandas as pd

if len(sys.argv) != 2:
    print(
        "Usage (needs one argument): python summarize_evaluation.py results_evaluation/icentia11k_resnet18_7_bs32_lr1e-3_wd1e-4_ep20"
    )
    sys.exit()

RESULTS_BASE_PATH = sys.argv[1]
result_dirs = glob(f"{RESULTS_BASE_PATH}_seed*")
os.makedirs(RESULTS_BASE_PATH, exist_ok=True)

methods = os.listdir(result_dirs[0])

for method in methods:
    for filename in ["results.csv", "results_absolute.csv"]:
        try:
            # aggregate results
            series_list = []
            for result_dir in result_dirs:
                result_df = pd.read_csv(os.path.join(result_dir, method, filename), header=None, index_col=0).T
                series_list.append(result_df)
            
            result_df = pd.concat(series_list, ignore_index=True)
            processed_metrics = pd.concat([result_df.mean().to_frame(), result_df.std().to_frame()], axis=1)

            processed_metrics.columns = ["mean", "stddev"]
            processed_metrics = processed_metrics.round(4)
            processed_metrics.to_csv(os.path.join(RESULTS_BASE_PATH, f"{method}_{filename}"))
        except FileNotFoundError:
            continue
