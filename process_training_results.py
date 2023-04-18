"""
Calculate mean and standard deviation of result metrics with the same settings and different seeds
"""
import os
import sys
from glob import glob

import pandas as pd

if len(sys.argv) != 2:
    print(
        "Usage (needs one argument): python process_training_results.py ./results/resnet18_7_bs32_lr1e-4_wd1e-4_ep10"
    )
    sys.exit()


RESULTS_BASE_PATH = sys.argv[1]
result_dirs = glob(f"{RESULTS_BASE_PATH}_seed*")

# aggregate results of the last epoch
result_metrics = pd.read_csv(os.path.join(result_dirs[0], "metrics.csv")).iloc[-1:]

for result_dir in result_dirs[1:]:
    _result_metrics = pd.read_csv(os.path.join(result_dir, "metrics.csv")).iloc[-1:]
    result_metrics = pd.concat([result_metrics, _result_metrics], ignore_index=True)
    
result_metrics = result_metrics.drop(columns=["epoch"])
processed_metrics = pd.concat(
    [result_metrics.mean().to_frame(), result_metrics.std().to_frame()], axis=1
)
processed_metrics.columns = ["mean", "stddev"]
dirname, filename = os.path.split(RESULTS_BASE_PATH)
processed_metrics = processed_metrics.round(4)
processed_metrics.to_csv(os.path.join(dirname, f"{filename}.csv"))
