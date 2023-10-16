import os
import sys

import pandas as pd

if len(sys.argv) != 3:
    print(
        "Usage (needs two arguments): python summarize_final_results_region_perturbation.py results_evaluation/icentia11k_resnet18_7_bs32_lr1e-3_wd1e-4_ep20 16"
    )
    sys.exit()
    
RESULTS_BASE_PATH = sys.argv[1]
PATCH_SIZE = int(sys.argv[2])
attr_methods = [
    "random_baseline", "saliency", "input_gradient", "guided_backprop", "integrated_gradients", "deep_lift", "deep_shap", "lrp", "lime", "kernel_shap", "gradcam", "guided_gradcam"
]
result_files = []
for attr_method in attr_methods:
    result_files.append(f"{attr_method}_results.csv")
    result_files.append(f"{attr_method}_results_absolute.csv")
    
metrics = [f"region_perturbation_patch_size_{PATCH_SIZE}_order_lerf", f"region_perturbation_patch_size_{PATCH_SIZE}_order_morf"]
df_columns = []
for metric in metrics:
    df_columns.append(f"{metric}_mean")
    df_columns.append(f"{metric}_stddev")
    df_columns.append(f"{metric}_rank")

total_result_df = pd.DataFrame(columns=df_columns)

for result_file in result_files:
    metric_name = result_file.split(".")[0]
    result_df = pd.read_csv(os.path.join(RESULTS_BASE_PATH, result_file), index_col=0)
    new_result_dict = {}
    for column in result_df.T.columns:
        new_result_dict[f"{column}_mean"] = result_df.T.loc["mean"][column]
        new_result_dict[f"{column}_stddev"] = result_df.T.loc["stddev"][column]
    total_result_df.loc[metric_name] = new_result_dict

total_result_df[f"{metrics[0]}_rank"] = total_result_df[f"{metrics[0]}_mean"].rank(ascending=True).astype(int)
total_result_df[f"{metrics[1]}_rank"] = total_result_df[f"{metrics[1]}_mean"].rank(ascending=False).astype(int)

total_result_df.to_csv(os.path.join(RESULTS_BASE_PATH, "_final_results.csv"))