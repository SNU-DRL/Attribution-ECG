import gzip
import os
import pickle
import random

import matplotlib.offsetbox as offsetbox
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
import pandas as pd

# plt.rcParams['font.family'] = 'Arial'

ATTR_STR_DICT = {
    "random_baseline": "Random (baseline)",
    "saliency": "Saliency",
    "input_gradient": "Input × Gradient",
    "guided_backprop": "Guided Backprop",
    "integrated_gradients": "Integrated Gradients",
    "deep_lift": "DeepLIFT",
    "deep_shap": "DeepSHAP",
    "lrp": "LRP",
    "lime": "LIME",
    "kernel_shap": "KernelSHAP",
    "gradcam": "Grad-CAM",
    "guided_gradcam": "Guided Grad-CAM",
}

ABS_SIGN = "°"
NUM_LEADS = 12
LEAD_DICT = {
    0: "Lead I",
    1: "Lead II",
    2: "Lead III",
    3: "Lead aVR",
    4: "Lead aVL",
    5: "Lead aVF",
    6: "Lead V1",
    7: "Lead V2",
    8: "Lead V3",
    9: "Lead V4",
    10: "Lead V5",
    11: "Lead V6",
}
FEATURE_MASK_SIZE = 32
ATTR_FIGSIZE = (40, 6 * (NUM_LEADS/2))
ECG_COLOR = "darkblue"
ECG_LW = 3
ATTR_COLOR = "crimson"
ATTR_ALPHA = 0.55
ATTR_LW = 4

NUM_MAX_SAMPLES = 10
METHOD_NAME_HIDDEN = True
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

label_mapping_df = pd.read_csv("ptb-xl/label_selection/label_mapping.csv", index_col=0)
attr_method_1 = "guided_gradcam"
attr_absolute_1 = True
attr_method_2 = "lime"
attr_absolute_2 = False

result_dir = f"./results_ptbxl/figures_plot_attribution_ptb-xl_12leads_pdf_survey_v2"
if not METHOD_NAME_HIDDEN:
    result_dir += "_method_name"
DATA_PATH_1 = f"results_ptbxl/results_attribution_selected/ptbxl_12leads_resnet18_7_bs32_lr1e-4_wd1e-4_ep20_seed0/{attr_method_1}"
DATA_PATH_2 = f"results_ptbxl/results_attribution_selected/ptbxl_12leads_resnet18_7_bs32_lr1e-4_wd1e-4_ep20_seed0/{attr_method_2}"

def main():
    class_indices = os.listdir(DATA_PATH_1)

    os.makedirs(result_dir, exist_ok=True)
    
    for class_idx in class_indices:
        class_idx = int(class_idx)
        class_name = label_mapping_df.iloc[class_idx]["Dx"]
        attr_dir_1 = f"{DATA_PATH_1}/{class_idx}"
        attr_dir_2 = f"{DATA_PATH_2}/{class_idx}"
        print(f"Processing... {class_name}({class_idx})")

        # load eval_attr_data & feature attribution
        try:
            eval_attr_data = pickle.load(gzip.GzipFile(f"{attr_dir_1}/eval_attr_data.pkl", "rb"))
            attr_list_1 = pickle.load(gzip.GzipFile(f"{attr_dir_1}/attr_list.pkl", "rb"))
            attr_list_2 = pickle.load(gzip.GzipFile(f"{attr_dir_2}/attr_list.pkl", "rb"))
        except FileNotFoundError:
            print(f"Skip class {class_name}")
            continue
        
        num_samples = len(eval_attr_data["id"])
        samples_list = [{"idx": idx, "method": attr_method_1, "abs": attr_absolute_1} for idx in range(num_samples)]
        samples_list.extend([{"idx": idx, "method": attr_method_2, "abs": attr_absolute_2} for idx in range(num_samples)])
        random.shuffle(samples_list)
        
        id_list = []
        method_list = []
        
        for question_num, sample_info_dict in enumerate(samples_list):
            sample_idx = sample_info_dict["idx"]
            attr_method = sample_info_dict["method"]
            attr_absolute = sample_info_dict["abs"]
            
            sample_id, x, y, beat_spans, prob = (
                eval_attr_data["id"][sample_idx],
                eval_attr_data["x"][sample_idx],
                eval_attr_data["y"][sample_idx],
                eval_attr_data["beat_spans"][sample_idx],
                eval_attr_data["prob"][sample_idx],
            )
            
            id_list.append(sample_id)
            method_list.append(attr_method)
            
            if attr_method == attr_method_1:
                attr_list = attr_list_1
            else:
                attr_list = attr_list_2

            attr_x = attr_list[sample_idx]
            
            x = x.squeeze()
            attr_x = attr_x.squeeze()

            if attr_absolute:
                attr_x = np.absolute(attr_x)
            if attr_method == "guided_gradcam":
                attr_x = chunk_attribution(attr_x, chunk_size=FEATURE_MASK_SIZE)
            
            num_leads = x.shape[0]
            len_x = x.shape[1]
            
            text_label = ATTR_STR_DICT[attr_method]
            if attr_absolute:
                text_label += ABS_SIGN

            label_string = r""
            for idx, row in label_mapping_df.loc[y==1].iterrows():
                if idx == class_idx:
                    label_string += r"$\bf{%s}$" % '\ '.join(row["Dx"].split())
                else:
                    label_string += row["Dx"]
                label_string += r", "
            label_string = label_string[:-2]
            
            fig, axs = plt.subplots(num_leads//2, 2, figsize=ATTR_FIGSIZE, sharex=True, gridspec_kw=dict(left=0.02, right=0.98, top=0.9, bottom=0.05, wspace=0.05, hspace=0.05))
            ecg_yrange = get_plot_range(np.min(x), np.max(x), 1.3)
            fig.suptitle(f"Question {question_num+1}.", fontsize=40)
            
            if METHOD_NAME_HIDDEN:
                fig.text(x=0.5, y=0.92, s=f"{label_string}", fontsize=28, ha="center")
            else:
                fig.text(x=0.5, y=0.92, s=f"ID: {sample_id}, Method: {attr_method}\nProb: {prob:.3f}\n{label_string}", fontsize=28, ha="center")
                
            for lead_idx in range(num_leads):
                ax = axs[lead_idx%6, lead_idx//6]
                lead_x = x[lead_idx]
                lead_attr_x = attr_x[lead_idx]
                
                ##### Attribution visualization 1) bar plots
                ax2 = ax.twinx()
                
                # ECG
                ax.plot(lead_x, c=ECG_COLOR, linewidth=ECG_LW)

                # Attribution
                max_abs_attr = np.max(np.abs(attr_x))
                attr_yrange = (-max_abs_attr * 1.2, max_abs_attr * 1.2)
                ax.set_frame_on(False)
                ax2.set_ylim(*attr_yrange)
                ax2.plot(lead_attr_x, c=ATTR_COLOR, alpha=ATTR_ALPHA, linewidth=ATTR_LW)
                ax2.get_yaxis().set_visible(False)
                ax.set_zorder(ax2.get_zorder() + 1)
                ax2.margins(x=0)
                #####
                
                ax.set_ylim(*ecg_yrange)
                ax.set_yticks([])
                ax.set_ylabel(LEAD_DICT[lead_idx], fontdict={"size": 28})
                ax.margins(x=0)
                ax.grid(which="major", axis="x", linestyle="--")

                if lead_idx % 6 == 5:
                    ax.set_xticks(ticks=[500*i for i in range(11)], labels=[i for i in range(11)], fontdict={"size": 24})
                    ax.set_xlabel("Time (seconds)", fontdict={"size": 28})
            # fig.tight_layout()
        save_image(f"{result_dir}/{class_idx}.{label_mapping_df.loc[class_idx]['Dx']}.pdf")
        plt.close("all")
        
        if not METHOD_NAME_HIDDEN:
            data_dict = {"sample_id": id_list, "attribution_method": method_list}
            df_method = pd.DataFrame.from_dict(data=data_dict)
            df_method.index = range(1, len(method_list)+1)
            df_method.to_csv(os.path.join(result_dir, f"{class_idx}.{label_mapping_df.loc[class_idx]['Dx']}.csv"))

def get_plot_range(min_value, max_value, coff=1):
    baseline_value = (min_value + max_value) / 2
    amplitude = max_value - baseline_value
    plot_range = (baseline_value - amplitude * coff, baseline_value + amplitude * coff)
    return plot_range

# https://www.geeksforgeeks.org/save-multiple-matplotlib-figures-in-single-pdf-file-using-python/
def save_image(filename): 
    
    # PdfPages is a wrapper around pdf  
    # file so there is no clash and 
    # create files with no error. 
    p = PdfPages(filename) 
      
    # get_fignums Return list of existing 
    # figure numbers 
    fig_nums = plt.get_fignums()   
    figs = [plt.figure(n) for n in fig_nums] 
      
    # iterating over the numbers in list 
    for fig in figs:  
        
        # and saving the files 
        fig.savefig(p, format='pdf')  
          
    # close the object 
    p.close()

def chunk_attribution(attr_x, chunk_size=32):
    res = np.zeros_like(attr_x)
    for lead_idx in range(attr_x.shape[0]):
        for idx in range(0, attr_x.shape[1], chunk_size):
            start_idx = idx
            end_idx = min(start_idx + chunk_size, attr_x.shape[1])
            res[lead_idx, start_idx:end_idx] = attr_x[lead_idx, start_idx:end_idx].mean()
    return res

if __name__ == "__main__":
    main()