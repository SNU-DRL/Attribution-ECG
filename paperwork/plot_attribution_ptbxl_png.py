import gzip
import os
import pickle

import matplotlib.offsetbox as offsetbox
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.family'] = 'Arial'


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

NUM_LEADS = 4
LEAD_DICT = {
    0: "Lead I",
    1: "Lead II",
    2: "Lead V1",
    3: "Lead aVF",
}
ATTR_FIGSIZE = (40, 6 * NUM_LEADS)
ECG_COLOR = "darkblue"
ECG_LW = 3
ATTR_COLOR = "crimson"
ATTR_ALPHA = 0.55
ATTR_LW = 4

label_mapping_df = pd.read_csv("ptb-xl/label_selection/label_mapping.csv", index_col=0)
attr_method = "guided_gradcam"
attr_absolute = True

result_dir = f"./results_ptbxl/figures_plot_attribution_ptb-xl_4leads_pdf/{attr_method}"
if attr_absolute == True:
    result_dir += "_absolute"
DATA_PATH = f"results_ptbxl/results_attribution/ptbxl_4leads_resnet18_7_bs32_lr1e-4_wd1e-4_ep20_seed0/{attr_method}"

def main():
    class_indices = os.listdir(DATA_PATH)

    for class_idx in class_indices:
        class_idx = int(class_idx)
        class_name = label_mapping_df.iloc[class_idx]["Dx"]
        attr_dir = f"{DATA_PATH}/{class_idx}"
        print(f"Processing... {class_name}({class_idx})")

        # load eval_attr_data & feature attribution
        try:
            eval_attr_data = pickle.load(gzip.GzipFile(f"{attr_dir}/eval_attr_data.pkl", "rb"))
            attr_list = pickle.load(gzip.GzipFile(f"{attr_dir}/attr_list.pkl", "rb"))
        except FileNotFoundError:
            print(f"Skip class {class_name}")
            continue
    
        save_dir = f"{result_dir}/{class_idx}"
        os.makedirs(save_dir, exist_ok=True)
        
        for sample_idx in range(len(eval_attr_data["id"])):
            sample_id, x, y, beat_spans, prob = (
                eval_attr_data["id"][sample_idx],
                eval_attr_data["x"][sample_idx],
                eval_attr_data["y"][sample_idx],
                eval_attr_data["beat_spans"][sample_idx],
                eval_attr_data["prob"][sample_idx],
            )
            attr_x = attr_list[sample_idx]

            if attr_absolute:
                attr_x = np.absolute(attr_x)
                
            num_leads = x.shape[-2]

            fig, axs = plt.subplots(num_leads, 1, figsize=ATTR_FIGSIZE, sharex=True)

            for lead_idx in range(num_leads):
                ax1 = axs[lead_idx]
                ax2 = ax1.twinx()
                
                # ECG
                ecg_yrange = get_plot_range(np.min(x), np.max(x), 1.55)
                ax1.set_ylim(*ecg_yrange)
                ax1.plot(x.squeeze()[lead_idx], c=ECG_COLOR, linewidth=ECG_LW)
                ax1.set_yticks([])
                ax1.set_ylabel(LEAD_DICT[lead_idx], fontdict={"size": 28})
                
                # Attribution
                max_abs_attr = np.max(np.abs(attr_x))
                attr_yrange = (-max_abs_attr * 1.55, max_abs_attr * 1.55)
                ax2.set_ylim(*attr_yrange)
                ax2.plot(attr_x.squeeze()[lead_idx], c=ATTR_COLOR, alpha=ATTR_ALPHA, linewidth=ATTR_LW)
                ax2.get_yaxis().set_visible(False)
                
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
                
                if lead_idx == 0:
                    ob = offsetbox.AnchoredText(label_string, loc="upper left", prop=dict(size=28))
                    ob.patch.set(boxstyle='round, pad=0, rounding_size=0.2', facecolor='skyblue', alpha=0.7)
                    ax1.add_artist(ob)
                if lead_idx == num_leads-1:
                    ax1.set_xticks(ticks=[500*i for i in range(11)], labels=[i for i in range(11)], fontdict={"size": 24})
                    ax1.set_xlabel("Time (seconds)", fontdict={"size": 28})

                ax1.set_zorder(ax2.get_zorder() + 1)
                ax1.set_frame_on(False)
                ax1.margins(x=0)
                ax2.margins(x=0)

                ax1.grid(which="major", axis="x", linestyle="--")
                
            save_filename = f"{save_dir}/{sample_id}_{prob:.3f}.png"
            
            plt.tight_layout()
            # plt.subplots_adjust(hspace=0.15)
            plt.savefig(save_filename)
            plt.close()

def get_plot_range(min_value, max_value, coff=1):
    baseline_value = (min_value + max_value) / 2
    amplitude = max_value - baseline_value
    plot_range = (baseline_value - amplitude * coff, baseline_value + amplitude * coff)
    return plot_range

if __name__ == "__main__":
    main()