import pickle
import gzip
import os

import numpy as np
import matplotlib.pyplot as plt

from src.utils import LABEL_MAPPING
import matplotlib.offsetbox as offsetbox


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

ATTR_FIGSIZE = (40, 20)
ECG_COLOR = "darkblue"
ECG_LW = 3
ATTR_COLOR = "crimson"
ATTR_ALPHA = 0.55
ATTR_LW = 4

# attr_list = ["random_baseline", "saliency", "input_gradient", "guided_backprop", "integrated_gradients", "deep_lift", "deep_shap", "lrp", "lime", "kernel_shap", "gradcam", "guided_gradcam"]

def main():
    DATASET_NAME = "mitdb" # mitdb, svdb, incartdb만 가능
    
    attr_methods = [
        {"method": "guided_gradcam", "absolute": True},
        {"method": "guided_backprop", "absolute": True},
        {"method": "gradcam", "absolute": False},
        {"method": "lime", "absolute": False},
        {"method": "deep_shap", "absolute": False},
        {"method": "integrated_gradients", "absolute": True},
    ]
    
    result_dir = "./figures_plot_attribution_mitdb"
    os.makedirs(result_dir, exist_ok=True)
    
    for sample_idx in range(1079):
    # for sample_idx in range(32, 43):
        print(f"Processing sample {sample_idx}...")
        fig, axs = plt.subplots(3, 2, figsize=ATTR_FIGSIZE)
        for plot_idx, attr_dict in enumerate(attr_methods):
            attr_method = attr_dict["method"]
            attr_absolute = attr_dict["absolute"]
            attr_dir = f"results_for_paper/results_attribution/mitdb_resnet18_7_bs32_lr5e-2_wd1e-4_ep20_seed1/{attr_method}"

            # load eval_attr_data & feature attribution
            eval_attr_data = pickle.load(gzip.GzipFile(f"{attr_dir}/eval_attr_data.pkl", "rb"))
            attr_list = pickle.load(gzip.GzipFile(f"{attr_dir}/attr_list.pkl", "rb"))
            
            x, y, beat_spans, prob = (
                eval_attr_data["x"][sample_idx],
                eval_attr_data["y"][sample_idx],
                eval_attr_data["beat_spans"][sample_idx],
                eval_attr_data["prob"][sample_idx],
            )
            attr_x = attr_list[sample_idx]
            
            if attr_absolute:
                attr_x = np.absolute(attr_x)
            
            ax1 = axs[plot_idx%3][plot_idx//3]
            label_index = LABEL_MAPPING[DATASET_NAME]["LABEL_INDEX"]
            ax2 = ax1.twinx()

            # ECG
            ecg_yrange = get_plot_range(np.min(x), np.max(x), 1.55)
            ax1.set_ylim(*ecg_yrange)
            ax1.plot(x.squeeze(), c=ECG_COLOR, linewidth=ECG_LW)
            # ax1.set_ylabel("ECG signal", color=ECG_COLOR)
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)

            # Attribution
            max_abs_attr = np.max(np.abs(attr_x))
            attr_yrange = (-max_abs_attr * 1.55, max_abs_attr * 1.55)
            ax2.set_ylim(*attr_yrange)
            ax2.plot(attr_x.squeeze(), c=ATTR_COLOR, alpha=ATTR_ALPHA, linewidth=ATTR_LW)
            # ax2.set_ylabel("Attribution value", color=ATTR_COLOR)
            ax2.get_xaxis().set_visible(False)
            ax2.get_yaxis().set_visible(False)

            for class_idx, class_span in beat_spans.items():
                for span in class_span:
                    if class_idx == 1:
                        ax2.fill_between(np.arange(*span), attr_yrange[0], attr_yrange[1], color=(221/256,238/256,254/256))
                    elif class_idx == 2:
                        ax2.fill_between(np.arange(*span), attr_yrange[0], attr_yrange[1], color=(233/256,249/256,220/256))
                    ax1.axvline(span[0], alpha=0.5, c="grey", linestyle="--")
            
            text_label = ATTR_STR_DICT[attr_method]
            if attr_absolute:
                text_label += ABS_SIGN
 
            ob = offsetbox.AnchoredText(text_label, loc="upper left", prop=dict(size=32))
            ob.patch.set(boxstyle='round, pad=0, rounding_size=0.2', facecolor='wheat', alpha=0.7)
            ax1.add_artist(ob)

            label = label_index[y]

            ax1.set_zorder(ax2.get_zorder() + 1)
            ax1.set_frame_on(False)
            
            ax1.margins(x=0)
            ax2.margins(x=0)
        
        save_filename = f"{result_dir}/{label}_{prob:.3f}_{sample_idx}.png"

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.03)
        plt.savefig(save_filename)
        plt.close()

def get_plot_range(min_value, max_value, coff=1):
    baseline_value = (min_value + max_value) / 2
    amplitude = max_value - baseline_value
    plot_range = (baseline_value - amplitude * coff, baseline_value + amplitude * coff)
    return plot_range
    
if __name__ == "__main__":
    main()