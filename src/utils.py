import os
import random

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

LABEL_MAPPING = {
    "icentia11k": {
        "BEAT_INDEX": {
            0: "undefined",  # Undefined
            1: "normal",  # Normal
            2: "pac",  # ESSV (PAC)
            3: "aberrated",  # Aberrated
            4: "pvc",  # ESV (PVC)
        },
        "LABEL_INDEX": {
            0: "normal",
            1: "pac",
            2: "pvc",
        },
        "LABEL_INDEX_REVERSE": {
            "normal": 0,
            "pac": 1,
            "pvc": 2,
        },
    },
    "mit-bih": {
        "BEAT_INDEX": {0: "N", 1: "SVEB", 2: "VEB", 3: "F", 4: "Q"},
        "LABEL_INDEX": {
            0: "N",
            1: "SVEB",
            2: "VEB",
            3: "F",
        },
        "LABEL_INDEX_REVERSE": {"N": 0, "SVEB": 1, "VEB": 2, "F": 3},
    },
    "st-petersburg": {
        "BEAT_INDEX": {0: "N", 1: "SVEB", 2: "VEB", 3: "F", 4: "Q"},
        "LABEL_INDEX": {
            0: "N",
            1: "SVEB",
            2: "VEB",
            3: "F",
        },
        "LABEL_INDEX_REVERSE": {"N": 0, "SVEB": 1, "VEB": 2, "F": 3},
    },
    "mit-bih_svdb": {
        "BEAT_INDEX": {0: "N", 1: "SVEB", 2: "VEB", 3: "F", 4: "Q"},
        "LABEL_INDEX": {
            0: "N",
            1: "SVEB",
            2: "VEB",
            3: "F",
        },
        "LABEL_INDEX_REVERSE": {"N": 0, "SVEB": 1, "VEB": 2, "F": 3},
    },
}


def preprocess(data):
    """
    Standardization
    data.shape: (# samples, frame_size)
    """
    m = np.expand_dims(data.mean(-1), -1)
    s = np.expand_dims(data.std(-1), -1)
    return (data - m) / (s + 1e-6)


def get_beat_spans(y_raw, len_x, dataset):
    label_index_reverse = LABEL_MAPPING[dataset]["LABEL_INDEX_REVERSE"]
    beats = extract_beats(y_raw, dataset)
    beats = dict(sorted(beats.items()))

    r_peaks = np.array(list(beats.keys()))
    beat_boundaries = (r_peaks[1:] + r_peaks[:-1]) // 2
    beat_onsets = np.insert(beat_boundaries, 0, 0)  # inclusive
    beat_offsets = np.append(beat_boundaries, len_x)  # exclusive
    beat_spans = {idx: [] for idx in label_index_reverse.values()}

    for beat, onset, offset in zip(beats.values(), beat_onsets, beat_offsets):
        if beat not in label_index_reverse.keys():
            continue
        beat_spans[label_index_reverse[beat]].append(
            (onset, offset)
        )  # span: [onset, offset-1]

    return beat_spans


def extract_beats(y_raw, dataset):
    beat_index = LABEL_MAPPING[dataset]["BEAT_INDEX"]
    label_dict = {}
    for i, indices in enumerate(y_raw):
        for j in indices:
            label_dict[j] = beat_index[i]
    return label_dict


def visualize(dataset, data_dict, attr_list, absolute, vis_dir, n_samples_vis):
    os.makedirs(vis_dir, exist_ok=True)
    sample_indices = random.sample(range(data_dict["length"]), n_samples_vis)

    for idx in tqdm(sample_indices):
        x, y, beat_spans, prob = data_dict["x"][idx], int(data_dict["y"][idx]), data_dict["beat_spans"][idx], data_dict["prob"][idx]
        attr_x = attr_list[idx]
        if absolute:
            attr_x = np.absolute(attr_x)
        vis_path = f"{vis_dir}/label{y}_prob{prob:.6f}_id{idx}.png"
        plot_attribution(x, y, beat_spans, prob, attr_x, dataset, vis_path)


ATTR_FIGSIZE = (25, 10)
YLIM = (-15, 15)
ECG_COLOR = "darkblue"
ECG_LW = 2
ATTR_COLOR = "darkgreen"
ATTR_ALPHA = 0.5
ATTR_LW = 2


def plot_attribution(x, y, beat_spans, prob, attr_x, dataset, path):
    label_index = LABEL_MAPPING[dataset]["LABEL_INDEX"]
    fig, ax1 = plt.subplots(figsize=ATTR_FIGSIZE)
    ax2 = ax1.twinx()

    # ECG
    ax1.set_ylim(*YLIM)
    ax1.plot(x.squeeze(), c=ECG_COLOR, linewidth=ECG_LW)
    ax1.set_ylabel("ECG signal", color=ECG_COLOR)

    for class_idx, class_span in beat_spans.items():
        for span in class_span:
            ax1.axvline(span[0], alpha=0.5, c="grey", linestyle="--")
            ax1.text(
                np.mean(span),
                -18,
                label_index[class_idx],
                fontsize=15,
                horizontalalignment="center",
            )

    label = label_index[y]
    plt.title(f"Label: {label}, Prob: {prob:.6f}")

    # Attribution
    ax2.plot(attr_x.squeeze(), c=ATTR_COLOR, alpha=ATTR_ALPHA, linewidth=ATTR_LW)
    ax2.set_ylabel("Attribution value", color=ATTR_COLOR)

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
