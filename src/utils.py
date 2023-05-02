import matplotlib.pyplot as plt
import numpy as np

ICENTIA_BEAT_INDEX = {
    0: "undefined",  # Undefined
    1: "normal",  # Normal
    2: "pac",  # ESSV (PAC)
    3: "aberrated",  # Aberrated
    4: "pvc",  # ESV (PVC)
}

ICENTIA_LABEL_MAPPING = {
    "normal": 0,
    "pac": 1,
    "pvc": 2,
}

ICENTIA_LABEL_MAPPING_REVERSE = {y: x for x, y in ICENTIA_LABEL_MAPPING.items()}


def preprocess(data):
    """
    Standardization
    data.shape: (# samples, frame_size)
    """
    m = np.expand_dims(data.mean(-1), -1)
    s = np.expand_dims(data.std(-1), -1)
    return (data - m) / (s + 1e-6)


def get_beat_spans(y_raw, len_x=2049):
    beats = extract_beats(y_raw)
    beats = dict(sorted(beats.items()))

    r_peaks = np.array(list(beats.keys()))
    beat_boundaries = (r_peaks[1:] + r_peaks[:-1]) // 2
    beat_onsets = np.insert(beat_boundaries, 0, 0)  # inclusive
    beat_offsets = np.append(beat_boundaries, len_x)  # exclusive
    beat_spans = {0: [], 1: [], 2: []}

    for beat, onset, offset in zip(beats.values(), beat_onsets, beat_offsets):
        if beat not in ["normal", "pac", "pvc"]:
            continue
        beat_spans[ICENTIA_LABEL_MAPPING[beat]].append(
            (onset, offset)
        )  # span: [onset, offset-1]

    return beat_spans


def extract_beats(y_raw):
    label_dict = {}
    for i, indices in enumerate(y_raw):
        for j in indices:
            label_dict[j] = ICENTIA_BEAT_INDEX[i]
    return label_dict


ATTR_FIGSIZE = (25, 10)
YLIM = (-15, 15)
ECG_COLOR = "darkblue"
ECG_LW = 2
ATTR_COLOR = "darkgreen"
ATTR_ALPHA = 0.5
ATTR_LW = 2


def plot_attribution(x, y, y_raw, prob, attr_x, path):
    beat_spans = get_beat_spans(y_raw)
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
                ICENTIA_LABEL_MAPPING_REVERSE[class_idx],
                fontsize=15,
                horizontalalignment="center",
            )

    label = ICENTIA_LABEL_MAPPING_REVERSE[y]
    plt.title(f"Label: {label}, Prob: {prob:.6f}")

    # Attribution
    ax2.plot(attr_x.squeeze(), c=ATTR_COLOR, alpha=ATTR_ALPHA, linewidth=ATTR_LW)
    ax2.set_ylabel("Attribution value", color=ATTR_COLOR)

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
