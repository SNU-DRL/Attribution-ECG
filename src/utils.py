import numpy as np


ds_beat_names = {
    0: "undefined",  # Undefined
    1: "normal",  # Normal
    2: "pac",  # ESSV (PAC)
    3: "aberrated",  # Aberrated
    4: "pvc",  # ESV (PVC)
}


def preprocess(data):
    """
    Standardization
    data.shape: (# samples, 1(lead), frame_size), numpy
    """
    m = np.expand_dims(data.mean(-1), -1)
    s = np.expand_dims(data.std(-1), -1)
    return (data - m) / (s + 1e-6)


def get_boundaries_by_label(raw_label):
    flat_raw_label = flatten_raw_label(raw_label)
    flat_raw_label = dict(sorted(flat_raw_label.items()))

    r_peaks = np.array(list(flat_raw_label.keys()))
    beat_boundaries = (r_peaks[1:] + r_peaks[:-1]) // 2
    beat_boundaries = np.insert(beat_boundaries, 0, 0)
    beat_boundaries = np.append(beat_boundaries, 2048)
    beat_boundary_per_beat = [
        (s, e) for s, e in zip(beat_boundaries[:-1], beat_boundaries[1:])
    ]
    beat_boundaries_per_label_dict = {0: [], 1: [], 2: []}

    for l, b in zip(flat_raw_label.values(), beat_boundary_per_beat):
        if l not in ["normal", "pac", "pvc"]:
            continue
        beat_boundaries_per_label_dict[{"normal": 0, "pac": 1, "pvc": 2}[l]].append(b)

    return beat_boundaries_per_label_dict


def flatten_raw_label(raw_label):
    raw_label_dict = {}
    for i, idx in enumerate(raw_label):
        for j in idx:
            raw_label_dict[j] = ds_beat_names[i]
    return raw_label_dict
