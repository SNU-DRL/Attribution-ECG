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

def preprocess(data):
    """
    Standardization
    data.shape: (# samples, 1(lead), frame_size)
    """
    m = np.expand_dims(data.mean(-1), -1)
    s = np.expand_dims(data.std(-1), -1)
    return (data - m) / (s + 1e-6)


def get_beat_spans(y_raw, len_x=2049):
    beats = extract_beats(y_raw)
    beats = dict(sorted(beats.items()))

    r_peaks = np.array(list(beats.keys()))
    beat_boundaries = (r_peaks[1:] + r_peaks[:-1]) // 2
    beat_onsets = np.insert(beat_boundaries, 0, 0) # inclusive
    beat_offsets = np.append(beat_boundaries, len_x) # exclusive
    beat_spans = {0: [], 1: [], 2: []}

    for beat, onset, offset in zip(beats.values(), beat_onsets, beat_offsets):
        if beat not in ["normal", "pac", "pvc"]:
            continue
        beat_spans[ICENTIA_LABEL_MAPPING[beat]].append((onset, offset)) # span: [onset, offset-1]

    return beat_spans


def extract_beats(y_raw):
    label_dict = {}
    for i, indices in enumerate(y_raw):
        for j in indices:
            label_dict[j] = ICENTIA_BEAT_INDEX[i]
    return label_dict
