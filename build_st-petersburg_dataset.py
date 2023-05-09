# https://www.physionet.org/content/mitdb/1.0.0/

import gzip
import os
import pickle
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import wfdb
from tqdm import tqdm

WINDOW_SECONDS = 8
DATA_DIR = 'physionet.org/files/incartdb/1.0.0'
RESULT_DIR = './data'
os.makedirs(RESULT_DIR, exist_ok=True)

# Automatic classification of heartbeats using ECG morphology and heartbeat interval features
# Train set
DS_1 = [f'I{i:02d}' for i in range(1, 38)]
# Test set
DS_2 = [f'I{i:02d}' for i in range(38, 76)]

DS = {"train": DS_1, "test": DS_2}

AAMI_classes = OrderedDict()
AAMI_classes['N'] = ['N', 'L', 'R']
AAMI_classes['SVEB'] = ['A', 'a', 'J', 'S', 'e', 'j']
AAMI_classes['VEB'] = ['V', 'E']
AAMI_classes['F'] = ['F']
AAMI_classes['Q'] = ['P', '/', 'f', 'u']

MITBIH_BEAT_INDEX = {
    0: "N",
    1: "SVEB",
    2: "VEB",
    3: "F",
    4: "Q"
}


def get_beat_idx(label):
    for class_idx, beat_classes in enumerate(AAMI_classes.values()):
        if label in beat_classes:
            return class_idx
    return 0 # if label not in any beat_classes -> return 0 (normal)


def build_label_dict(label_indices, locations):
    label_dict = {}
    for label in np.unique(label_indices):
        label_idx = locations[label == label_indices]
        label_dict[label] = label_idx
    return label_dict

result_dict = {}
for set_key, set_pids in DS.items():
    X = []
    Y = []
    for pid in tqdm(set_pids):
        record = wfdb.rdrecord(f'{DATA_DIR}/{pid}')
        attr = wfdb.rdann(f'{DATA_DIR}/{pid}', extension='atr')
        sampling_rate = record.fs
        recording = record.p_signal.astype(np.float32)
        
        window_size = WINDOW_SECONDS * sampling_rate
        n_samples = recording.shape[0] // window_size
        
        for i in range(n_samples):
            window_start = i * window_size
            window_end = window_start + window_size
        
            x = recording[window_start:window_end, 1] # use lead II

            beat_location_indices = ((window_start <= attr.sample) & (attr.sample < window_end)).nonzero()[0]
            beat_locations = attr.sample[beat_location_indices] - window_start
            
            beat_classes = np.take(attr.symbol, beat_location_indices)
            beat_label_indices = np.array(list(map(get_beat_idx, beat_classes)))

            labels, labels_count = np.unique(beat_label_indices, return_counts=True)
            is_abnormal = (labels != 0)
            
            if any(is_abnormal):
                abnormal_count = np.where(is_abnormal, labels_count, 0)
                label = labels[np.argmax(abnormal_count)]
            else:
                label = 0

            label_dict = build_label_dict(beat_label_indices, beat_locations)

            y = {
                'pid': pid,
                'y': label,
                'y_raw': label_dict
            }

            X.append(x)
            Y.append(y)

    X = np.array(X)

    result_dict[set_key] = {"X": X, "Y": Y}

with gzip.open(f'{RESULT_DIR}/st-petersburg.pkl', 'wb') as f:
    pickle.dump(result_dict, f)
