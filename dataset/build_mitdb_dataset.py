# https://www.physionet.org/content/mitdb/1.0.0/

import gzip
import os
import pickle

import numpy as np
import wfdb
from tqdm import tqdm

from utils import build_label_array, get_beat_idx

WINDOW_SECONDS = 8
DATA_DIR = './source/mit-bih-arrhythmia-database-1.0.0'
RESULT_DIR = './data'
os.makedirs(RESULT_DIR, exist_ok=True)

# Train set
DS_1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
# Test set
DS_2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

DS = {"train": DS_1, "test": DS_2}

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
        
            x = recording[window_start:window_end, 0] # use lead II

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

            label_array = build_label_array(beat_label_indices, beat_locations)

            y = {
                'pid': pid,
                'y': label,
                'y_raw': label_array
            }

            X.append(x)
            Y.append(y)

    X = np.array(X)

    result_dict[set_key] = {"X": X, "Y": Y}

with gzip.open(f'{RESULT_DIR}/mitdb.pkl', 'wb') as f:
    pickle.dump(result_dict, f)
