# https://www.physionet.org/content/mitdb/1.0.0/

import argparse
import gzip
import pickle
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import tqdm
import wfdb

parser = argparse.ArgumentParser()
parser.add_argument('--type', default='train', choices=['train', 'test']) 
args = parser.parse_args()

# Automatic classification of heartbeats using ECG morphology and heartbeat interval features
DS_1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS_2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

DS = DS_1 if args.type == 'train' else DS_2

MITBIH_classes = ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F']#, 'P', '/', 'f', 'u']

AAMI_classes = OrderedDict()
AAMI_classes['N'] = ['N', 'L', 'R']
AAMI_classes['SVEB'] = ['A', 'a', 'J', 'S', 'e', 'j']
AAMI_classes['VEB'] = ['V', 'E']
AAMI_classes['F'] = ['F']
AAMI_classes['Q'] = ['P', '/', 'f', 'u']

WINDOW_SEC = 8
DATA_DIR = 'mit-bih-arrhythmia-database-1.0.0'

def get_label_idx(label):
    label_idx = np.where(list(map(lambda x: label in x, AAMI_classes.values())))[0]
    return label_idx[0] if len(label_idx) != 0 else 0

def get_label_idx_per_beat(label_per_beat, beat_idx):
    label_idx_per_beat = {}
    for label in np.unique(label_per_beat):
        label_idx = beat_idx[label == label_per_beat]
        label_idx_per_beat[label] = label_idx
    return label_idx_per_beat

X = []
Y = []

for pid in tqdm.tqdm(DS):
    record = wfdb.rdrecord(f'{DATA_DIR}/{pid}')
    attr = wfdb.rdann(f'{DATA_DIR}/{pid}', extension='atr')
    sampling_rate = record.fs
    recording = record.p_signal
    
    window_size = WINDOW_SEC * sampling_rate
    n_samples = recording.shape[0] // window_size
    
    for i in range(n_samples):
        from_idx = i * window_size
        to_idx = (i+1) * window_size
    
        x = recording[from_idx:to_idx].T
        beat_pos_rec = np.arange(len(attr.sample))[(attr.sample > from_idx) * (attr.sample < to_idx)]
        beat_pos_sample = np.take(attr.sample, beat_pos_rec) - from_idx

        class_per_beat = np.take(attr.symbol, beat_pos_rec)
        label_per_beat = np.array(list(map(get_label_idx, class_per_beat)))

        labels, labels_count = np.unique(label_per_beat, return_counts=True)
        is_disease = labels != 0
        
        if any(is_disease):
            disease_count = np.where(is_disease, labels_count, 0)
            y = labels[np.argmax(disease_count)]
        else:
            y = 0

        label_idx_per_beat = get_label_idx_per_beat(label_per_beat, beat_pos_sample)

        y = {
            'pid': pid,
            'y': y,
            'y_raw': label_idx_per_beat
        }

        X.append(x)
        Y.append(y)

X = np.array(X)
with gzip.open(f'mitbih_{args.type}.pkl', 'wb') as fh:
    pickle.dump((X, Y), fh, protocol=4)
