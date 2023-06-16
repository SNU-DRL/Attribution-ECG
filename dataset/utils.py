from collections import OrderedDict

import numpy as np

MITBIH_classes = ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F']#, 'P', '/', 'f', 'u']

AAMI_classes = OrderedDict()
AAMI_classes['N'] = ['N', 'L', 'R']
AAMI_classes['SVEB'] = ['A', 'a', 'J', 'S', 'e', 'j']
AAMI_classes['VEB'] = ['V', 'E']
AAMI_classes['F'] = ['F']
AAMI_classes['Q'] = ['P', '/', 'f', 'u']

BEAT_INDEX = {
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


def build_label_array(label_indices, locations):
    label_array = [np.array([]) for _ in range(len(BEAT_INDEX))]
    for label in np.unique(label_indices):
        label_idx = locations[label == label_indices]
        label_array[label] = label_idx
    return label_array
