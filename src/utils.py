import numpy as np


def standardization(data):
    """
    data.shape: (# samples, 1(lead), frame_size), numpy
    """
    m = np.expand_dims(data.mean(-1), -1)
    s = np.expand_dims(data.std(-1), -1)
    return (data - m) / (s + 1e-6)


def preprocess(data):
    return standardization(data)
