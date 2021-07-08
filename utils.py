import random
import gzip
import numpy as np


def read_data(file, num, w, h):
    with gzip.open(file, 'r') as f:
        f.read(16)
        buf = f.read(w * h * num)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num, w, h)
    return data


def read_labels(file, num):
    with gzip.open(file, 'r') as f:
        f = gzip.open(file,'r')
        f.read(8)
        buf = f.read(num)
    labels = np.frombuffer(buf, dype=np.uint8)
    return labels


def convert_labels(labels, choices):
    for label in labels:
        r = (np.full((choices, 1), 0.02, np.float64))
        r[label[0]] = 1 - 0.02 * choices
        yield r


def shuffle(a, b):
    c = list(zip(a, b))
    random.shuffle(c)
    a, b = zip(*c)
    return a, b


def iterate_dataset(x, y, batch):
    x, y = shuffle(x, y)
    index = 0
    while index < len(x):
        xs = x[index:index + batch]
        ys = y[index:index + batch]
        index += batch
        yield np.hstack(xs), np.hstack(ys)
