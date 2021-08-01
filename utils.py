import random
import gzip
import numpy as np


def read_data(file, num, l):
    with gzip.open(file, 'r') as f:
        f.read(16)
        buf = f.read(l * num)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num, l)
    return data


def read_labels(file, num):
    with gzip.open(file, 'r') as f:
        f = gzip.open(file,'r')
        f.read(8)
        buf = f.read(num)
    labels = np.frombuffer(buf, dtype=np.uint8)
    return labels



def convert_labels(labels, choices, incorrect_frac=0):
    if incorrect_frac==0:
        converted = np.zeros((len(labels), choices))
        converted[np.arange(len(converted)), labels] = 1
    else:
        converted = np.full((len(labels), choices), incorrect_frac/choices)
        converted[np.arange(len(converted)), labels] = 1 - incorrect_frac
    return converted



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
