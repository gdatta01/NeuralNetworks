import gzip
import numpy as np
import os
from dataloader import Dataloader
import traceback


class MNIST:
    TRAIN_IMAGES_FILE = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS_FILE = 'train-labels-idx1-ubyte.gz'
    TRAIN_SIZE = 60000
    TEST_IMAGES_FILE = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS_FILE = 't10k-labels-idx1-ubyte.gz'
    TEST_SIZE = 10000
    CLASSES = 10
    INPUTS = 784


class SINE:
    TRAIN_FILE = 'train.npz'
    TRAIN_SIZE = 100000
    TEST_FILE = 'test.npz'
    TEST_SIZE = 1000


def get_dataloader(name, path, holdout_frac, batch_size, smoothing=0.0, test=False, normalize=True):
    datasets = {'mnist': get_mnist, 'sin': get_sin}
    data, labels = datasets[name](path, test, smoothing)
    data, labels = shuffle(data, labels)
    length = len(data)
    if holdout_frac:
        holdout_index = int(length * (1 - holdout_frac))
        data, val_data = data[:holdout_index], data[holdout_index:]
        labels, val_labels = labels[:holdout_index], labels[holdout_index:]
        return Dataloader(data, labels, batch_size or len(data), normalize), Dataloader(val_data, val_labels, len(val_data), normalize)
    else:
        return Dataloader(data, labels, batch_size or len(data), normalize)


def get_sin(folder, test, smoothing):
    if test:
        d = np.load(os.path.join(folder, SINE.TEST_FILE))
    else:
        d = np.load(os.path.join(folder, SINE.TRAIN_FILE))
    x, y = d['x'], d['y']
    d.close()
    return x, y




def get_mnist(folder, test, smoothing):
    if test:
        data = read_mnist_data(os.path.join(folder, MNIST.TEST_IMAGES_FILE), MNIST.TEST_SIZE)
        labels = convert_labels(read_mnist_labels(os.path.join(folder, MNIST.TEST_LABELS_FILE), MNIST.TEST_SIZE),
            MNIST.CLASSES, smoothing)
    else:
        data = read_mnist_data(os.path.join(folder, MNIST.TRAIN_IMAGES_FILE), MNIST.TRAIN_SIZE)
        labels = convert_labels(read_mnist_labels(os.path.join(folder, MNIST.TRAIN_LABELS_FILE), MNIST.TRAIN_SIZE),
                                      MNIST.CLASSES, smoothing)

    return data, labels


def read_mnist_data(file, num):
    with gzip.open(file, 'r') as f:
        f.read(16)
        buf = f.read(MNIST.INPUTS * num)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num, -1)
    return data


def read_mnist_labels(file, num):
    with gzip.open(file, 'r') as f:
        f = gzip.open(file, 'r')
        f.read(8)
        buf = f.read(num)
    labels = np.frombuffer(buf, dtype=np.uint8)
    return labels


def convert_labels(labels, choices, smoothing=0):
    if smoothing == 0:
        converted = np.zeros((len(labels), choices))
        converted[np.arange(len(converted)), labels] = 1
    else:
        converted = np.full((len(labels), choices), smoothing/choices)
        converted[np.arange(len(converted)), labels] = 1 - smoothing
    return converted


rng = np.random.default_rng(seed=0)


def shuffle(a, b):
    c = np.c_[a.reshape(len(a), -1), b.reshape(len(b), -1)]
    rng.shuffle(c)
    a2 = c[:, :a.size // len(a)].reshape(a.shape)
    b2 = c[:, a.size // len(a):].reshape(b.shape)
    return a2, b2