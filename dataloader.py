import numpy as np


class Dataloader:
    
    def __init__(self, data, labels, batch_size, normalize=True, flatten=False):
        if flatten and len(data.shape) > 2:
            data = data.reshape(-1, np.prod(data.shape[1:]))
        data = data - data.mean(axis=1).reshape((-1, 1))
        self.data = data / np.abs(data).max()
        self.labels = labels
        self.batch_size = batch_size


    def dataset(self):
        index = 0
        while index < len(self.data):
            xs = self.data[index:index + self.batch_size]
            ys = self.labels[index:index + self.batch_size]
            index += self.batch_size
            yield xs, ys


    def __call__(self):
        return self.dataset()

    def __iter__(self):
        return self()
