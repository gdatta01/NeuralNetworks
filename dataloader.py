import numpy as np


class Dataloader:
    
    def __init__(self, data, labels, batch_size, normalize=True):
        data = np.array(data)
        if normalize:
            data = (data - np.mean(data, axis=1).reshape((-1, 1))) / np.std(data, axis=1).reshape(-1, 1)
        self.data = data
        self.labels = np.array(labels)
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
