import logging
from datetime import datetime
from functions.loss import squared_loss
from functions.activations import softmax
import numpy as np


class Trainer:
    def __init__(self, network, optimizer, train_loader, val_loader):
        self.network = network
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.batch_losses = []
        self.epoch_losses = []
        self.val_accuracy = []
        if self.network.outputs == 1:
            self.calc_accuracy = self.regression_accuracy
        else:
            self.calc_accuracy = self.classification_accuracy
        self.logger = logging.getLogger(__name__)


    def train_epoch(self):
        self.batch_losses = []
        for batch in self.train_loader:
            x, y = batch
            output = self.network(x)
            self.optimizer.backward(y)
            self.batch_losses.append(self.optimizer.loss(y, output))
            self.optimizer.step()


    def train(self, epochs):
        for n in range(epochs):
            d1 = datetime.now()
            self.optimizer.update_lr(n)
            self.train_epoch()
            d2 = datetime.now()
            self.epoch_losses.append(sum(self.batch_losses) / len(self.batch_losses))
            self.validate()
            delta = str(d2 - d1)[:-5]
            self.logger.info(f"Epoch {n} Duration {delta} Train Loss {self.epoch_losses[-1]:.4f} Val "
                        f"{self.val_accuracy[-1]:.4f}")


    def validate(self):
        self.val_accuracy.append(0.0)
        for batch in self.val_loader:
            x, y = batch
            output = self.network(x)
            acc = self.calc_accuracy(y, output)
            self.val_accuracy.append(acc)


    @staticmethod
    def classification_accuracy(y, x):
        y_hat = softmax(x)
        return np.count_nonzero(np.argmax(y_hat, axis=1) == np.argmax(y, axis=1)) / y.shape[0]


    @staticmethod
    def regression_accuracy(y, y_hat):
        return squared_loss(y, y_hat)




