from optimizer import Optimizer
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, network, loss, dataset_factory, lr, lr_schedule=None, momentum=0, weight_decay=0):
        self.network = network
        self.optimizer = Optimizer(network, loss, lr, lr_schedule, momentum, weight_decay)
        self.dataset_factory = dataset_factory
        self.batch_losses = []
        self.epoch_losses = []


    def train_epoch(self):
        self.batch_losses = []
        dataset = self.dataset_factory()
        for batch in dataset:
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
            delta = str(d2 - d1)[:-5]
            logger.info(f"{d2} Epoch {n} Duration {delta} Loss {self.epoch_losses[-1]:.2f}")
