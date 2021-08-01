from optimizer import Optimizer
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, network, loss, dataset_factory, lr, lr_schedule=None, momentum=0):
        self.network = network
        # self.loss, self.loss_deriv = supported_loss[loss]
        self.optimizer = Optimizer(network, loss, lr, lr_schedule, momentum)
        self.dataset_factory = dataset_factory


    def train_epoch(self):
        dataset = self.dataset_factory()
        for batch in dataset:
            x, y = batch
            self.network(x)
            self.optimizer.backward(y)
            self.optimizer.step()


    def train(self, epochs):
        for n in range(epochs):
            d1 = datetime.now()
            self.optimizer.update_lr(n)
            self.train_epoch()
            d2 = datetime.now()
            delta = str(d2 - d1)[:-5]
            logger.info(f"{d2} Epoch {n} Duration {delta}")

