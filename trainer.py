from optimizer import Optimizer
from NeuralNet import NeuralNet
from loss import supported_loss


class Trainer:
    def __init__(self, network, loss, dataset_factory, lr, lr_schedule=None, momentum=0):
        self.network = network
        self.loss, self.loss_deriv = supported_loss[loss]
        self.optimizer = Optimizer(network, lr, lr_schedule, momentum)
        self.dataset_factory = dataset_factory


    def train_epoch(self):
        dataset = self.dataset_factory()
        for batch, labels in dataset:
            predicted = self.network(batch)
            self.network.backward(self.loss_deriv(labels, predicted))
            self.optimizer.step()
        
    def train(self, epochs):
        for n in range(epochs):
            self.train_epoch()
            self.optimizer.update_lr(n)

