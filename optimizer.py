import numpy as np
from lr_schedules import default

class Optimizer:
    def __init__(self, network, lr, lr_schedule=None, momentum=0):
        self.nn = network
        self.lr = lr
        self.momentum = momentum
        self.lr_schedule = lr_schedule or default
        self.weight_gradients = [np.zeros(w.shape) for w in self.nn.weights]
        self.bias_gradients = [np.zeros(b.shape) for b in self.nn.biases]

    def update_lr(self, n):
        self.lr = self.lr_schedule(self.lr, n)

    def step(self):
        for i in range(len(self.nn.weights)):
            self.weight_gradients[i] = self.nn.grad_weights[i] * (1 - self.momentum) \
                + self.momentum * self.weight_gradients[i]
            self.nn.weights[i] += -1 * self.lr * self.weight_gradients[i]

            self.bias_gradients[i] = self.nn.grad_biases[i] * (1 - self.momentum) \
                + self.momentum * self.bias_gradients[i]
            self.nn.biases[i] += -1 * self.lr * self.bias_gradients[i]
        

