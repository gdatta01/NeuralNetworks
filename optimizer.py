from functions.activations import get_activation_deriv
import numpy as np
from lr_schedules import supported_lr_schedules
from functions.loss import get_loss_and_deriv


class Optimizer:
    def __init__(self, network, loss, lr, lr_schedule=None, momentum=0, weight_decay=0):
        self.nn = network
        self.lr = lr
        self.loss, self.loss_deriv = get_loss_and_deriv(loss)
        self.momentum = momentum
        self.wd = weight_decay
        self.lr_schedule = supported_lr_schedules[lr_schedule or 'none']
        self.dweights = 0
        self.dbiases = 0
        self.weight_gradients = [np.zeros(w.shape) for w in self.nn.weights]
        self.bias_gradients = [np.zeros(b.shape) for b in self.nn.biases]


    def backward(self, y):
        dLoss_dy_hat = self.loss_deriv(y, self.nn.x[-1])
        self.dweights = [0 for _ in self.nn.weights]
        self.dbiases = [0 for _ in self.nn.biases]
        dLoss_dx = [0] * len(self.nn.x)
        L = self.nn.depth - 1
        batch_size = dLoss_dy_hat.shape[0]
        dLoss_dx[L] = dLoss_dy_hat
        while L > 0:
            dx_dz = get_activation_deriv(self.nn.activation_names[L - 1])(self.nn.z[L])
            dLoss_dz = dLoss_dx[L] * dx_dz
            self.dbiases[L - 1] = 1 / batch_size * np.sum(dLoss_dz, axis=0) + self.wd * self.nn.biases[L - 1]
            self.dweights[L - 1] = 1 / batch_size * np.tensordot(dLoss_dz, self.nn.x[L - 1].T, axes=[0, 1]) + \
                                   self.wd * self.nn.weights[L - 1]
            dLoss_dx[L - 1] = dLoss_dz @ self.nn.weights[L - 1]
            L -= 1


    def update_lr(self, n):
        self.lr = self.lr_schedule(self.lr, n)


    def step(self):
        for i in range(len(self.nn.weights)):
            self.weight_gradients[i] = self.dweights[i] * (1 - self.momentum) \
                                       + self.momentum * self.weight_gradients[i]
            self.nn.weights[i] += -1 * self.lr * self.weight_gradients[i]
            self.bias_gradients[i] = self.dbiases[i] * (1 - self.momentum) \
                                     + self.momentum * self.bias_gradients[i]
            self.nn.biases[i] += -1 * self.lr * self.bias_gradients[i]
