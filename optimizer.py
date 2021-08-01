from functions.activations import get_activation_deriv
import numpy as np
from lr_schedules import supported_lr_schedules
from functions.loss import get_loss_and_deriv

class Optimizer:
    def __init__(self, network, loss, lr, lr_schedule=None, momentum=0):
        self.nn = network
        self.lr = lr
        self.loss, self.loss_deriv = get_loss_and_deriv(loss)
        self.momentum = momentum
        self.lr_schedule = supported_lr_schedules[lr_schedule or 'none']
        self.dweights = [np.zeros(w.shape) for w in self.nn.weights]
        self.dbiases = [np.zeros(b.shape) for b in self.nn.biases]
        self.weight_gradients = [np.zeros(w.shape) for w in self.nn.weights]
        self.bias_gradients = [np.zeros(b.shape) for b in self.nn.biases]


    def backward(self, y):
        # print(y[0][0], self.nn.x[-1][0][0], end=' ')
        dLoss_dy_hat = self.loss_deriv(y, self.nn.x[-1])
        for dw, db in zip(self.dweights, self.dbiases):
            dw[...] = 0.0
            db[...] = 0.0
        dLoss_dx = [0] * len(self.nn.x)
        L = self.nn.depth - 1
        batch_size = dLoss_dy_hat.shape[0]
        dLoss_dx[L] = dLoss_dy_hat
        while L > 0:
            dx_dz = get_activation_deriv(self.nn.activation_names[L - 1])(self.nn.z[L])
            dLoss_dz = dLoss_dx[L] * dx_dz
            self.dbiases[L - 1] = 1 / batch_size * np.sum(dLoss_dz, axis=0)
            self.dweights[L - 1] = 1 / batch_size * np.tensordot(dLoss_dz, self.nn.x[L - 1].T, axes=[0, 1])
            dLoss_dx[L - 1] = dLoss_dz @ self.nn.weights[L - 1]
            L -= 1
        # print(dLoss_dy_hat.max(), dLoss_dz.max(), dLoss_dx[0].max(), end = ' ')
        

    def update_lr(self, n):
        self.lr = self.lr_schedule(self.lr, n)


    def step(self):
        # print(self.nn.weights[0].max(), end=' ')
        for i in range(len(self.nn.weights)):
            self.weight_gradients[i] = self.dweights[i] * (1 - self.momentum) \
                + self.momentum * self.weight_gradients[i]
            self.nn.weights[i] += -1 * self.lr * self.weight_gradients[i]
            self.bias_gradients[i] = self.dbiases[i] * (1 - self.momentum) \
                + self.momentum * self.bias_gradients[i]
            self.nn.biases[i] += -1 * self.lr * self.bias_gradients[i]
        # print(self.dweights[0].max(), self.weight_gradients[0].max(), self.nn.weights[0].max())
