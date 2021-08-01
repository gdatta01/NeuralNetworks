import numpy as np


eps = 1e-5

def squared_loss(y, y_hat):
    return np.mean(0.5 * np.sum(np.power(y - y_hat, 2), axis=1))


def squared_loss_deriv(y, y_hat):
    return (y_hat - y) / y.shape[0]


def crossentropy_loss(y, x):
    return np.mean(np.sum(-y * (x - np.log(np.sum(np.exp(x), axis=1)).reshape(-1, 1)), axis=1))


def crossentropy_loss_deriv(y, x):
    ex = np.exp(x - x.max(axis=1).reshape(-1, 1))
    return -y * (1 - ex / np.sum(ex, axis=1).reshape(-1, 1)) / y.shape[0]


_supported_loss = {
    'crossentropy': (crossentropy_loss, crossentropy_loss_deriv),
    'squared': (squared_loss, squared_loss_deriv)
}

def get_loss_and_deriv(loss):
    return _supported_loss[loss]
