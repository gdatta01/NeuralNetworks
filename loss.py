import numpy as np


def squared_loss(y, y_hat):
    return 0.5 * np.sum(np.power(y - y_hat, 2))


def squared_loss_deriv(y, y_hat):
    return y_hat - y


def crossentropy_loss(y, y_hat):
    return -1 * np.sum(y * np.log(y_hat))


def crossentropy_loss_deriv(y, y_hat):
    return -1 * y / y_hat 


supported_loss = {
    'crossentropy': (crossentropy_loss, crossentropy_loss_deriv),
    'squared': (squared_loss, squared_loss_deriv)
}