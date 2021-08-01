import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    y = sigmoid(x)
    return y * (1 - y) 


def ReLU(x):
    return x.clip(0)


def ReLU_deriv(x):
    return np.where(x >= 0, 1, 0)


def hardtanh(x):
    return x.clip(-1, 1)


def hartanh_deriv(x):
    return np.where(np.logical_and(x >= -1, x <= 1), 1, 0)


def softmax(x):
    x = x - np.max(x, axis=1).reshape((-1, 1))
    ex = np.exp(x)
    return ex / np.sum(ex, axis=1).reshape((-1, 1))


def identity(x):
    return x


def identity_deriv(x):
    return 1


class ActivationKeys:
    RELU = 'relu'
    SIGMOID = 'sigmoid'
    SOFTMAX = 'softmax'
    HARDTANH = 'hardtanh'
    IDENTITY = 'identity'

_supported_activations = {
    ActivationKeys.RELU: (ReLU, ReLU_deriv),
    ActivationKeys.SIGMOID: (sigmoid, sigmoid_deriv),
    ActivationKeys.SOFTMAX: (softmax, sigmoid_deriv),
    ActivationKeys.HARDTANH: (hardtanh, hartanh_deriv),
    ActivationKeys.IDENTITY: (identity, identity_deriv)
}


def get_activation(activation):
    return _supported_activations[activation.lower()][0]


def get_activation_deriv(activation):
    return _supported_activations[activation.lower()][1]