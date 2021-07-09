import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    y = sigmoid(x)
    return y * (1 - y) 


def ReLU(x):
    return np.maximum(0, x)


def ReLU_deriv(x):
    return np.where(x >= 0, 1, 0)


def softmax(x):
    ex = np.exp(x)
    return ex / np.sum(ex, axis=1)


supported_activations = {
    'relu': (ReLU, ReLU_deriv),
    'sigmoid': (sigmoid, sigmoid_deriv),
    'softmax': (softmax, sigmoid_deriv)
}
