import numpy as np


def squared_loss(z, y):
    return 0.5 * np.pow(y - z, 2)

def squared_loss_deriv(z, y):
    return y - z

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
    return ex / np.apply_along_axis(np.sum, 0, ex)

def logistic_loss(y, z):
    return -1 * y * np.log(z)

def logistic_loss_deriv(y, z):
    return -1 * y / z 

supported_activations = {
    'relu': (ReLU, ReLU_deriv),
    'sigmoid': (sigmoid, sigmoid_deriv),
    'softmax': (softmax, sigmoid_deriv)
}

supported_loss = {
    'logistic': (logistic_loss, logistic_loss_deriv), 
    'squared': (squared_loss, squared_loss_deriv)
}