import numpy as np
from functions import activations


# from network_config import cfg

class NeuralNet:
    def __init__(self, inputs, layers):
        
        self.depth = len(layers) + 1
        self.inputs = inputs
        self.layer_sizes = [inputs]
        self.activation_names = []
        self.activations = []
        self.activation_derivs = []
        for layer in layers:
            if isinstance(layer, list):
                if len(layer) == 2:
                    size, act = layer
                else:
                    size = layer[0]
                    act = activations.ActivationKeys.IDENTITY
            else:
                size = layer
                act = activations.ActivationKeys.IDENTITY
            self.layer_sizes.append(size)
            activation = activations.get_activation(act)
            self.activations.append(activation)
            self.activation_names.append(act)

        self.outputs = self.layer_sizes[-1]
        self.x = [[] for _ in range(self.depth)]
        self.z = [[] for _ in range(self.depth)]

        rng = np.random.default_rng(seed=0)
        self.weights = [rng.normal(loc=0, scale=np.sqrt(1 / max(self.layer_sizes[i - 1], 64)), size=(self.layer_sizes[i], self.layer_sizes[i - 1])) for i in range(1, len(self.layer_sizes))]
        self.biases = [rng.normal(loc=0, scale=np.sqrt(1 / max(self.layer_sizes[i - 1], 64)), size=(self.layer_sizes[i],)) for i in range(1, len(self.layer_sizes))]


    def __call__(self, x):
        return self.forward(x)


    def __str__(self):
        arrow = "\n |\n\\ /\n\n"
        s = f"INPUTS: {self.layer_sizes[0]}"
        for i in range(1, self.depth):
            s += (f"{arrow}LAYER {i}\n"
            f"Units: {self.layer_sizes[i]}\n"
            f"W: {self.weights[i - 1].shape}\n"
            f"B: {self.biases[i - 1].shape}\n"
            f"Activation: {self.activation_names[i - 1]}")
        return s


    def forward(self, x):
        """computes output of network for given batch of inputs, x is (B x L), where B is batch size and L is size of input layer"""
        self.x[0] = x
        self.z[0] = x
        for i in range(1, self.depth):
            x = np.matmul(x, self.weights[i - 1].T + self.biases[i - 1])
            self.z[i] = x
            x = self.activations[i - 1](x)
            self.x[i] = x
        return self.x[-1]
