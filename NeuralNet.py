import numpy as np
from activations import supported_activations
import time
import datetime
from utils import *
# from network_config import cfg

class NeuralNet:
    def __init__(self, layers, activations):
        
        self.depth = len(layers)
        self.layer_sizes = [layers[0]]
        self.activation_names = []
        self.activations = []
        self.activation_derivs = []
        for size, act in zip(layers[1:], activations):
            self.layer_sizes.append(size)
            activation, activation_deriv = supported_activations[act.lower()]
            self.activations.append(activation)
            self.activation_names.append(act)
            self.activation_derivs.append(activation_deriv)

        self.x = [[] for _ in range(self.depth)]
        self.z = [[] for _ in range(self.depth)]

        rng = np.random.default_rng(seed=0)
        self.weights = [rng.random((self.layer_sizes[i], self.layer_sizes[i - 1])) - 0.5 for i in range(1, len(self.layer_sizes))]
        self.biases = [rng.random((self.layer_sizes[i],)) for i in range(1, len(self.layer_sizes))]

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

    def train(self, input_data, output_data, batch_size, hold_out):
        train_num = len(input_data) - hold_out
        train_inputs = input_data[:train_num]
        train_outputs = output_data[:train_num]
        val_inputs = input_data[train_num:]
        val_outputs = output_data[train_num:]
        start_time = time.time()
        curr_time = start_time
        train_count = 0
        #delta = np.array([[np.zeros(self.weights[i].shape) for i in range(0, len(self.weights))], [np.zeros(self.biases[i].shape) for i in range(0, len(self.biases))]])
        while True:
            validation_accuracy = 0
            for ins, outs in iterate_dataset(train_inputs, train_outputs, batch_size):
                self.feedforward(ins)
                self.backprop_update(outs, batch_size)
                self.training_atts += batch_size
                train_count += batch_size
                #delta = -1 * self.learning_rate_schedule(self.step) * avg_grad + self.alpha * delta

            for ins, outs in iterate_dataset(val_inputs, val_outputs, batch_size):
                validation_accuracy += self.test(ins, outs)
            
            print("Trained:", train_count)
            round_time = curr_time
            curr_time = time.time()
            print("Elapsed:", datetime.timedelta(seconds = round(curr_time - start_time)))
            rate = train_num / (curr_time - round_time)
            print(f'Rate: {rate:.3f} images per sec')
            validation_accuracy /= (hold_out / batch_size)
            print(f'Accuracy over validation set: {validation_accuracy * 100:.5f}%\n')
            if 1 - validation_accuracy <= self.error_rate:
                break
            #print(self.weights[0])
            #print(self.z[-1])
            # remaining = datetime.timedelta(seconds = round((len(input_data) - train_count) / rate))
            # print("Estimated Time remaining:", remaining)
            # print(f'Average Cost over last batch: {avg_cost[0]:.5f}\n')

        #plt.plot([batch_size * i for i in range(1, n // batch_size)], cost_over_time)
        #plt.ylabel("Cost")
        #plt.xlabel("Training Examples")
        #plt.show()

    def test(self, input_data, actual_data):
        self.feedforward(input_data)
        atts = len(input_data[0])
        self.test_atts += atts
        correct = 0
        outputs = np.argmax(self.z[-1], 0)
        actuals = np.argmax(actual_data, 0)
        for i in range(atts):
            if outputs[i] == actuals[i]:
                self.test_correct += 1
                correct += 1
        return correct / atts


    def forward(self, x):
        """computes output of network for given batch of inputs, x is (B x L), where B is batch size and L is size of input layer"""
        self.x[0] = x
        self.z[0] = x
        for i in range(1, self.depth):
            x = x @ self.weights[i - 1].T + self.biases[i - 1]
            self.z[i] = x
            x = self.activations[i - 1](x)
            self.x[i] = x
        return self.x[-1]


    def backward(self, loss_deriv):
        self.grad_weights = [0] * len(self.weights)
        self.grad_biases = [0] * len(self.biases)
        dLoss_dx = [0] * len(self.x)
        L = self.depth - 1
        batch_size = loss_deriv.shape[0]
        dLoss_dx[L] = loss_deriv
        while L > 0:
            dx_dz = self.activation_derivs[L - 1](self.z[L])
            dLoss_dz = dLoss_dx[L] * dx_dz
            self.grad_biases[L - 1] = 1 / batch_size * np.sum(dLoss_dz, axis=0)
            self.grad_weights[L - 1] = 1 / batch_size * np.tensordot(dLoss_dz, self.x[L - 1].T, axes=[0, 1])
            dLoss_dx[L - 1] = dLoss_dz @ self.weights[L - 1]
            L -= 1
        

    def backprop_update(self, y, batch_size):
        grad_weights = [0 for i in range(0, len(self.weights))]
        grad_biases = [0 for i in range(0, len(self.biases))]
        dLoss_dz = [0 for i in range(0, len(self.z))]
        l = len(self.z) - 1
        dLoss_dz[l] = self.loss_deriv(self.z[l], y)
        l -= 1
        dLoss_da = dLoss_dz[l + 1] * self.last_activation_deriv(self.a[l + 1])
        grad_biases[l] = np.sum(dLoss_da, 1).reshape(self.biases[l].shape)        
        self.biases[l] += -1 / batch_size * self.alpha * grad_biases[l]
        grad_weights[l] = np.zeros(self.weights[l].shape)
        for i in range(batch_size):
            grad_weights[l] += np.outer(dLoss_da[:,i], self.z[l][:,i])
        self.weights[l] += -1 / batch_size * self.alpha * grad_weights[l]
        dLoss_dz[l] = self.weights[l].T @ dLoss_da
        l -= 1
        while l >= 0:
            dLoss_da = dLoss_dz[l + 1] * self.hidden_activation_deriv(self.a[l + 1])
            grad_biases[l] = np.sum(dLoss_da, 1).reshape(self.biases[l].shape)
            self.biases[l] += -1 / batch_size * self.alpha * grad_biases[l]
            # grad_weights[l] = np.apply_over_axes(np.sum, self.z[l] @ dLoss_da, 0)
            grad_weights[l] = np.zeros(self.weights[l].shape)
            for i in range(len(self.z[l][0])):
                grad_weights[l] += np.outer(dLoss_da[:,i], self.z[l][:,i])
            #grad_biases[l] = np.apply_over_axes(np.sum, dLoss_da, 0)
            self.weights[l] += -1 / batch_size * self.alpha * grad_weights[l]
            dLoss_dz[l] = self.weights[l].T @ dLoss_da
            l -= 1

