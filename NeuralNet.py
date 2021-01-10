import numpy as np
from math_functions import *
import time
import datetime
from utils import *

class NeuralNet:
    def __init__(self, layer_sizes, alpha, activation, use_softmax, loss, error):
        self.training_atts = 0
        self.test_correct = 0
        self.test_atts = 0
        
        self.layers = layer_sizes
        self.a = [[] for i in layer_sizes]
        self.z = [[] for i in layer_sizes]

        self.hidden_activation = activation
        self.hidden_activation_deriv = eval(activation.__name__ + '_deriv')
        if use_softmax:
            self.last_activation = softmax
            self.last_activation_deriv = sigmoid_deriv
        else:
            self.last_activation = activation
            self.last_activation_deriv = self.hidden_activation_deriv

        self.loss_fn = loss
        self.loss_deriv = eval(loss.__name__ + '_deriv')
        rng = np.random.default_rng()
        self.weights = [rng.random((layer_sizes[i], layer_sizes[i - 1])) - 0.5 for i in range(1, len(layer_sizes))]
        self.biases = [rng.random((layer_sizes[i],)) for i in range(1, len(layer_sizes))]
        self.alpha = alpha

        self.error_rate = error
        
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

    def feedforward(self, x):
        """computes output of network for given batch of inputs, x is (input-neurons x batch-size)"""
        last = len(self.z) - 1
        self.z[0] = x
        for l in range(1, last):
            self.a[l] = self.weights[l - 1] @ self.z[l - 1]
            for i in range(self.a[l].shape[0]):
                self.a[l][i] += self.biases[l - 1][i]
            self.z[l] = self.hidden_activation(self.a[l])
        self.a[last] = self.weights[last - 1] @ self.z[last - 1]
        for i in range(self.a[last].shape[0]):
            self.a[last][i] += self.biases[last - 1][i]
        self.z[last] = self.last_activation(self.a[last])

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

