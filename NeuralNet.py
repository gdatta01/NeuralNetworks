import numpy as np
from math_functions import *

class NeuralNet:
    def __init__(self, layer_sizes, step = 0.1, alpha = 0.5, activation=ReLU, use_softmax=True, loss=logistic_loss):
        self.training_correct = 0
        self.training_atts = 0
        self.test_correct = 0
        self.test_atts = 0
        
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
        
        self.weights = [np.random.rand(layer_sizes[i], layer_sizes[i - 1]) - 0.5 for i in range(1, len(layer_sizes))]
        self.biases = [np.random.rand((layer_sizes[i], 1)) for i in range(1, len(layer_sizes))]
        self.alpha = alpha
        
    def train(self, input_data, output_data, batch_size=100):
        """should take in entire dataset to train, feedforward,
           then optimize for each training example"""
        """create sets of batch examples, feedforward and compute cost, then backprop"""
        m, n = 0, batch_size
        cost_over_time = []
        start_time = time.time()
        #dstep = 0.001665
        train_count = 0
        #delta = np.array([[np.zeros(self.weights[i].shape) for i in range(0, len(self.weights))], [np.zeros(self.biases[i].shape) for i in range(0, len(self.biases))]])
        while n <= len(input_data):
            batch_in = input_data[m:n]
            batch_out = output_data[m:n]
            count = 0
            sum_grad = np.array([[np.zeros(self.weights[i].shape) for i in range(0, len(self.weights))], [np.zeros(self.biases[i].shape) for i in range(0, len(self.biases))]])
            sum_cost = 0
            round_time = time.time()
            for input, expected in zip(batch_in, batch_out):
                count += 1
                a = self.feedforward(input)
                self.training_atts += 1
                train_count += 1
                gradient = self.backprop(a, expected)
                sum_grad += gradient
                sum_cost += sum(cost_calc(a, expected))

            avg_grad = sum_grad / count
            avg_cost = sum_cost / count
            cost_over_time.append(avg_cost)

            #delta = -1 * self.learning_rate_schedule(self.step) * avg_grad + self.alpha * delta
            
            self.weights -= self.alpha * avg_grad[0]
            self.biases -= -self.alpha * avg_grad[1]
            
            m += batch_size
            n += batch_size

            print("Trained:", self.training_atts)
            curr_time = time.time()
            print("Elapsed:", datetime.timedelta(seconds = round(curr_time - start_time)))
            rate = batch_size / (curr_time - round_time)
            print(f'Rate: {rate:.3f} images per sec')
            remaining = datetime.timedelta(seconds = round((len(input_data) - train_count) / rate))
            print("Estimated Time remaining:", remaining)
            print(f'Average Cost over last batch: {avg_cost[0]:.5f}\n')

        #plt.plot([batch_size * i for i in range(1, n // batch_size)], cost_over_time)
        #plt.ylabel("Cost")
        #plt.xlabel("Training Examples")
        #plt.show()

    def test(self, input_data, actual_data):
        z = self.feedforward(input_data)
        atts = len(input_data[0])
        self.test_atts += atts
        correct = 0
        outputs = np.argmax(z, 0)
        actuals = np.argmax(actual_data, 0)
        for i in range(atts):
            if outputs[i] == actuals[i]:
                self.test_correct += 1
                correct += 1
        return correct / atts

    def feedforward(self, x):
        """computes output of network for given batch of inputs, x is (len input-neurons x batch-size)"""
        last = len(self.z) - 1
        self.z[0] = x
        for l in range(1, last):
            self.a[l] = self.weights[l - 1] @ self.z[l - 1] + self.biases[l - 1]
            self.z[l] = self.hidden_activation(self.a[l])
        self.a[last] = self.weights[last - 1] @ self.z[last - 1] + self.biases[last - 1]
        self.z[last] = self.last_activation(self.a[last])
        return self.z[last]

    def backprop(self, y, z):
        grad_weights = [[] for i in range(0, len(self.weights))]
        grad_biases = [[] for i in range(0, len(self.biases))]
        dLoss_dz = [[] for i in range(0, len(self.z))]
        
        l = len(self.z) - 1
        dLoss_dz[l] = self.loss_deriv(z, y)
        l -= 1


        dLoss_da = dLoss_dz[l + 1] * self.last_activation_deriv(self.a[l])
        grad_weights[l] = np.apply_over_axes(np.sum, dLoss_da * self.z[l], 0)
        grad_biases[l] = np.apply_over_axes(np.sum, dLoss_da, 0)
        dLoss_dz[l] = self.weights[l] @ dLoss_da
        l -= 1
        
        while l > 0:
            dLoss_da = dLoss_dz[l + 1] * self.hidden_activation_deriv(self.a[l])
            grad_weights[l] = np.apply_over_axes(np.sum, dLoss_da * self.z[l], 0)
            grad_biases[l] = grad_biases[l] = np.apply_over_axes(np.sum, dLoss_da, 0)
            dLoss_dz[l] = self.weights[l] @ dLoss_da
            l -= 1
        return [grad_weights, grad_biases]
