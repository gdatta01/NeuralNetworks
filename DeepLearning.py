import numpy as np
import matplotlib.pyplot as plt
import math
import gzip
import time
from math import sqrt
import datetime
import random

def cost(a, y):
    return (a - y)**2

def deriv_cost(a, y):
    return 2 * (a - y)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def inv_sigmoid(x):
    return math.log(x/(1-x))

def relu(x):
    return max(0, x)

def deriv_sigmoid(x):
    y = sigmoid(x)
    return y * (1 - y)

def deriv_relu(x):
    return 0 if x < 0 else 1

def get_images(file, num_images, image_size):
    try:
        f = gzip.open(file,'r')
    except Error as e:
        print("File not found")
        exit()

    f.read(16)
    buf = f.read(image_size * image_size * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_images, image_size, image_size, 1)
    images = []
    for i in range(num_images):
        images.append(np.asarray(data[i]).squeeze())
    return images

def get_labels(file, num_images):
    f = gzip.open(file,'r')
    f.read(8)
    labels = []
    for i in range(0,num_images):
        buf = f.read(1)
        labels.append(np.frombuffer(buf, dtype=np.uint8).astype(np.int64))
    return labels

def convert_labels(labels, choices):
    lst = []
    for label in labels:
        lst.append(np.zeros(choices).reshape(choices, 1))
        lst[-1][label[0]] = 1
    return lst

def vectorize_image(img):
    return np.reshape(img, (img.size, 1))

def grad_size(grad):
    total = 0
    for side in grad:
        for layer in side:
            for row in layer:
                for val in row:
                    total += val ** 2
    return sqrt(total)

def memo(f):
    cache = {}
    def memoized(l, j):
        if (l, j) not in cache:
            cache[(l, j)] = f(l, j)
        return cache[(l, j)]
    return memoized

def shuffle(a, b):
    c = list(zip(a, b))
    random.shuffle(c)
    a, b = zip(*c)
    return a, b

class NeuralNet:
    def __init__(self, layers, step = 0.1, alpha = 0.5, activation=sigmoid, da_dz=deriv_sigmoid, learning_rate_schedule = lambda x: x):
        self.training_correct = 0
        self.training_atts = 0
        self.test_correct = 0
        self.test_atts = 0
        self.neurons = []
        self.z = []
        for layer in layers:
            self.neurons.append(np.zeros(layer).reshape((layer, 1)))
            self.z.append(np.zeros(layer).reshape((layer, 1)))
        self.activation = np.vectorize(activation)
        self.activation_deriv = da_dz
        self.weights = [2 * np.random.rand(layers[i], layers[i - 1]) - 1 for i in range(1, len(layers))]
        self.biases = [np.zeros((layers[i], 1)) for i in range(1, len(layers))]
        self.step = step
        self.alpha = alpha
        self.learning_rate_schedule = learning_rate_schedule
        
    def train(self, input_data, output_data, batch_size=100):
        """should take in entire dataset to train, feedforward,
           then optimize for each training example"""
        """create sets of batch examples, feedforward and compute cost, then backprop"""
        cost_calc = np.vectorize(cost)
        m, n = 0, batch_size
        cost_over_time = []
        start_time = time.time()
        #dstep = 0.001665
        train_count = 0
        delta = np.array([[np.zeros(self.weights[i].shape) for i in range(0, len(self.weights))], [np.zeros(self.biases[i].shape) for i in range(0, len(self.biases))]])
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

            delta = -1 * self.learning_rate_schedule(self.step) * avg_grad + self.alpha * delta
            
            self.weights += delta[0]
            self.biases += delta[1]
            
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

    def test(self, input_data, output_data):
        a = self.feedforward(input_data)
        self.test_atts += 1
        if np.argmax(a) == np.argmax(output_data):
            self.test_correct += 1
        
    def feedforward(self, input):
        """computes output of network for given input"""
        assert type(input) is np.ndarray
        assert input.shape == self.neurons[0].shape
        self.neurons[0] = input
        for l in range(1, len(self.neurons)):
            self.z[l] = np.dot(self.weights[l - 1], self.neurons[l - 1]) + self.biases[l - 1]
            self.neurons[l] = self.activation(self.z[l])
        return self.neurons[-1]

    def backprop(self, a, y):
        grad_weights = [np.zeros(self.weights[i].shape) for i in range(0, len(self.weights))]
        grad_biases = [np.zeros(self.biases[i].shape) for i in range(0, len(self.biases))]
        dc_da = [np.zeros(self.neurons[i].shape) for i in range(0, len(self.neurons))]
        l = len(self.neurons) - 1
        for h in range(len(self.neurons[l])):
            dc_da[l][h] = deriv_cost(a[h], y[h])
        while l > 0:
            for i in range(len(self.weights[l - 1])):
                z_prime = self.activation_deriv(self.z[l][i])
                if l < len(self.neurons) - 1:
                    dc_da[l][i] = sum([self.weights[l][h, i] * self.activation_deriv(self.z[l][h]) * dc_da[l + 1][h] for h in range(len(self.neurons[l + 1]))])
                dc_dali = dc_da[l][i]
                for j in range(len(self.weights[l - 1][i])):
                    grad_weights[l - 1][i, j] = self.neurons[l - 1][j] * z_prime * dc_dali
                grad_biases[l - 1][i] = z_prime * dc_dali
            l -= 1
        return np.array([grad_weights, grad_biases])


def run_training(num=10000, batch=100, repeat = True, img_file='train-images-idx3-ubyte.gz', label_file='train-labels-idx1-ubyte.gz', size=28, choices=10):
    images = get_images(img_file, num, size)
    images = [vectorize_image(img) for img in images]
    labels = get_labels(label_file, num)
    output = convert_labels(labels, choices)
    nn.train(images, output, batch)
    if repeat:
        batch *= 10
        nn.step = 0.05
        nn.alpha = 0.25
        images, output = shuffle(images, output)
        nn.train(images, output, batch)
        nn.step = 0.01
        nn.alpha = 0.1
        images, output = shuffle(images, output)
        nn.train(images, output, batch)

def run_testing():
    cur_correct = nn.test_correct
    cur_atts = nn.test_atts
    for img, actual in zip(iter(test_images), iter(test_labels)):
        img = vectorize_image(img)
        nn.test(img, actual)
    print(f'Accuracy: {((nn.test_correct - cur_correct) / (nn.test_atts - cur_atts) *100):.3f}%\n')

nn = NeuralNet([784, 16, 16, 10])
test_images = get_images("t10k-images-idx3-ubyte.gz", 1000, 28)
image_iter = iter(test_images)
test_labels = get_labels("t10k-labels-idx1-ubyte.gz", 1000)
label_iter = iter(convert_labels(test_labels, 10))
def test_one():
    img = next(image_iter)
    picture = np.asarray(img).squeeze()
    plt.imshow(picture)
    plt.show()
    image = vectorize_image(img)
    a = nn.feedforward(image)
    label = next(label_iter)
    print(a)
    print(np.argmax(a))
    print(label)
    print(sum(np.vectorize(cost)(a, label)))



