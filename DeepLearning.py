import numpy as np
import matplotlib.pyplot as plt
import math
import gzip
import time

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

def change_step(step):
    if step > 0.1:
        return step - 0.003
    return step


class NeuralNet:
    def __init__(self, layers, step = 0.05, activation=sigmoid, da_dz=deriv_sigmoid):
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
        self.biases = [0.2 * np.zeros((layers[i], 1)) for i in range(1, len(layers))]
        self.step = step
        
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
        while n <= len(input_data):
            batch_in = input_data[m:n]
            batch_out = output_data[m:n]
            #avg_cost = 0
            #avg_grad = np.array([[np.zeros(self.weights[i].shape) for i in range(0, len(self.weights))], [np.zeros(self.biases[i].shape) for i in range(0, len(self.biases))]])
            count = 0
            sum_grad = np.array([[np.zeros(self.weights[i].shape) for i in range(0, len(self.weights))], [np.zeros(self.biases[i].shape) for i in range(0, len(self.biases))]])
            sum_cost = 0
            round_time = time.time()
            for input, expected in zip(batch_in, batch_out):
                count += 1
                a = self.feedforward(input)
                self.training_atts += 1
                train_count += 1
                #avg_cost = ((self.training_atts % 100 - 1) * avg_cost + sum(cost_calc(a, expected))) / (self.training_atts % 100)
                gradient = self.backprop(a, expected)
                sum_grad += gradient
                sum_cost += sum(cost_calc(a, expected))
                #avg_cost = ((count - 1) * avg_cost + sum(cost_calc(a, expected))) / (count)
                #avg_grad = 1 / (self.training_atts % 100) * ((self.training_atts % 100 - 1) * avg_grad + gradient)
                #avg_grad = 1 / (count) * ((count - 1) * avg_grad + gradient)
            avg_grad = sum_grad / count
            avg_cost = sum_cost / count
            cost_over_time.append(avg_cost)
            self.weights += -1 * self.step * avg_grad[0]
            self.biases += -1  * self.step * avg_grad[1]
            #self.step = change_step(self.step)
            m += batch_size
            n += batch_size
            print("Trained:", self.training_atts)
            curr_time = time.time()
            elapsed = curr_time - start_time
            print("Elapsed:", elapsed, "s")
            rate = batch_size / (curr_time - round_time)
            print("Rate:", rate, "images per sec")
            remaining = (len(input_data) - train_count) / rate
            print("Estimated Time remaining:", remaining, "s")
            print("Average Cost over last", batch_size, "samples:", avg_cost, "\n")
        plt.plot([batch_size * i for i in range(1, n // batch_size)], cost_over_time)
        plt.ylabel("Cost")
        plt.xlabel("Training Examples")
        plt.show()
        print(cost_over_time)

    def test(self, input_data, output_data):
        self.feedforward(input_data, output_data)
        
    def feedforward(self, input):
        """computes output of network for given input"""
        #assert type(input) is np.ndarray
        #assert input.shape == self.neurons[0].shape
        self.neurons[0] = input
        for l in range(1, len(self.neurons)):
            self.z[l] = np.dot(self.weights[l - 1], self.neurons[l - 1]) + self.biases[l - 1]
            self.neurons[l] = self.activation(self.z[l])
        return self.neurons[-1]

    def backprop(self, a, y):
        def dc_da(l, j):
            if l == len(self.neurons) - 1:
                return deriv_cost(a[j], y[j])
            return sum([self.weights[l][i, j] * self.activation_deriv(self.z[l + 1][i]) * dc_da(l + 1, i) for i in range(len(self.neurons[l + 1]))])
        grad_weights = [np.zeros(self.weights[i].shape) for i in range(0, len(self.weights))]
        grad_biases = [np.zeros(self.biases[i].shape) for i in range(0, len(self.biases))]
        for l in range(1, len(self.neurons)):
            #grad_weights.append(np.zeros(self.neurons[l], self.neurons[l - 1]))
            for j in range(len(self.weights[l - 1])):
                z_prime = self.activation_deriv(self.z[l][j])
                dc_dalj = dc_da(l, j)
                for k in range(len(self.weights[l - 1][0])):
                    grad_weights[l - 1][j, k] = self.neurons[l - 1][k] * z_prime * dc_dalj
                grad_biases[l - 1][j] = z_prime * dc_dalj
            #self.weights[l - 1] += -1 * self.step * grad_weights[l - 1]
            #grad_biases.append(np.zeros(len(self.neurons[l])).T)
            #self.biases[l - 1] += -1 * self.step * grad_biases[l - 1]
        #print("del_weights = ", grad_weights)
        #print("del_biases = ", grad_biases)
        return np.array([grad_weights, grad_biases])

def run_training(num=10000, batch=100, img_file='train-images-idx3-ubyte.gz', label_file='train-labels-idx1-ubyte.gz', size=28, choices=10):
    images = get_images(img_file, num, size)
    images = [vectorize_image(img) for img in images]
    labels = get_labels(label_file, num)
    output = convert_labels(labels, choices)
    nn.train(images, output, batch)

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
    print(np.argmax(label))
    print(sum(np.vectorize(cost)(a, label)))



