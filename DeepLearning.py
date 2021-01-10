import numpy as np
import matplotlib.pyplot as plt
import math
from math import floor
from math import sqrt
from NeuralNet import *
import argparse
from math_functions import *
from utils import *


# def grad_size(grad):
#     total = 0
#     for side in grad:
#         for layer in side:
#             for row in layer:
#                 for val in row:
#                     total += val ** 2
#     return sqrt(total)

# def get_time_based(d, lr0, r):
#     def time_based(lr, n):
#         return lr / (1 + d * n)
#     return time_based

# def get_step_based(d, lr0, r):
#     def step_based(lr, n):
#         return lr0 * d ** floor((1 + n) / r)
#     return step_based

# def get_exponential(d, lr0, r):
#     def exponential(lr, n):
#         return lr0 * np.exp(d * n)
#     return exponential



def run_training(network, data, labels, batch, hold_out):
    images = list(vectorize_images(data))
    output = list(convert_labels(labels, network.layers[-1]))
    network.train(images, output, batch, hold_out)

# def run_testing():
#     cur_correct = nn.test_correct
#     cur_atts = nn.test_atts
#     for img, actual in zip(iter(test_images), iter(test_labels)):
#         img = vectorize_image(img)
#         nn.test(img, actual)
#     print(f'Accuracy: {((nn.test_correct - cur_correct) / (nn.test_atts - cur_atts) *100):.3f}%\n')

# def test_one():
#     img = next(image_iter)
#     picture = np.asarray(img).squeeze()
#     plt.imshow(picture)
#     plt.show()
#     image = vectorize_image(img)
#     a = nn.feedforward(image)
#     label = next(label_iter)
#     print("Output Vector: \n", a)
#     print("Selection:", selection(a))
#     print("Actual:", np.argmax(label))
#     print("Error:", sum(nn.cost_calc(a, label)))


# def selection(a):
#     """Returns the selection with minimum error"""
#     I = np.identity(len(a))
#     costs = [sum(nn.cost_calc(a, I[:,[i]])) for i in range(len(a))]
#     return np.argmin(costs)

def main(args):
    layers = args['layers']
    activation = eval(args['g'])
    use_softmax = args['softmax']
    alpha = args['alpha']
    loss = eval(args['loss'] + '_loss')
    error = args['error_rate']
    nn = NeuralNet(layers, alpha, activation, use_softmax, loss, error)

    in_size = args['input_dim']
    training_files = (args['train_data'], args['train_labels'])
    testing_files = (args['test_data'], args['test_labels'])
    num_training = args['num_train']
    num_testing = args['num_test']
    train_data = read_data(training_files[0], num_training, in_size[0], in_size[1])
    train_labels = read_labels(training_files[1], num_training)
    
    test_data = read_data(testing_files[0], num_testing, in_size[0], in_size[1])
    
    validation = args['validation']
    hold_out = int(num_training * validation)
    
    batch = args['batch']

    run_training(nn, train_data, train_labels, batch, hold_out)

def create_parser():
    parser = argparse.ArgumentParser(description='Deep Learning')
    parser.add_argument('--layers', nargs='*', type=int, default=[784, 16, 16, 10], help='Specify neurons in each layer of network')
    parser.add_argument('--g', type=str, choices=['ReLU', 'sigmoid'],  default='ReLU', help='Specify activation function')
    parser.add_argument('--softmax', action='store_true', help='Use softmax activation for output layer')
    parser.add_argument('--loss', type=str, default='squared', choices=['squared', 'logistic'], help='Specify loss function')
    parser.add_argument('--alpha', type=float, default=0.05, help='Specify learning rate')
    parser.add_argument('--input-dim', nargs='+', default=[28,28], type=int, help='Specify dimensions of input data')
    parser.add_argument('--train-data', type=str, default='train-images-idx3-ubyte.gz', help='File containing training data')
    parser.add_argument('--train-labels', type=str, default='train-labels-idx1-ubyte.gz', help='File containing training labels')
    parser.add_argument('--num-train', type=int, default=60000, help='Number of training examples')
    parser.add_argument('--num-test', type=int, default=10000, help='Number of testing examples')
    parser.add_argument('--test-data', type=str, default='t10k-images-idx3-ubyte.gz', help='File containing testing data')
    parser.add_argument('--test-labels', type=str, default='t10k-labels-idx1-ubyte.gz', help='File containing testing data')
    parser.add_argument('--batch', type=int, default=100, help='Specify batch size')
    parser.add_argument('--validation', type=float, default=0.1, help='Fraction of training data to hold out')
    parser.add_argument('--error-rate', type=float, default=0.1, help='Minimum error rate to stop training')
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(vars(args))

