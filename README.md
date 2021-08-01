# NeuralNetworks
Using Python to experiment with deep learning and neural networks, without automatic differentiation.

While some API design is inspired by PyTorch, the implementations are my own.

### Classes
`nn.NeuralNet` The neural network class. Supports fully-connected DNNs.  
`optimizer.Optimizer` Performs batch gradient descent to minimize a loss function for a `NeuralNet`. Computes the gradients using the backpropagation algorithm and updates the network weights with optional momentum and weight decay.  
`dataloader.Dataloader` Iterable over data used for training or testing.  
`trainer.Trainer` Training interface takes a `NeuralNet`, `Optimizer`, and `Dataloader`s to train the NN.

### Modules
`functions.activations` ReLU, sigmoid, hardtanh, softmax, and linear activations  
`functions.loss` Squared and softmax cross-entropy loss  

### Usage
`python3 train.py --cfg configs/mnist_classifier.yaml`  
Trains a classifier on the MNIST handwritten digits dataset. Reaches 96.5% accuracy on the test set in 20 epochs.


### Dependencies
The only dependencies are NumPy and YACS.