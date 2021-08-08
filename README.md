# NeuralNetworks
Using Python to experiment with deep learning and neural networks, without automatic differentiation.

While some API design is inspired by PyTorch, the implementations are my own.

### Classes
`nn.NeuralNet` The neural network class. Supports fully-connected DNNs.  
`optimizer.Optimizer` Performs batch gradient descent to minimize a loss function for a `NeuralNet`. Computes the gradients using the backpropagation algorithm and updates the network weights with optional momentum and weight decay.  
`dataloader.Dataloader` Iterable over data used for training or testing.  
`trainer.Trainer` Training interface takes a `NeuralNet`, `Optimizer`, and training and validation `Dataloader`s to train the NN.

### Modules
`functions.activations` ReLU, sigmoid, hardtanh, softmax, and linear activations  
`functions.loss` Squared and softmax cross-entropy loss  

### Usage
`python3 deeplearning.py --cfg configs/mnist_classifier.yaml`  
Trains a classifier on the MNIST handwritten digits dataset.  
#### Results
Best accuracy for 2 hidden-layer network in 40 epochs: 97.65%  
128, 64 neurons with ReLU activations, LR 0.01, Momentum 0.2

<!-- | Epochs Trained  | Test Set Accuracy (%) |
|---   |---|
| 1 | 87.10 |
| 5 | 93.50 |
| 10 | 95.27 |
| 20 | 96.51 |
| 50 | 97.22 |
| 100 | 97.32 | -->


### Todo
- Learning rate scheduling


### Dependencies
NumPy, YACS.
