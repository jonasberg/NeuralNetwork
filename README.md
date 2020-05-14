# Neural Network 

This is a project written for the purpose of learning the mathematical details of neural networks, including how to form predictions, backpropagation and optimization of the network. The code is functional, but by no means optimized and should not be used for any other purposes than educational ones.

## Overview

* **NeuralNetork.py** - contains the `NeuralNetwork` class, which controls adding layers, training, making predictions etc. 

*  **NeuralNetworkLayers.py** - contains the layers that can be added to the network, e.g. dense, convolutional, dropout, activations and so forth. These are usually added by invoking methods  of `NeuralNetwork` objects, but one can look at this file to get a deeper understanding of how they work. Also contains test cases to numerically check that the derivatives of the classes `DenseLayer` and `ConvolutionalLayer` are correct.
* **NeuralNetworkOptimizers.py**  - contains optimizers to train the network. Currently, only stochastic gradient descent is implemented, but stochastic gradient descent with momentum is in the workings.

## Known issues / Further improvements

**Improvements**

* Implement training history of optimizers to be able to visualize the training progress of the models and compare the performance of different optimizers.

**Issues**

* Testing the network on one dimensional functions (e.g. ![equation](https://latex.codecogs.com/gif.latex?sin(x)+x,\%20\%20x\in\mathbb{R})) indicate that the biases of the dense network layers are updating slowly, leading to poor approximation of non-linear functions. This is obviously a large issue (since then the networks will always simply perform linear or logistic regression), which should be investigated further.

