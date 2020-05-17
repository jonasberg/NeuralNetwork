# Neural Network 

This is a project written for the purpose of learning the mathematical details of neural networks, including how to form predictions, use backpropagation and optimize the network. The code is functional, but by no means computationally optimized and should not be used for any other purposes than educational ones.

## Demo

There are two Google Colab notebooks illustrating how to use the code to either build an ordinary neural networks and apply them to the MNIST dataset. The first uses an ordinary neural network, and the second a convolutional one. The links are read-only, but by clicking on "Open in playground" you can make a copy of the notebook and try the code out for yourself.

**Neural Network:** https://colab.research.google.com/drive/1wCnFfZLO-nKBR3ch3T88-yC_UioEcHS_?usp=sharing

**Convolutional network:** https://colab.research.google.com/drive/1Eljq-i-aUj-A8NX3aMZX7VH152bnumEM?usp=sharing

## Overview

* **NeuralNetork.py** - contains the `NeuralNetwork` class, which controls adding layers, training, making predictions etc. 

*  **NeuralNetworkLayers.py** - contains the layers that can be added to the network, e.g. dense, convolutional, dropout, activations and so forth. These are usually added by invoking methods  of instances of the class `NeuralNetwork`, but one can look at this file to get a deeper understanding of how they work. Also contains test cases to numerically check that the derivatives of the classes `DenseLayer` and `ConvolutionalLayer` are correct.
* **NeuralNetworkOptimizers.py**  - contains optimizers to train the network. Currently, only stochastic gradient descent and stochastic gradient descent with momentum are implemented.

## Known issues / Further improvements

**Improvements**

* Implement training history of optimizers to be able to visualize the training progress of the models and compare the performance of different optimizers.

**Issues**

* Testing the network on one dimensional functions (e.g. ![equation](https://latex.codecogs.com/gif.latex?sin(x)+x,\%20\%20x\in\mathbb{R})) indicate that the biases of the dense network layers are updating slowly, leading to poor approximation of non-linear functions. This is obviously a large issue (since then the networks will always simply perform linear or logistic regression), which should be investigated further.

