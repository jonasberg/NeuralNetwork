import numpy as np
from NeuralNetworkLayers import ComputationalLayer

class Optimizer():
    ''' Base class for typechecking purposes. '''
    pass

class SGD(Optimizer):
    '''
    A stochastic gradient descent optimizer.

    Todo:
        * Currenly does not check for whether convergence has been obtained.

    Args:
        learning_rate (float): The initial learning rate used during training
        batch_size (int): The number of observations in each batch
        max_epochs (int): The maximal number of epochs over which to train the 
            network.
    '''
    def __init__(self, learning_rate=1e-2, batch_size=64, max_epochs=50):
        self.alpha = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs

    def train(self, net, X, y, verbose=True):
        '''
        Trains a neural network.

        Args:
            net (NeuralNetwork): Network to train
            X (ndarray): Training data with which to train the network
            y (ndarray): Labels / values for each observation in X
            verbose (bool): Whether to print information about the training 
                progress or not.
        '''
        n = len(X)

        for epoch in range(self.max_epochs):
            if verbose: 
                print("Epoch number {} / {}".format(epoch, self.max_epochs))
            mean_loss = 0

            idx = np.random.permutation(n) # Index to shuffle X and y
            X_shuffled = X[idx]
            y_shuffled = y[idx]
            progress = 0
            for batch in range(0, n, self.batch_size):
                X_batch = X_shuffled[batch:batch+self.batch_size]
                y_batch = y_shuffled[batch:batch+self.batch_size]
                
                # Forward pass
                net.layers[0].forward(X_batch)

                y_hat = net.layers[-1].output
                mean_loss += net.loss(y_hat, y_batch)*self.batch_size/n
                
                # Backwards pass
                net.layers[-1].backward(net.loss_derivative(y_hat,y_batch))

                # Gradient descent
                for l in net.layers:
                    if isinstance(l, ComputationalLayer):
                        l.W -= self.alpha*l.dW
                        l.b -= self.alpha*l.db

                # Track progress in increments of 5% of epoch
                if int(20*batch/n) > progress and verbose:
                    progress = int(20*batch/n)
                    print(
                        "\r"+ # Carriage return, to overwrite previous
                        "["+
                        "="*(progress-1)+ 
                        ">" + 
                        "."*(20-progress)+
                        "] {}% of epoch.".format(5*progress)
                    , end="")

            if verbose:
                print("\rMean loss during epoch: ",mean_loss)

class MomentumSGD(Optimizer):
    '''
    A stochastic gradient descent optimizer, with momentum.

    At each iteration, each step is a linear combination of the previous step
    and a step in the new gradient direction.

    Implementation is still a work in progress.

    Args:
        beta (float): Value between 0 and 1 determining how weight should be 
            given to the previous step direction
        learning_rate (float): The initial learning rate used during training
        batch_size (int): The number of observations in each batch
        max_epochs (int): The maximal number of epochs over which to train the 
            network

    Attributes:
        Vw (list): A list of the weight updates made during the previous step 
            of the training
        Vb (list): A list of the bias updates made during the previous step of
            the training
    '''
    def __init__(self, beta=0.8, learning_rate=1e-2, batch_size=64, max_epochs=50):
        self.beta = beta
        self.alpha = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs

        self.Vw = None # Last update of weights
        self.Vb = None # Last update of biases

    def train(self, net, X, y, verbose=True):
        '''
        Trains a neural network.

        Args:
            net (NeuralNetwork): Network to train
            X (ndarray): Training data with which to train the network
            y (ndarray): Labels / values for each observation in X
            verbose (bool): Whether to print information about the training 
                progress or not.
        '''
        n = len(X)

        for epoch in range(self.max_epochs):
            if verbose: 
                print("Epoch number {} / {}".format(epoch, self.max_epochs))
            mean_loss = 0

            idx = np.random.permutation(n) # Index to shuffle X and y
            X_shuffled = X[idx]
            y_shuffled = y[idx]
            progress = 0
            for batch in range(0, n, self.batch_size):
                X_batch = X_shuffled[batch:batch+self.batch_size]
                y_batch = y_shuffled[batch:batch+self.batch_size]

                # Forward pass
                net.layers[0].forward(X_batch)
                y_hat = net.layers[-1].output
                mean_loss += net.loss(y_hat, y_batch)*self.batch_size/n
                
                # Backwards pass
                net.layers[-1].backward(net.loss_derivative(y_hat,y_batch))

                trainable_layers = [l for l in net.layers if isinstance(l, ComputationalLayer)]

                if type(self.Vw) == type(None): # If first pass, initialize
                    self.Vw = [np.zeros_like(l.dW) for l in trainable_layers]
                    self.Vb = [np.zeros_like(l.db) for l in trainable_layers]

                # Gradient descent
                for i, l in enumerate(trainable_layers):
                    Vw_prev = self.Vw[i]
                    Vb_prev = self.Vb[i]

                    Vw_new = self.beta*Vw_prev + self.alpha*l.dW
                    Vb_new = self.beta*Vb_prev + self.alpha*l.db

                    self.Vw[i] = Vw_new
                    self.Vb[i] = Vb_new

                    l.W -= Vw_new
                    l.b -= Vb_new
                
                # Track progress in increments of 5% of epoch
                if int(20*batch/n) > progress and verbose:
                    progress = int(20*batch/n)
                    print(
                        "\r"+ # Carriage return, to overwrite previous
                        "["+
                        "="*(progress-1)+ 
                        ">" + 
                        "."*(20-progress)+
                        "] {}% of epoch.".format(5*progress)
                    , end="")

            if verbose:
                print("\rMean loss during epoch: ",mean_loss)


