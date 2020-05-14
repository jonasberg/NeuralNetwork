from NeuralNetworkLayers import *
from NeuralNetworkOptimizers import *

class NeuralNetwork():
    '''
    An implementation of a neural network.

    Quite basic in it's structure, mainly aimed at being an educational 
    resource rather than an implementation to be used in real applications.

    Args:
        input_shape (tuple): Shape of the input data. One dimensional if the
            first layer is dense, three dimensional (channels, height, width)
            if using a convolutional first layer
        n_output_nodes (int): The number of output nodes. One if using the
            network for regression or binary classification, otherwise same as 
            the number of classes present in the data
        loss (str): The loss to be used. Can be either cross entropy 
            ("cross_entropy") or sum of square errors ("sse")
        verbose (bool): If true, the network will e.g. print progress during 
            training

    Attributes:
        layers (list): list of the layers comprising the network
        is_setup (bool): flag indicating whether the self.setup has been called
        optimizer (Optimizer): Optimizer object used during training, set in
            the method self.compile
    '''
    def __init__(self, input_shape, n_output_nodes, loss="cross_entropy", verbose=False):
        if type(input_shape) == int:
            self.input_shape = [input_shape,]
        else: # Multi-dimensional input, e.g. image
            self.input_shape = input_shape

        self.n_output_nodes = n_output_nodes
        self.layers = []
        self.is_setup = False
        self.optimizer = None
        self.verbose = verbose

        # Loss 
        if loss == "cross_entropy":
            self.loss = self.cross_entropy_loss
            self.loss_derivative = self.cross_entropy_loss_derivative
        elif loss == "sse":
            self.loss = self.SSE_loss
            self.loss_derivative = self.SSE_loss_derivative
        else:
            raise Exception("Loss function '{}' not known.".format(loss))
        
    def add_dense_layer(self, n_nodes, activation="relu", dropout=False, dropout_rate=0.5):
        '''
        Adds a dense layer and one activation layer to the network.

        A wrapper to make adding a layer to the network easier. Verifies that
        the output of the previous layer is one dimensional and ensures the 
        new layer has appropriate input shape (matching the output shape of 
        the previous layer). 

        If the argument dropout is set to True, then a dropout layer will also
        be added after the activation layer.

        Args:
            n_nodes (int): The number of nodes in the dense layer
            activation (str): Activation function of the layer; either "none", 
                "relu" or "softmax"
            dropout (bool): Flag for whether to include a dropout layer
            dropout_rate (float): The dropout rate for the dropout layer

        Returns:
            None
        '''
        if activation == "relu":
            activation_class = ReLU
        elif activation == "softmax":
            activation_class = Softmax
        elif activation == "none":
            activation_class = None
        else:
            raise Exception("Unknown activation function.")

        if len(self.layers) == 0: 
            # Verify that the input is one dimensional
            if len(self.input_shape) != 1:
                raise Exception("Invalid input shape for a dense starting layer.")
            layer = DenseLayer(self.input_shape[0], n_nodes)
        else: 
            # Verify that the output of last layer is one dimensional
            if len(self.layers[-1].output_shape) != 1:
                raise Exception("Invalid input shape for a dense starting layer.")

            layer = DenseLayer(self.layers[-1].output_shape[0], n_nodes)
        
        self.layers.append(layer)

        if activation_class != None:
            self.layers.append(activation_class(layer.output_shape))
        if dropout:
            self.layers.append(Dropout([n_nodes,], dropout_rate=dropout_rate))


    def add_convolutional_layer(self, n_filters, filter_size=3, stride=1, activation="relu"):
        '''
        Adds a convolutional layer and one activation layer to the network.

        A wrapper to make adding a convolutional layer to the network easier.
        Verifies that the output of the previous layer is three dimensional and
        ensures the new layer has appropriate input shape (matching the output 
        shape of the previous layer). 

        Args:
            n_filters (int): The number of nodes/filters in the layer
            filter_size (int): The size of the filters
            stride (int): The stride used in the convolution
            activation (str): Activation function of the layer; currently only 
                "relu" is supported

        Returns:
            None
        '''

        if activation == "relu":
            activation_class = ReLU
        else: # Only relu supported at the moment
            raise Exception("Unknown activation function.")

        if len(self.layers) == 0:
            # Verify that the input is three dimensional
            if len(self.input_shape) != 3:
                raise Exception("Invalid input shape for a convolutional starting layer.")
            layer = ConvolutionalLayer(n_filters, filter_size, stride, input_shape=self.input_shape)
            
        else:
            # Verify that the input is three dimensional
            if len(self.layers[-1].output_shape) != 3:
                raise Exception("Invalid input shape for a convolutional starting layer.")
            prev_shape = self.layers[-1].output_shape
            layer = ConvolutionalLayer(n_filters, filter_size, stride, input_shape=prev_shape)

        self.layers.append(layer)
        self.layers.append(activation_class(layer.output_shape))
        
    def add_flatten_layer(self):
        '''
        Adds a flatten layer to the network.

        Ususally works as a bridge from a convolutional set of layers in
        the beginning of the network to a section containing dense layers at
        the end of the network.

        Returns:
            None
        '''
        input_shape = self.layers[-1].output_shape
        layer = Flatten(input_shape)
        self.layers.append(layer)

    def setup(self, output_activation="softmax"):
        '''
        Adds output layer and links the layers in self.layers.

        Should always be called before using the network. However, if 
        self.setup is False it will be called automatically by self.compile.
        
        The layers rely on a doubly linked list structure, and those references
        to neighbouring layers are set by calling this method. 

        Args:
            output_activation (str): Specifies which activation should be
                used for the output layer. Can use any of the activations
                supported by self.add_dense_layer.

        Returns:
            None
        '''
        # Add output layer
        self.add_dense_layer(self.n_output_nodes, activation = output_activation)

        # Complete the "linked list" structure of layers
        for i in range(1, len(self.layers)-1):
            l = self.layers[i]
            l.prev_layer = self.layers[i-1]
            l.next_layer = self.layers[i+1]

            self.layers[0].next_layer = self.layers[1]
            self.layers[-1].prev_layer = self.layers[-2]

        self.is_setup = True

    def compile(self, optimizer="sgd", batch_size=32, max_epochs=10, learning_rate=1e-2):
        '''
        Performs setup operations and sets an optimizer to use for training.

        Args:
            optimizer (Optimizer): Can either be an initialized optimizer 
                inheriting from the Optimizer base class, or "sgd" in which 
                case this method will initialize an SGD optimizer.
            batch_size (int): The batch size to use (in case optimizer is "sgd")
            max_epochs (int): The max number of epochs allowed (in case 
                optimizer is "sgd")
            learning_rate (float): The initial learning rate to use (in case 
                optimizer is "sgd")

        Returns:
            None
        '''
        if not self.is_setup:
            self.setup()
        
        if optimizer == "sgd":
            self.optimizer = SGD(
                batch_size = batch_size, 
                max_epochs = max_epochs,
                learning_rate = learning_rate
            )
            
        elif isinstance(optimizer, Optimizer):
            self.optimizer = optimizer

        else:
            raise ValueError('Specified optimizer not recognized.')


    def fit(self, X, y):
        '''
        Trains the network given the training data X and y.

        Args:
            X (ndarray): Training data, where the first dimension correspond to 
                separate observations and subsequent dimensions contain 
                features for the observation. Must be either two dimensional
                if the first layer is dense, or four dimensional if the first
                layer is convolutional.
            
            y (ndarray): Labels / values for each observation in X.

        Returns:
            None
        '''

        self.optimizer.train(self, X, y, verbose=self.verbose)
    
    def predict(self, X):
        '''
        Predicts the labels / values for the observations in X.

        Performs a forward pass and reutrns the output of the last layer in the
        network. Before the pass all dropout layers are deactivated, and they  
        are activated again after the prediction has been computed.

        Args:
            X (ndarray): The observations for which predictions should be made.

        Returns:
            yhat (ndarray): The predictions made by the network.
        '''

        # Set dropout layers to predict-mode
        for l in self.layers:
            if isinstance(l, Dropout):
                l.predict = True 

        # Predict
        self.layers[0].forward(X)
        yhat = self.layers[-1].output.copy()

        # Reset dropout layers
        for l in self.layers:
            if isinstance(l, Dropout):
                l.predict = False 

        return yhat

    def SSE_loss(self, yhat, y):
        '''
        Computes the sum of squared errors loss.

        Most commonly used for regression. If used for multiclass predictions,
        note that the labels must be one-hot encoded.

        Args:
            yhat (ndarray): Predicted y-values
            y (ndarray): True values

        Returns:
            Sum of squared errors
        '''
        return np.sum((yhat-y)**2)

    def SSE_loss_derivative(self, yhat, y):
        '''
        Computes the gradient of the sum of squares loss with respect to yhat.

        Args:
            yhat (ndarray): Predicted y-values
            y (ndarray): True values

        Returns:
            The gradient of SSE_loss w.r.t. the network output
        '''
        return -2*y+2*yhat

    def cross_entropy_loss(self, yhat, y):
        '''
        Computes the cross entropy loss.

        Most commonly used for classification tasks. Note that the labels 
        must be one-hot encoded.

        Args:
            yhat (ndarray): Predicted y-values
            y (ndarray): True values

        Returns:
            Cross entropy loss
        '''
        return -np.sum(y*np.log(yhat+1e-20))

    def cross_entropy_loss_derivative(self, yhat, y):
        '''
        Computes the gradient of the cross entropy loss with respect to yhat.

        Args:
            yhat (ndarray): Predicted y-values
            y (ndarray): True values

        Returns:
            The gradient of cross_entropy_loss w.r.t. the network output
        '''

        sparse_matrix = y*yhat
        idx = (sparse_matrix != 0)
        sparse_matrix[idx] = 1/sparse_matrix[idx]
        return -sparse_matrix

    def n_weights(self):
        ''' Returns the number of trainable weights in the network.'''
        n = 0
        for l in self.layers:
            if (l.__class__.__name__ == "DenseLayer"):
                n += l.W.size
                n += l.b.size
        return n

    def __str__(self):
        ''' Returns a string summarizing the network architecture. '''
        s = "="*80+"\n"
        s += "{:<30}{:<20}{:<20}{:<10}\n".format(
            "Layer type", 
            "Input shape", 
            "Output shape", 
            "# Params"
        )
        s += "="*80+"\n"
        for i, l in enumerate(self.layers):
            s += l.__str__()        
            s += "_"*80+"\n"
            
        s += "Number of weights: {}".format(self.n_weights())
        return s
        