from NeuralNetworkLayers import *
from NeuralNetworkOptimizers import *

class NeuralNetwork():
    def __init__(self, input_shape, n_output_nodes, loss="cross_entropy", verbose=False):
        # General params
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
        input_shape = self.layers[-1].output_shape
        layer = Flatten(input_shape)
        self.layers.append(layer)

    def setup(self, output_activation="softmax"):
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


    def fit(self, X, y, batch_size=32):
        if len(X.shape) == 2: # Data is on matrix format
            n,d = X.shape
        elif len(X.shape) == 4: # Data is on image format
            n,c,h,w = X.shape
        else:
            raise Exception("Unknown input dimension")

        self.optimizer.train(self, X, y, verbose=self.verbose)
    
    def predict(self, X):
        # Set dropout layers to predict-mode
        for l in self.layers:
            if (l.__class__.__name__ == "Dropout"):
                l.predict = True 

        # Predict
        self.layers[0].forward(X)
        yhat = self.layers[-1].output.copy()

        # Reset dropout layers
        for l in self.layers:
            if (l.__class__.__name__ == "Dropout"):
                l.predict = False 

        # Return appropriate output depending on whether the network performs 
        # regression or classification.
        if self.loss == self.cross_entropy_loss:
            return np.argmax(yhat, axis=1)
        elif self.loss == self.SSE_loss:
            return yhat

    def SSE_loss(self, yhat, y):
        return np.sum((yhat-y)**2)

    def SSE_loss_derivative(self, yhat, y):
        return -2*y+2*yhat

    def cross_entropy_loss(self, yhat, y):
        return -np.sum(y*np.log(yhat+1e-20))

    def cross_entropy_loss_derivative(self, yhat, y):
        sparse_matrix = y*yhat
        idx = (sparse_matrix != 0)
        sparse_matrix[idx] = 1/sparse_matrix[idx]
        return -sparse_matrix

    def n_weights(self):
        n = 0
        for l in self.layers:
            if (l.__class__.__name__ == "DenseLayer"):
                n += l.W.size
                n += l.b.size
        return n

    def __str__(self):
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
        