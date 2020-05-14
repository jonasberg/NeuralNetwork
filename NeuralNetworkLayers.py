
import numpy as np

class Layer():
    '''
    Base class specifying the common attributes and methods for layers. 
    
    It serves both as an interface when implementing new layers, but also what 
    attributes can be accessed when interacting with layers in an application.

    Attributes:
        prev_layer (Layer): A reference to the previous layer in the network
        next_layer (Layer): A reference to the next layer in the network
        output_shape (tuple): A tuple of integers specifying the shape the output
        output (ndarray): A ndarray resulting from a forward pass of the layer
    '''
    def __init__(self):
        # Allowing for a linked list type of structure for the network
        self.prev_layer = None
        self.next_layer = None

        # Must be specified for all layers
        self.output_shape = None

        # Save output of all layers; needed for backward pass
        self.output = None

    def forward(self, X):
        '''
        Performs a forward pass, using X as input.

        Will also attempt to call forward on the next_layer with the output of
        the layer as input argument, if next_layer is not None.

        Args:
            X (ndarray): An ndarray with input data

        Returns:
            None
        '''
        pass

    def backward(self, output_derivative):
        '''
        Performs a backward pass.

        Will also attempt to call backward on the prev_layer with the 
        derivative with respect to the input of the layer as argument, in case
        prev_layer is not None.

        Args:
            output_derivative (ndarray): 
                An ndarray with the derivative of the loss function with 
                respect to the output of the current layer (elementwise)

        Returns:
            None
        '''
        pass

    def __str__(self):
        '''
        A string giving a description of the layer.

        To enable a structured string representation of a whole network, this
        string should follow the following structure:

        30 characters: String specifying layer type
        20 characters: String specifying the input shape expected by the layer
        20 characters: String specifying the output shape of the layer 
        10 characters: String specifying the # of trainable parameters
        
        These should be left aligned. This can be achieved by using format
        with the following template:

        "{:<30}{:<20}{:<20}{:<10}".format(...)
        '''
        pass

class ComputationalLayer(Layer):
    '''
    Subclass used for computational layers, e.g. dense or convolutional layers.

    This is to allow for easy filtering out relevant layers when updating 
    weights during optimization.

    Attributes:
        W (ndarray): Contains the layer weights
        b (ndarray): Contains the layer biases
        dW (ndarray): The derivative with respect to each weight in W
        db (ndarray): The derivative with respect to each bias in b
    '''
    def __init__(self):
        super().__init__()

        self.W = None
        self.b = None
        self.dW = None
        self.db = None

# -------------------------------- Computational layers ------------------------
class DenseLayer(ComputationalLayer):
    '''
    A fully connected neural network layer.

    The most basic building block for neural networks. Each input node is 
    connected to each output node. 

    Args:
        in_features (int): The number of input nodes
        n_nodes (int): The number of nodes in the layer (output nodes)
    '''
    def __init__(self, in_features, n_nodes):
        super().__init__()

        self.in_features = in_features
        self.n_nodes = n_nodes
        self.output_shape = [n_nodes,]

        # Using Kaiming initialization.
        # See article below for an excelent summary of why this is a good idea.
        # https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79
        self.W = np.random.normal(size=(in_features, n_nodes))*np.sqrt(2/in_features) 
        self.b = np.zeros((1, n_nodes))

    def forward(self, X):
        # Save input if first layer, needed for backpropagation.
        if self.prev_layer == None:
            self.input = X

        self.output = X @ self.W + self.b

        if self.next_layer != None:
            self.next_layer.forward(self.output)

    def backward(self, output_derivative):
        n,_ = output_derivative.shape
        
        if self.prev_layer == None:
            input_ = self.input
        else:
            input_ = self.prev_layer.output
        
        self.db = np.sum(output_derivative, axis=0) / n
        self.dW = (input_.T @ output_derivative) / n

        if self.prev_layer != None:
            derivative = output_derivative @ self.W.T
            self.prev_layer.backward(derivative)

    def __str__(self):
        return "{:<30}{:<20}{:<20}{:<10} \n".format(
            "Dense layer",
            "("+str(self.in_features)+",)", 
            "("+str(self.n_nodes)+",)",
            self.W.size + self.b.size
        )

class ConvolutionalLayer(ComputationalLayer): 
    '''
    A convolutional neural network layer.

    Performs a convolution between the weight matrix (filter) of each node and 
    the input data in the forward pass, yielding an activation map for each 
    node. These activation maps are stacked to form the output.

    Implementation is based on Alejandro Escontrela's Medium article:
    https://towardsdatascience.com/convolutional-neural-networks-from-the-ground-up-c67bb41454e1

    Args:
        n_filters (int): The number of nodes / filters in the layer
        filter_size (int): The size of the filters; typically 3.
        stride (int): The stride used in the convolution
        input_shape (tuple): The shape of a single input observation, must be
            3-dimensional and follow the convention: (channels, height, width).
    '''

    def __init__(self, n_filters, filter_size=3, stride=1, input_shape=(1,28,28)): # in_shape, 
        super().__init__()

        c,h,w = input_shape

        self.n_filters = n_filters
        self.filter_size = filter_size
        self.stride = stride

        self.W = np.random.normal(size=(
            n_filters, 
            c, 
            filter_size, 
            filter_size
        ))*np.sqrt(2/(n_filters*c*filter_size**2)) # Correct initialization?
        self.b = np.zeros((n_filters))

        output_w = (w - filter_size)//stride + 1
        output_h = (h - filter_size)//stride + 1
        self.input_shape = input_shape
        self.output_shape = (n_filters, output_h, output_w)

    def forward(self, image_batch):
        if len(image_batch.shape) == 4:
            pass
        elif len(image_batch.shape) == 3:
            c, h, w = image_batch.shape
            n = 1
            image_batch = image_batch.copy()
            image_batch = image_batch.reshape((n,c,h,w))
        else:
            raise Exception("Unknown dimension of image batch.")
        
        # Save input if first layer
        if self.prev_layer == None:
            self.input = image_batch

        n, c, h, w = image_batch.shape

        _, output_w, output_h = self.output_shape

        self.output = np.zeros((n, self.n_filters, output_h, output_w))

        # Convolve
        f = self.filter_size
        curr_y = out_y = 0
        while curr_y + f <= h:
            curr_x = out_x = 0

            while curr_x + f <= w:
                im_slice = image_batch[:,:,curr_y:curr_y+f, curr_x:curr_x+f]

                self.output[:,:, out_y, out_x] = np.sum(
                    im_slice[:,None,:,:,:]*self.W[None,:,:,:,:],
                    axis = (2,3,4)
                ) + self.b.T
                curr_x += self.stride
                out_x += 1

            curr_y += self.stride
            out_y += 1

        if self.next_layer != None:
            self.next_layer.forward(self.output)
        
    def backward(self, output_derivative):

        if self.prev_layer == None:
            input_ = self.input
        else:
            input_ = self.prev_layer.output

        (n,c,h,w) = input_.shape
        
        # Initialize derivative matrices
        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)
        derivative = np.zeros(input_.shape)

        # Loop
        f = self.filter_size
        curr_y = out_y = 0
        while curr_y + f <= h:
            curr_x = out_x = 0

            while curr_x + f <= w:
                # Update filter derivative
                output_derivative_term = output_derivative[:, :, out_y, out_x].reshape(
                    (n,self.n_filters,1,1)
                )
                input_term = input_[:,:,curr_y:curr_y + f, curr_x: curr_x + f] # Shape (N x C x F x F)
                mean_dW_term = np.mean(
                    output_derivative_term[:,:,None]*input_term[:,None,:,:,:]
                ,axis=0) # Shape (N_Filters x C x K x K)
                
                self.dW += mean_dW_term
                
                # Update derivative of input 
                derivative[:, :, curr_y:curr_y + f, curr_x:curr_x + f] += np.sum(
                    output_derivative_term[:,:,None,:,:] * self.W[None, :,:,:,:]
                , axis=1) # Sum over all filters in W
                curr_x += self.stride
                out_x += 1

            curr_y += self.stride
            out_y += 1
        self.db = np.mean(np.sum(output_derivative, axis=(2,3)), axis=0)
        
        if self.prev_layer != None:
            self.prev_layer.backward(derivative)
    
    def __str__(self):
        return "{:<30}{:<20}{:<20}{:<10} \n".format(
            "Convolutional layer",
            "("+ ", ".join(map(str, self.input_shape))+")",
            "("+ ", ".join(map(str, self.output_shape))+")",
            self.W.size + self.b.size
        )
# ------------------------------------------------------------------------------


# -------------------------------- Activations ---------------------------------
class ReLU(Layer):
    '''
    Applies ReLU activation elementwise to the input data.

    ReLU activation is defined as f(x) = max(0, x).

    Args:
        output_shape (tuple): Specifying the output dimension of the layer. Same
            as the output of the previous layer.
    '''
    def __init__(self, output_shape):
        super().__init__()
        self.output_shape = output_shape

    def forward(self, X):
        out = X.copy()
        out[out<0] = 0
        self.output = out

        if self.next_layer != None:
            self.next_layer.forward(self.output)

    def backward(self, output_derivative):
        derivative = output_derivative.copy()
        derivative[self.output == 0] = 0

        if self.prev_layer != None:
            self.prev_layer.backward(derivative)

    def __str__(self):
        return "{:<30}{:<20}{:<20}{:<10}\n".format(
            "ReLU activation", 
            "("+ ", ".join(map(str, self.output_shape))+")",
            "("+ ", ".join(map(str, self.output_shape))+")",
            "0"
        )

class Softmax(Layer):
    '''
    Applies Softmax activation to the input data.

    For explaination of the matematics of this activation, see Stephen Oman's
    article on the topic: 
    https://aimatters.wordpress.com/2019/06/17/the-softmax-function-derivative/

    Args:
        output_shape (tuple): Specifying the output dimension of the layer. Same
            as the output of the previous layer.
    '''

    def __init__(self, output_shape):
        super().__init__()
        self.output_shape = output_shape

    def forward(self,  X):
        X = X.copy()
        n,d = X.shape

        # Numerically stable computation of softmax
        row_max = np.max(X, axis=1).reshape(n,1)
        log_term = np.log(np.sum(np.exp(X-row_max),axis=1)).reshape(n,1)
        log_cross_entropy = X - row_max - log_term

        self.output = np.exp(log_cross_entropy)

        if self.next_layer != None:
            self.next_layer.forward(self.output)

    def backward(self,output_derivative):
        # Elementwise multiply next_layer_derivative with output
        out_and_out_grad = self.output*output_derivative
        
        # Create a matrix of dimension NxDxD, where each DxD layer is the outer
        # product between a row of self.output and itself.
        outer_matrix = self.output[:,:, None] * out_and_out_grad[:,None]
        # Sum along one of the axes (not important which due to symmetry)
        outer_matrix_summed = np.sum(outer_matrix, axis=2)
        
        # Derivative of loss with respect to input
        derivative = out_and_out_grad - outer_matrix_summed

        if self.prev_layer != None:
            self.prev_layer.backward(derivative)

    def __str__(self):
        return "{:<30}{:<20}{:<20}{:<10}\n".format(
            "Softmax activation", 
            "("+ ", ".join(map(str, self.output_shape))+")",
            "("+ ", ".join(map(str, self.output_shape))+")",
            "0"
        )
# ------------------------------------------------------------------------------

# -------------------------------- Other layers --------------------------------
class Dropout(Layer):
    '''
    Randomly sets a fraction of the input values to 0 to reduce overfitting.

    Effectively turns the network into an ensemble model of smaller networks,
    by training subsets of nodes separately. The final prediction then becomes
    an average of all these individual networks. Thus the capacity of the 
    architecture can be increased with reduced risk of overfitting.

    Args:
        output_shape (tuple): Specifying the output dimension of the layer. Same
            as the output of the previous layer.
        dropout_rate (float): The fraction of nodes that should be deactivated.
            If 0, the layer does nothing. If 1, all output values will be 0.
    '''
    def __init__(self, output_shape, dropout_rate=0.5):
        super().__init__()
        
        self.random_matrix = None
        self.predict = False

        self.output_shape = output_shape
        self.dropout_rate = dropout_rate

    def forward(self, X):
        # Do no dropout when predicting
        if self.predict:
            dropout_rate = 0
        else:
            dropout_rate = self.dropout_rate

        # Sets a new random matrix with each forward run
        n,d = X.shape
        M = np.random.rand(n,d)
        M[M < dropout_rate] = 0
        M[M >= dropout_rate] = 1
        self.random_matrix = M
        self.output = X*M
        
        if self.next_layer != None:
            self.next_layer.forward(self.output)
        
    def backward(self, output_derivative):
        derivative = self.random_matrix * output_derivative 

        if self.prev_layer != None:
            self.prev_layer.backward(derivative)

    def __str__(self):
        return "{:<30}{:<20}{:<20}{:<10}\n".format(
            "Dropout layer", 
            "("+ ", ".join(map(str, self.output_shape))+")",
            "("+ ", ".join(map(str, self.output_shape))+")",
            "0"
        )

class Flatten(Layer):
    '''
    Flattens the input into a one dimensional object.

    Commonly used to convert the output from a convolutional layer into valid
    input to a dense layer.

    Args:
        input_shape (tuple): Shape of the input.
    '''
    def __init__(self, input_shape):
        super().__init__()

        (c,h,w) = input_shape
        self.input_shape = input_shape
        self.output_shape = [c*h*w,]
    
    def forward(self, X):
        (n,c,h,w) = X.shape
        self.input_shape = (n,c,h,w)
        self.output = X.reshape((n, c*h*w))

        if self.next_layer != None:
            self.next_layer.forward(self.output)
    
    def backward(self, output_derivative):
        derivative = output_derivative.reshape(self.input_shape)

        if self.prev_layer != None:
            self.prev_layer.backward(derivative)

    def __str__(self):
        return "{:<30}{:<20}{:<20}{:<10}\n".format(
            "Flatten layer", 
            "("+ ", ".join(map(str, self.input_shape))+")",
            "("+ str(self.output_shape[0])+",)",
            "0"
        )
# ------------------------------------------------------------------------------


if __name__ == "__main__":
    # Testing convolution layer to allow improving implementation in terms of 
    # efficiency without worying about breaking the functionality.
    from scipy.optimize import approx_fprime
    from scipy.signal import convolve2d
    import time
    import gzip
    import os
    import pickle

    with gzip.open(os.path.join('.', 'data', 'mnist.pkl.gz'), 'rb') as f:
            train_set, _, _ = pickle.load(f, encoding="latin1")

    X, y = train_set

    
    print("\n"+"="*50)
    print("   Test script for layers in neural network.")
    print("="*50)
    print("Enter the number of the test you wish to run:\n")
    print("0:   Convolutional Layer")
    print("1:   Dense Layer")
    print("_"*50)
    test_number = input(">>> ")

    try:
        test_number = int(test_number)
    except ValueError as ve:
        print("Invalid input. Quitting...")
        quit()
    
    if test_number == 0:
        C = ConvolutionalLayer(2,filter_size=5)
        X_im = X.reshape((len(X),1,28,28))
        
        print("\n{:^50}".format("Check forward direction"))
        print("_"*50)
        C.forward(X_im[1])
        scipy_conv = convolve2d(X_im[1,0], np.flip(C.W[0,0]),mode="valid")

        print("Sum of abs of my convolution:        {:0.6f}".format(np.sum(np.abs(C.output[0,0]))))
        print("Sum of abs of scipy convolution:     {:0.6f}".format(np.sum(np.abs(scipy_conv))))
        print("Sum of abs of difference:            {:0.8f}".format(np.sum(np.abs(scipy_conv - C.output[0,0]))))
        print("_"*50)

        sample = X_im[:1] # Sample over which to approximate / compute prime

        print("Time for forward pass:              ", end=" ", flush=True) 
        start = time.time()
        C.forward(sample)
        print("{:0.8f}".format(time.time()-start))

        start = time.time()
        C.backward(np.ones(C.output.shape))
        backward_runtime = time.time()-start

        # Helper function make the layer compatible with numerical optimization
        # algorithms, but reshaping input appropriately and using a "loss" of
        # taking the sum of all output elements.
        def helper_dx(x):
            x = x.reshape(C.input.shape)
            C.forward(x)
            return np.sum(C.output)

        def helper_dw(w):
            C.W = w.reshape(C.W.shape)
            C.forward(sample)
            return np.sum(C.output)

        def helper_db(b):
            C.b = b.reshape(C.b.shape)
            C.forward(sample)
            return np.sum(C.output)

        
        print("\n\n{:^50}".format("Check backward direction"))
        print("_"*50)
        conv_prime_dx = approx_fprime(
            sample.flatten(),
            helper_dx,
            epsilon=1e-2
        ).reshape(sample.shape)

        print("dX derivative:")
        print("______________")
        print("Sum of abs of my dX:                 {:0.6f}".format(np.sum(np.abs(C.derivative))))
        print("Sum of abs of numerical dX:          {:0.6f}".format(np.sum(np.abs(conv_prime_dx))))
        print("Sum of abs of difference:            {:0.8f}".format(np.sum(np.abs(C.derivative - conv_prime_dx))))

        conv_prime_dw = approx_fprime(
            C.W.flatten(),
            helper_dw,
            epsilon=1e-2
        ).reshape(C.W.shape)/len(sample)

        print("\ndW derivative:")
        print("______________")
        print("Sum of abs of my dW:                 {:0.6f}".format(np.sum(np.abs(C.dW))))
        print("Sum of abs of numerical dW:          {:0.6f}".format(np.sum(np.abs(conv_prime_dw))))
        print("Sum of abs of difference:            {:0.8f}".format(np.sum(np.abs(C.dW - conv_prime_dw))))

        conv_prime_db = approx_fprime(
            C.b.flatten(),
            helper_db,
            epsilon=1e-2
        ).reshape(C.b.shape)/len(sample)

        print("\ndb derivative:")
        print("______________")
        print("Sum of abs of my db:                 {:0.6f}".format(np.sum(np.abs(C.db))))
        print("Sum of abs of numerical db:          {:0.6f}".format(np.sum(np.abs(conv_prime_db))))
        print("Sum of abs of difference:            {:0.8f}".format(np.sum(np.abs(C.db - conv_prime_db))))
        print("_"*50)
        print("Time for forward pass:               {:0.8f}".format(backward_runtime)) 

    elif test_number == 1:
        n,d = X.shape
        D = DenseLayer(d, n_nodes=100)

        # Only to force computation of dx even though it is the first layer
        class DummyLayer():
            def backward(self, x):
                pass
        D.prev_layer = DummyLayer()

        sample = X[:3] # Sample over which to approximate / compute prime

        print("\nTime for a forward pass:            ", end=" ", flush=True) 
        start = time.time()
        D.forward(sample)
        print("{:0.8f}".format(time.time()-start))

        start = time.time()
        D.backward(np.ones(D.output.shape))
        backward_runtime = time.time()-start

        # Helper function make the layer compatible with numerical optimization
        # algorithms, but reshaping input appropriately and using a "loss" of
        # taking the sum of all output elements.
        def helper_dx(x):
            x = x.reshape(D.input.shape)
            D.forward(x)
            return np.sum(D.output)

        def helper_dw(w):
            D.W = w.reshape(D.W.shape)
            D.forward(sample)
            return np.sum(D.output)

        def helper_db(b):
            D.b = b.reshape(D.b.shape)
            D.forward(sample)
            return np.sum(D.output)
        
        print("\n\n{:^50}".format("Check backward direction"))
        print("_"*50)

        numerical_dx = approx_fprime(
            sample.flatten(),
            helper_dx,
            epsilon=1e-2
        ).reshape(sample.shape)

        print("dX derivative:")
        print("______________")
        print("Sum of abs of my dX:                 {:0.6f}".format(np.sum(np.abs(D.derivative))))
        print("Sum of abs of numerical dX:          {:0.6f}".format(np.sum(np.abs(numerical_dx))))
        print("Sum of abs of difference:            {:0.8f}".format(np.sum(np.abs(D.derivative - numerical_dx))))

        numerical_dw = approx_fprime(
            D.W.flatten(),
            helper_dw,
            epsilon=1e-2
        ).reshape(D.W.shape)/len(sample)

        print("\ndW derivative:")
        print("______________")
        print("Sum of abs of my dW:                 {:0.6f}".format(np.sum(np.abs(D.dW))))
        print("Sum of abs of numerical dW:          {:0.6f}".format(np.sum(np.abs(numerical_dw))))
        print("Sum of abs of difference:            {:0.8f}".format(np.sum(np.abs(D.dW - numerical_dw))))

        numerical_db = approx_fprime(
            D.b.flatten(),
            helper_db,
            epsilon=1e-2
        ).reshape(D.b.shape)/len(sample)

        print("\ndb derivative:")
        print("______________")
        print("Sum of abs of my db:                 {:0.6f}".format(np.sum(np.abs(D.db))))
        print("Sum of abs of numerical db:          {:0.6f}".format(np.sum(np.abs(numerical_db))))
        print("Sum of abs of difference:            {:0.8f}".format(np.sum(np.abs(D.db - numerical_db))))

        print("_"*50)
        print("Time for backward pass:               {:0.8f}".format(backward_runtime)) 
    else:
        print("Unknown test case. Quitting...")
        