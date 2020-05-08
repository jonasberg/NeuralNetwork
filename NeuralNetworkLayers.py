
import numpy as np

# -------------------------------- Computational layers ------------------------
class DenseLayer():
    def __init__(self, in_features, n_nodes):
        self.in_features = in_features
        self.n_nodes = n_nodes
        self.output_shape = [n_nodes,]
        self.prev_layer = None
        self.next_layer = None

        # See https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79
        # for explaination of initialization scheme of weights.
        self.W = np.random.normal(size=(in_features, n_nodes))*np.sqrt(2/in_features) 
        self.b = np.zeros((1, n_nodes))
        self.dW = None
        self.db = None
        self.input = None
        self.output = None
        self.derivative = None

    def forward(self, X):
        self.input = X
        self.output = X @ self.W + self.b

        if self.next_layer != None:
            self.next_layer.forward(self.output)

    def backward(self, output_derivative):
        n,_ = output_derivative.shape
        
        self.db = np.sum(output_derivative, axis=0) / n
        self.dW = (self.input.T @ output_derivative) / n

        if self.prev_layer != None:
            self.derivative = output_derivative @ self.W.T
            self.prev_layer.backward(self.derivative)

    def __str__(self):
        return "{:<30}{:<20}{:<20}{:<10} \n".format(
            "Dense layer",
            "("+str(self.in_features)+",)", 
            "("+str(self.n_nodes)+",)",
            self.W.size + self.b.size
        )

class ConvolutionalLayer(): 
    # Implementation is based on: https://towardsdatascience.com/convolutional-neural-networks-from-the-ground-up-c67bb41454e1

    def __init__(self, n_filters, filter_size=3, stride=1, input_shape=(1,28,28)): # in_shape, 
        c,h,w = input_shape

        self.n_filters = n_filters
        self.filter_size = filter_size
        self.stride = stride

        self.prev_layer = None
        self.next_layer = None

        self.W = np.random.normal(size=(
            n_filters, 
            c, 
            filter_size, 
            filter_size
        ))*np.sqrt(2/(n_filters*c*filter_size**2)) # Correct initialization?
        self.b = np.zeros((n_filters))

        self.dW = None
        self.db = None
        self.derivative = None

        self.input = None
        self.output = None

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
        (n,c,h,w) = self.input.shape
        
        
        # Initialize derivative matrices
        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)
        self.derivative = np.zeros(self.input.shape)

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
                input_term = self.input[:,:,curr_y:curr_y + f, curr_x: curr_x + f] # Shape (N x C x F x F)
                mean_dW_term = np.mean(
                    output_derivative_term[:,:,None]*input_term[:,None,:,:,:]
                ,axis=0) # Shape (N_Filters x C x K x K)
                
                self.dW += mean_dW_term
                
                # Update derivative of input 
                self.derivative[:, :, curr_y:curr_y + f, curr_x:curr_x + f] += np.sum(
                    output_derivative_term[:,:,None,:,:] * self.W[None, :,:,:,:]
                , axis=1) # Sum over all filters in W
                curr_x += self.stride
                out_x += 1

            curr_y += self.stride
            out_y += 1
        self.db = np.mean(np.sum(output_derivative, axis=(2,3)), axis=0)
        
        if self.prev_layer != None:
            self.prev_layer.backward(self.derivative)
    
    def __str__(self):
        return "{:<30}{:<20}{:<20}{:<10} \n".format(
            "Convolutional layer",
            "("+ ", ".join(map(str, self.input_shape))+")",
            "("+ ", ".join(map(str, self.output_shape))+")",
            self.W.size + self.b.size
        )
# ------------------------------------------------------------------------------


# -------------------------------- Activations ---------------------------------
class ReLU():
    def __init__(self, output_shape):
        self.prev_layer = None
        self.next_layer = None
        self.output = None
        self.derivative = None
        self.output_shape = output_shape

    def forward(self, X):
        out = X.copy()
        out[out<0] = 0
        self.output = out

        if self.next_layer != None:
            self.next_layer.forward(self.output)

    def backward(self, output_derivative):
        self.derivative = output_derivative.copy()
        self.derivative[self.output == 0] = 0

        if self.prev_layer != None:
            self.prev_layer.backward(self.derivative)

    def __str__(self):
        return "{:<30}{:<20}{:<20}{:<10}\n".format(
            "ReLU activation", 
            "("+ ", ".join(map(str, self.output_shape))+")",
            "("+ ", ".join(map(str, self.output_shape))+")",
            "0"
        )

class Softmax():
    # Mathematical derivation found at: https://aimatters.wordpress.com/2019/06/17/the-softmax-function-derivative/
    def __init__(self, output_shape):
        self.prev_layer = None
        self.next_layer = None
        self.output = None
        self.derivative = None
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
        self.derivative = out_and_out_grad - outer_matrix_summed

        if self.prev_layer != None:
            self.prev_layer.backward(self.derivative)

    def __str__(self):
        return "{:<30}{:<20}{:<20}{:<10}\n".format(
            "Softmax activation", 
            "("+ ", ".join(map(str, self.output_shape))+")",
            "("+ ", ".join(map(str, self.output_shape))+")",
            "0"
        )
# ------------------------------------------------------------------------------

# -------------------------------- Other layers --------------------------------

class Dropout():
    def __init__(self, output_shape, dropout_rate=0.5):
        self.prev_layer = None
        self.next_layer = None
        self.output = None
        self.derivative = None
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
        self.derivative = self.random_matrix * output_derivative 

        if self.prev_layer != None:
            self.prev_layer.backward(self.derivative)

    def __str__(self):
        return "{:<30}{:<20}{:<20}{:<10}\n".format(
            "Dropout layer", 
            "("+ ", ".join(map(str, self.output_shape))+")",
            "("+ ", ".join(map(str, self.output_shape))+")",
            "0"
        )

class Flatten():
    def __init__(self, input_shape):
        self.prev_layer = None
        self.next_layer = None
        self.input_shape = None

        (c,h,w) = input_shape
        self.input_shape = input_shape
        self.output_shape = [c*h*w,]
    
    def forward(self, X):
        (n,c,h,w) = X.shape
        self.input_shape = (n,c,h,w)
        output = X.reshape((n, c*h*w))

        if self.next_layer != None:
            self.next_layer.forward(output)
    
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