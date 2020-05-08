# Neural Network 

This is a project written for the purpose of learning the mathematical details of neural networks, including how to form predictions, backpropagation and optimization of the network. The code is functional, but by no means optimized and should not be used for any other purposes than educational ones.

## Known issues

* The biases of the network have a tendency to update very slowly, leading to slow convergence during training. This leads to a poor ability to effectively approximate non-linear decision boundaries at present. Some experimentation has shown that using a more sophisticated optimization method (e.g. utilizing momentum) seems to resolve this issue. 