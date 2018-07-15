import numpy as np 
import h5py
import matplotlib.pyplot as  plt 
from testCases_v3 import *
from dnn_utils_v2 import sigmoid,sigmoid_backward,relu,relu_backward

#%matplotlib inline
plt.rcParams['figure.figsize'] = (5.0, 4.0); # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

#%load_ext autoreload
#%autoreload 2

np.random.seed(1)
#########################

def initialize_parameters(n_x, n_h, n_y):
    ##--这个函数用来初始化两层神经网络的权重参数--##
    """
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    Returns:
    parameters -- python dictionary containing your parameters:
    W1 -- weight matrix of shape (n_h, n_x)
    b1 -- bias vector of shape (n_h, 1)
    W2 -- weight matrix of shape (n_y, n_h)
    b2 -- bias vector of shape (n_y, 1)

    """
    np.random.seed(1)
    ###---start---###
    W1=np.random.randn(n_h,n_x)*0.01
    b1=np.zeros((n_h,1))
    W2=np.random.randn(n_y,n_h)*0.01
    b2=np.zeros((n_y,1))

    ###---debug---###
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))
    ###---debug_end---###
    parameters={
        "W1":W1,
        "b1":b1,
        "W2":W2,
        "b2":b2
    }
    return parameters
###---test---###
parameters=initialize_parameters(2,3,1)
print("W1= "+str(parameters["W1"]))
print("b1= "+str(parameters["b1"]))
print("W2= "+str(parameters["W2"]))
print("b2= "+str(parameters["b2"]))
###---test_end---###
    



