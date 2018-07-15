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

"""if L == 1:
    parameters["W" + str(L)] = np.random.randn(layer_dims[1], layer_dims[0]) * 0.01
    parameters["b" + str(L)] = np.zeros((layer_dims[1], 1))"""
def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
    bl -- bias vector of shape (layer_dims[l], 1)
    """
    np.random.seed(3)
    parameters={}
    L=len(layer_dims)

    for l in range(1,L):
        parameters['W'+str(l)]=np.random.randn(layer_dims[l],layer_dims[l-1])*0.01
        parameters['b'+str(l)]=np.zeros((layer_dims[l],1))
        assert(parameters['W'+str(l)].shape==(layer_dims[l],layer_dims[l-1]))
        assert(parameters['b'+str(l)].shape==(layer_dims[l],1))
    return parameters
parameters=initialize_parameters_deep([5,4,3])
for i in range (1,len([5,4,3])):
    print('W'+str(i)+' '+str(parameters['W'+str(i)]))
    print('b'+str(i)+' '+str(parameters['b'+str(i)]))
