import numpy as np 
from testCases_v3 import *
from myhomework_PART1_period_4_2_a import linear_forward,linear_activation_forward
from dnn_utils_v2 import sigmoid,sigmoid_backward,relu,relu_backward

def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()

    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
    every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
    the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """
    
    caches = []
    A = X
    L = len(parameters) // 2 # number of layers in the neural network
                             #这里面只有 w和b，所以parameters需要整除2
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):#循环从L到L-1
        A_prev = A
        ###---start---###
        A,cache=linear_activation_forward(A_prev,W=parameters['W'+str(l)],b=parameters['b'+str(l)],activation="relu")
        caches.append(cache)
        ###---end---###

    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    ###---start---### (≈ 2 lines of code)
    AL,cache=linear_activation_forward(A,W=parameters['W'+str(L)],b=parameters['b'+str(L)],activation="sigmoid")
    caches.append(cache)
    ###---end---###

    assert(AL.shape == (1,X.shape[1]))

    return AL, caches

X, parameters = L_model_forward_test_case_2hidden()
AL, caches = L_model_forward(X, parameters)
print("AL = " + str(AL))
print("Length of caches list = " + str(len(caches)))

