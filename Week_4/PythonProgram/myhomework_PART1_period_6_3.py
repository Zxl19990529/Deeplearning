import numpy as np 
from myhomework_PART1_period_6_2 import linear_activation_backward
from testCases_v3 import print_grads,L_model_backward_test_case
def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
    every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
    the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

    Returns:
    grads -- A dictionary with the gradients
    grads["dA" + str(l)] = ...
    grads["dW" + str(l)] = ...
    grads["db" + str(l)] = ...
    """
    grads = {}
    L = len(caches) # the number of layers
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

    # Initializing the backpropagation
    ###---start---### (1 line of code)
    dAL=-Y/AL +(1-Y)/(1-AL)
    ###---end---###

    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    ###---start---### (approx. 2 lines)
    """current_cache=caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")
    """
    current_cache=caches[L-1]
    grads["dA"+str(L)],grads["dW"+str(L)],grads["db"+str(L)]=linear_activation_backward(dAL,current_cache,activation="sigmoid")
    ###---end---###

    for l in reversed(range(L-1)):        #假如有三层，L=3，那么l从1开始，到0结束
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
        
        ###---start---### (approx. 5 lines)
        print(l)
        current_cache=caches[l]
        dA_prev_t,dW_t,db_t=linear_activation_backward(dA=grads["dA"+str(l+2)],cache=current_cache,activation="relu")
        grads["dA"+str(l+1)]=dA_prev_t
        grads["dW"+str(l+1)]=dW_t
        grads["db"+str(l+1)]=db_t        
        ###---end---###

    return grads

AL, Y_assess, caches = L_model_backward_test_case()
grads = L_model_backward(AL, Y_assess, caches)
print_grads(grads)