#this is a 5-layer network
# And we use the codes we have done before
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v2 import *

np.random.seed(1)

train_x_orign,train_y,test_x_orign,test_y,classes=load_data()

train_x_flatten=train_x_orign.reshape(train_x_orign.shape[0],-1).T 
#print(format(train_x_flatten.shape))# (12288,209)
test_x_flatten=test_x_orign.reshape(test_x_orign.shape[0],-1).T 
#print(format(test_x_flatten.shape))#(12288,50)
###---standarlize---###
train_x=train_x_flatten/255
test_x=test_x_flatten/255

layers_dims = [12288, 20, 7, 5, 1] # 5-layer model

#############################
###- - - - - - - - - - - -###
###- - B I G S T A R T - -###
###- - - - - - - - - - - -###
#############################

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = [] # keep track of cost

    # Parameters initialization.
    ###---start---###
    # parameters = initialize_parameters_deep(layers_dims)
    parameters=initialize_parameters_deep(layers_dims)
    ###---end---###

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        ###---start---### (≈ 1 line of code)
        # AL, caches = L_model_forward(X, parameters)
        A_last,caches=L_model_forward(X=X,parameters=parameters)
        ###---end---###

        # Compute cost.
        ###---start---### (≈ 1 line of code)
        # cost = compute_cost(AL, Y)
        cost=compute_cost(AL=A_last,Y=Y)
        ###---end---###

        # Backward propagation.
        ###---start---### (≈ 1 line of code)
        # grads = L_model_backward(AL, Y, caches)
        grads=L_model_backward(AL=A_last,Y=Y,caches=caches)
        ###---end---###

        # Update parameters.
        ###---start---### (≈ 1 line of code)
        # parameters = update_parameters(parameters, grads, learning_rate)
        parameters=update_parameters(parameters=parameters,learning_rate=learning_rate,grads=grads)
        ###---end---###

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration {}:{}" .format(i,np.squeeze(cost)))
        #if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters
parameters=L_layer_model(train_x,train_y,layers_dims,learning_rate=0.0075,num_iterations=2500,print_cost=True)

predict(train_x,train_y,parameters)
predict(test_x,test_y,parameters)

my_image="my_image.png"
my_image_label=[1]
image_name="images/"+my_image





