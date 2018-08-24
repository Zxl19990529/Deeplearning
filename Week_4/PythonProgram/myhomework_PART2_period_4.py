#this is a 2-layer network
# And we use the codes we have done before
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v2 import *
""" 前期数据准备 """
np.random.seed(2)
train_x_orign,train_y,test_x_orign,test_y,classes=load_data()
#print("train_x_orign shape: "+ str(train_x_orign.shape))
train_x_flatten=train_x_orign.reshape(train_x_orign.shape[0],-1).T 
#after flatten:(12288,209)
test_x_flatten=test_x_orign.reshape(test_x_orign.shape[0],-1).T 
#after flatten:(12288,209)
###---standarlize---###
train_x=train_x_flatten/255.
test_x=test_x_flatten/255.

###---内容决定了是几层网络---####
n_x = 12288 # num_px * num_px * 3
n_h = 7     # 隐藏层有7个单元
n_y = 1     # 输出有一个
layers_dims = (n_x, n_h, n_y)
"""
Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
"""

#############################
###- - - - - - - - - - - -###
###- - B I G S T A R T - -###
###- - - - - - - - - - - -###
#############################


###---2层神经网络---###
def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.

    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- If set to True, this will print the cost every 100 iterations

    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    """

    np.random.seed(1)
    grads = {}
    costs = [] # to keep track of the cost
    m = X.shape[1] # number of examples
    (n_x, n_h, n_y) = layers_dims

    # Initialize parameters dictionary, by calling one of the functions you'd previously implemented
    # 初始化参数
    ###---start---### (≈ 1 line of code)
    parameters = initialize_parameters(n_x, n_h, n_y)
    ###---end---###

    # Get W1, b1, W2 and b2 from the dictionary parameters.
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Loop (gradient descent)

    for i in range(0, num_iterations):
        #前向传播
        # Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID. Inputs: "X, W1, b1". Output: "A1, cache1, A2, cache2".
        ###---start---### (≈ 2 lines of code)
        # A1, cache1 = linear_activation_forward(X, W1, b1, activation="relu")
        # A2, cache2 = linear_activation_forward(A1, W2, b2, activation="sigmoid")
        A1,cache1=linear_activation_forward(A_prev=X,W=W1,b=b1,activation="relu")
        A2,cache2=linear_activation_forward(A_prev=A1,W=W2,b=b2,activation="sigmoid")
        ###---end---###

        # 计算成本函数
        ###---start---### (≈ 1 line of code)
        #cost = compute_cost(A2, Y)
        cost=compute_cost(AL=A2,Y=Y)
        ###---end---###

        # Initializing backward propagation
        #初始化反向传播
        #dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        dA2=-(Y/A2  - (1-Y)/1-A2)

        # Backward propagation. Inputs: "dA2, cache2, cache1". Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
        #反向传播
        ###---start---### (≈ 2 lines of code)
        # dA1, dW2, db2 = linear_activation_backward(dA2, cache2, activation="sigmoid")
        # dA0, dW1, db1 = linear_activation_backward(dA1, cache1, activation="relu")
        dA1,dW2,db2=linear_activation_backward(dA=dA2,cache=cache2,activation="sigmoid")
        dA0,dW1,db1=linear_activation_backward(dA=dA1,cache=cache1,activation="relu")
        ###---end---###

        # Set grads['dWl'] to dW1, grads['db1'] to db1, grads['dW2'] to dW2, grads['db2'] to db2
        #设置梯度
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        # Update parameters.
        # 更新参数
        ###---start---### (approx. 1 line of code)
        #parameters = update_parameters(parameters, grads, learning_rate)
        parameters=update_parameters(parameters,grads,learning_rate)
        ###---end---###

        # Retrieve W1, b1, W2, b2 from parameters
        # 替换更新
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        # Print the cost every 100 training example
        # 每一百次学习查看成本损失
        
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        # if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters
# 运行模型

#parameters = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True)
parameters=two_layer_model(train_x,train_y,learning_rate=0.0075,layers_dims=(n_x,n_h,n_y),num_iterations=2000,print_cost=True)
predictions_train = predict(train_x, train_y, parameters)
predictions_test = predict(test_x, test_y, parameters)




























