# -*- coding: utf-8 -*-

import numpy as np
import h5py
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
from dnn_app_utils_v2 import sigmoid,load_data
from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward

plt.rcParams['figure.figsize'] = (5.0, 4.0) 
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

# 随机初始化权重w和偏置b
def initialize_parameters_deep(layer_dims):
    #layer_dims是一个list结构，包含了每一层的单元数量
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L): #循环从1到L-1
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])/ np.sqrt(layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
    return parameters #parameters中保存了每一层神经元的权重w和偏置b

# 线性传播，即前向传播的线性计算过程
def linear_forward(A_prev, W, b):
    Z = np.dot(W,A_prev)+b
    assert(Z.shape == (W.shape[0], A_prev.shape[1]))
    cache = (A_prev, W, b)
    return Z, cache #保存一些参数到cache中，注：这里的A_prev指前一层的A

# 加入激励函数
def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b) #得到Z、A_prev、W、b
        A, activation_cache = sigmoid(Z) #得到当前层的L和Z
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache) #cache中的值依次为A_prev, W, b, Z
    return A, cache

#前向传播模型
def L_model_forward(X, parameters):#parameters代表权重和偏置
    caches = []
    A = X
    L = len(parameters) // 2 #len(parameters)=2*L        
    for l in range(1, L):#循环从1到L-1
        A_prev = A 
        #前几层采用ReLU作为激励函数
        A, cache = linear_activation_forward(A_prev, parameters["W"+str(l)], parameters["b"+str(l)],activation = "relu")
        caches.append(cache)
    #最后一层用sigmoid作为激励函数，说明这是一个二分类的模型
    AL, cache = linear_activation_forward(A, parameters["W"+str(L)], parameters["b"+str(L)],activation = "sigmoid")
    caches.append(cache)
    assert(AL.shape == (1,X.shape[1]))
    return AL, caches

# 计算损失函数
def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = (-1./m)*np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL))
    #cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    
    cost = np.squeeze(cost)    
    assert(cost.shape == ()) 
    return cost

# 线性反向传播
def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1./m)*np.dot(dZ,A_prev.T)
    db = (1./m)*np.sum(dZ,axis = 1 ,keepdims = True)
    dA_prev = np.dot(W.T,dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) 
    
    dAL =  - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation = "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
        
    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2 
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
    return parameters

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 2500, print_cost=False):
    np.random.seed(1)
    costs = []
    
    parameters = initialize_parameters_deep(layers_dims)
    
    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

def predict(parameters, X):
    AL, caches = L_model_forward(X, parameters)
    predictions = (AL>0.5) #这句话的意思是，如果A2>0.5，则predictions的值为1，否则为0
    return predictions #返回的是400个预测值


print("加载数据.....")
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
print("加载数据完毕.....")

#打印一张图片
index = 10
plt.imshow(train_x_orig[index])
"""
print("显示训练集样本和测试集样本基本信息：")
print("训练集样本X维度："+str(train_x_orig.shape))
print("训练集样本Y维度："+str(train_y.shape))
print("测试集样本X维度："+str(test_x_orig.shape))
print("测试集样本Y维度："+str(test_y.shape))
m_train = train_x_orig.shape[0]
m_test = test_x_orig.shape[0]
print ("训练集样本数为: " + str(m_train))
print ("测试集样本数为: " + str(m_test))
num_px = train_x_orig.shape[1]
print ("每个图像的尺寸为: (" + str(num_px) + ", " + str(num_px) + ", 3)")

#reshape训练集与测试集
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
print("reshape后的训练集维度："+str(train_x_flatten.shape))
print("reshape后的测试集维度："+str(test_x_flatten.shape))

#归一化数据
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.
#初始化网络参数
layers_dims = [12288, 20, 7, 5, 1]

print("\n开始训练模型.....")
parameters=L_layer_model(train_x, train_y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=True)
print("\n训练结束，打印训练集准确率:")
train_predictions = predict(parameters, train_x)
print ('Accuracy: %d' % float((np.dot(train_y,train_predictions.T) + np.dot(1-train_y,1-train_predictions.T))/float(train_y.size)*100) + '%')
print("\n打印测试集准确率:")
test_predictions = predict(parameters, test_x)
print ('Accuracy: %d' % float((np.dot(test_y,test_predictions.T) + np.dot(1-test_y,1-test_predictions.T))/float(test_y.size)*100) + '%')
"""












