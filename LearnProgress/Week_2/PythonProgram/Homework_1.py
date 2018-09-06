import numpy as np 
#Exercise_1: Implement the sigmoid function using numpy. 

def sigmod(x):
    s=1/(1+np.exp(x))
    return s

x=np.array([1,2,3])
print ("sigmod is "+str(sigmod(x)))
# Here use np.exp instead of math because the input is a vector
#
# Esercise_2:
# Implement the function sigmoid_grad() to compute the gradient of the sigmoid function with respect to its input x. 
# sigmoid_derivative(x)=σ′(x)=σ(x)(1−σ(x))
# 
def sigmoid_derivative(x):
    s=sigmod(x)
    ds=s*(1-s)
    return ds
print("sigmoid_derivative is "+str(sigmoid_derivative(x)))
# Exercise_3: Reshaping arrays
# Two common numpy functions used in deep learning are np.shape and np.reshape(). 
#   -X.reshape() is used to reshape X into some other dimension.   
#   -X.shape is used to get the shape (dimension) of a matrix/vector X. 
#Implement image2vector() that takes an input of shape (length, height, 3) and returns a vector of shape (length*height*3, 1).
# For example, if you would like to reshape an array v of shape (a, b, c) into a vector of shape (a*b,c) you would do:
# 
def image2vector(image):
    res = image.reshape((image.shape[0] * image.shape[1] * image.shape[2], 1))
    return res
image = np.array([[[ 0.67826139,  0.29380381],
        [ 0.90714982,  0.52835647],
        [ 0.4215251 ,  0.45017551]],

       [[ 0.92814219,  0.96677647],
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]],

       [[ 0.60659855,  0.00533165],
        [ 0.10820313,  0.49978937],
        [ 0.34144279,  0.94630077]]])
print("image reshaped like this "+"\n"+str(image2vector(image)))
# Exercise_4 : Broadcasting
# This uses the broadcasting technquie
def broadcast(x,y):
    s=x/y
    return s

    
X=np.array([[1,2,3],
            [4,5,6]])
X_2=np.array([[5.0],
             [2.0]])
print(broadcast(X,X_2))            
# 
# 2.1 Implement the L1 and L2 loss functions
# Exercise: Implement the numpy vectorized version of the L1 loss. You may find the function abs(x) (absolute value of x) useful.
#########---L1---#########
import numpy as np
def L1(yhat, y):
    loss = np.sum(np.abs(y - yhat))
    return loss
yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L1 = " + str(L1(yhat,y)))
#########---L1---#########
def L2(yhat, y):
    loss =np.sum(np.power((y - yhat), 2))
    return loss
yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L2 = " + str(L2(yhat,y)))