import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v2 import *
###---以上是需要引入的库---###

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)
###---准备好画布---###
###---加载图像数据---###
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
###---随便显示一张图片---###
index = 101
plt.imshow(train_x_orig[index])
# plt.show()
print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") + " picture.")

# 看看h5里面都有什么
m_train = train_x_orig.shape[0] #训练集的数量
num_px = train_x_orig.shape[1] #图片长/宽
m_test = test_x_orig.shape[0]#测试集的数量

print ("Number of training examples: " + str(m_train)+" 训练集的数量")
print ("Number of testing examples: " + str(m_test)+" 测试集的数量")
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)"+ " 图片长/宽")
print ("train_x_orig shape: " + str(train_x_orig.shape)+" 训练集x的形状")
print ("train_y shape: " + str(train_y.shape)+" 训练集y的形状")
print ("test_x_orig shape: " + str(test_x_orig.shape)+" 测试集x的形状")
print ("test_y shape: " + str(test_y.shape)+" 测试集y的形状")
###---在使用之前需要把图片标准化---###
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
print("train_x_flatten shape:"+str(train_x_flatten.shape))
print("test_x_flatten shape:"+str(test_x_flatten.shape))

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))



