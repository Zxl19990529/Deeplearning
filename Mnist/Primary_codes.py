import tensorflow as tf
import numpy as np 
from tensorflow.examples.tutorials.mnist import input_data
###---关闭警告---###
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
###---关闭警告---###
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder("float", [None, 784])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W) + b)

#计算 交叉熵
y_ = tf.placeholder("float", [None,10])

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#初始化我们创建的变量：
#init = tf.initialize_all_variables()
init=tf.global_variables_initializer()
#现在我们可以在一个Session里面启动我们的模型，并且初始化变量：

sess = tf.Session()
sess.run(init)
#然后开始训练模型，这里我们让模型循环训练1000次！

for i in range(1000):
    
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  if(i%100==0):
      print("After iterated "+str(i))

#评估我们的模型
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print (sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))