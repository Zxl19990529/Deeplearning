import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
mnist= input_data.read_data_sets('MNIST_data',one_hot=True)

# 初始化权重
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def bias_Variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
# 定义卷积和池化
def conv2d(input,W):
     return tf.nn.conv2d(input=input,filter=W,strides=[1,1,1,1],padding='SAME')
def max_pool_2x2(input):
    return tf.nn.max_pool(input,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#x_image(batch, 28, 28, 1) -> h_pool1(batch, 14, 14, 32)
x=tf.placeholder(float,shape=[None,784])
x_image=tf.reshape(x,[-1,28,28,1])
###---conv_1---###
W_conv1=weight_variable([5,5,1,32])
b_conv1=bias_Variable([32])
conv_1=conv2d(x_image,W_conv1)+b_conv1
out_conv1=tf.nn.relu(conv_1)
    # x_image -> [batch, in_height, in_width, in_channels]
    #            [batch, 28, 28, 1]
    # W_conv1 -> [filter_height, filter_width, in_channels, out_channels]
    #            [5, 5, 1, 32]
    # out_conv1-> [batch, out_height, out_width, out_channels]
    #            [batch, 28, 28, 32]
pool_conv1=max_pool_2x2(out_conv1)
    # out_conv1 -> [batch, in_height, in_weight, in_channels]
    #            [batch, 28, 28, 32]
    # pool_conv1  -> [batch, out_height, out_weight, out_channels]
    #            [batch, 14, 14, 32]

###---conv_2---###
W_conv2=weight_variable([5,5,32,64]) #用5,5,32  的卷积核（32通道），用64个
b_conv2=bias_Variable([64])
conv_2=conv2d(pool_conv1,W_conv2)+b_conv2
out_conv2=tf.nn.relu(conv_2)

    # pool_conv1 -> [batch, 14, 14, 32]
    # W_conv2 -> [5, 5, 32, 64]
    # out_conv2  -> [batch, 14, 14, 64]
pool_conv2=max_pool_2x2(out_conv2)
    # out_conv2 -> [batch, 14, 14, 64]
    # pool_conv2  -> [batch, 7, 7, 64]

###---全连接---###
# pool_conv2(batch, 7, 7, 64) -> h_fc1(1, 1024)
W_fullconnection_1=weight_variable([7*7*64,1024])
b_fullconnection_1=bias_Variable([1024])

pool_conv2_flat=tf.reshape(pool_conv2,[-1,7*7*64])

out_fullconnection=tf.nn.relu(tf.matmul(pool_conv2_flat,W_fullconnection_1)+b_fullconnection_1)

###---drop---###
"""防止过拟合"""
keep_prob=tf.placeholder("float")
out_fullconnection_drop=tf.nn.dropout(out_fullconnection,keep_prob)

###---softmax---###
W_soft_max=weight_variable([1024,10])
b_soft_max=bias_Variable([10])

out_softmax=tf.nn.softmax(tf.matmul(out_fullconnection_drop,W_soft_max)+b_soft_max)

###---训练&&评估---###
'''
ADAM优化器来做梯度最速下降,feed_dict中加入参数keep_prob控制dropout比例
'''
y_real=tf.placeholder('float',[None,10])
cross_entropy=-tf.reduce_sum(y_real*tf.log(out_softmax))
train_step=tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)
correct_prediction=tf.equal(tf.argmax(out_softmax,1),tf.argmax(y_real,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,'float'))

sess=tf.Session()
sess.run(tf.global_variables_initializer())
mnist= input_data.read_data_sets('MNIST_data',one_hot=True)
for i in range(5000):
    batch=mnist.train.next_batch(50)
    train_step.run(session=sess,feed_dict={x:batch[0],y_real:batch[1],keep_prob:0.5})
    if i % 100 == 0:
        train_accuracy = accuracy.eval(session = sess,feed_dict = {x:batch[0], y_real:batch[1], keep_prob:1.0})
        print("step %d, train_accuracy %g" %(i, train_accuracy))
        

print("test accuracy %g" %accuracy.eval(session = sess,feed_dict = {x:mnist.test.images, y_real:mnist.test.labels,keep_prob:1.0}))



