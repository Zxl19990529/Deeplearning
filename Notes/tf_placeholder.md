## tf.placeholder函数 学习笔记<h1>
[原文链接](https://blog.csdn.net/zj360202/article/details/70243127)
### 函数原型<h2>
```python
tf.placeholder(dtype, shape=None
```

> 此函数可以理解为形参，用于定义过程，在执行的时候再赋具体的值

## 参数：<h3> 
- **dtype**：数据类型。常用的是tf.float32,tf.float64等数值类型
- **shape**：数据形状。默认是None，就是一维值，也可以是多维，比如[2,3], [None, 3]表示列是3，行不定
- **name**：名称。
```python
x = tf.placeholder(tf.float32, shape=(1024, 1024))
y = tf.matmul(x, x)
 
with tf.Session() as sess:
  print(sess.run(y))  # ERROR: 此处x还没有赋值.
 
  rand_array = np.random.rand(1024, 1024)
  print(sess.run(y, feed_dict={x: rand_array}))  # Will succeed.
```
- **返回**：Tensor 类型