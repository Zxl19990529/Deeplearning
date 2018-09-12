对于python2 :
```py
import sys
sys.path.insert(0,'/home/zhb/Downloads/caffe-master/python')
import caffe

```
对于caff.net():
```
import caffe 之后
# 提取网络
dnn=caffe.Net("lenet.prototxt",weights_.caffemodel,caffe.TEST)

layers=dnn06.params.keys() # 提取层的名称
print (layers)
blobs_=dnn06.blobs.keys() # 提取 blob 
print(blobs_)

conv1_weights=dnn06.params['conv1'][0].data
conv1_bias=dnn06.params['conv1'][1].data
```
