# Deeplearning 回顾
![](http://7pn4yt.com1.z0.glb.clouddn.com/blog-cnn.png)

|模型名|	AlexNet|	VGG|	GoogLeNet|	ResNet
|-----|-----------|-------|-------------|-------|
|初入江湖|	2012|	2014|	2014|	2015|
|层数|	8|	19|	22|	152|
|Top-5错误|	16.4%|	7.3%|	6.7%|	3.57%|
|Data Augmentation|	+|	+|	+|	+|
|Inception(NIN)|	–|	–|	+|	–|
|卷积层数|	5|	16|	21|	151|
|卷积核大小|	11,5,3|	3|	7,1,3,5	7,1,3,5|
|全连接层数	3|	3|	1|	1|
|全连接层大小|	4096,4096,1000|	4096,4096,1000|	1000|	1000|
|Dropout|	+|	+|	+|	+|
|Local| Response Normalization|	+|	–|	+|	–|
|Batch| Normalization|	–|	–|	–|	+|
## 最古老的CNN
1985年，Rumelhart和Hinton等人提出了[后向传播（Back Propagation，BP）算法](http://www.dtic.mil/dtic/tr/fulltext/u2/a164453.pdf)，使得神经网络的训练变得简单可行，几年后，LeCun利用BP算法来训练多层神经网络用于[识别手写邮政编码](http://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf)，这个工作就是CNN的开山之作
## LeNet
1998年的LeNet5[4]标注着CNN的真正面世，但是这个模型在后来的一段时间并未能火起来，主要原因是费机器。但麻雀虽小，五脏俱全，卷积层、pooling层、全连接层，这些都是现代CNN网络的基本组件。  
![](http://7pn4yt.com1.z0.glb.clouddn.com/blog-lenet.jpg)
## [AlexNet](https://blog.csdn.net/lg1259156776/article/details/52551158)
[论文链接](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)  
不可否认，深度学习的热潮正是由2012年AlexNet的出现而引发的。在AlexNet之前，深度学习已经沉寂了很久。历史的转折在2012年到来，AlexNet 在当年的ImageNet图像分类竞赛中，top-5错误率比上一年的冠军下降了十个百分点，而且远远超过当年的第二名。  
而AlexNet 之所以能够成功，深度学习之所以能够重回历史舞台，原因在于：  
```
非线性激活函数：ReLU
防止过拟合的方法：Dropout，Data augmentation
大数据训练：百万级ImageNet图像数据
其他：GPU实现，LRN归一化层的使用
```
### Architecture   

![](https://img-blog.csdn.net/20180105160330931?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvenltMTk5NDExMTk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
## VGG
[论文链接](https://arxiv.org/pdf/1409.1556.pdf) 
VGG是Oxford的**V**isual **G**eometry **G**roup的组提出的（大家应该能看出VGG名字的由来了）。该网络是在ILSVRC 2014上的相关工作，主要工作是证明了增加网络的深度能够在一定程度上影响网络最终的性能。VGG有两种结构，分别是VGG16和VGG19，两者并没有本质上的区别，只是网络深度不一样。  
*VGG很好地继承了AlexNet的衣钵，一个字：深，两个字：更深。*
### VGG原理
VGG16相比AlexNet的一个改进是采用连续的几个3x3的卷积核代替AlexNet中的较大卷积核（11x11，7x7，5x5）。对于给定的感受野（与输出有关的输入图片的局部大小），采用堆积的小卷积核是优于采用大的卷积核，因为多层非线性层可以增加网络深度来保证学习更复杂的模式，而且代价还比较小（参数更少）。
### VGG 结构
[高清VGG](https://dgschwend.github.io/netscope/#/preset/vgg-16)  
![](https://d2mxuefqeaa7sj.cloudfront.net/s_8C760A111A4204FB24FFC30E04E069BD755C4EEFD62ACBA4B54BBA2A78E13E8C_1491022251600_VGGNet.png)

- VGG16包含了16个隐藏层（13个卷积层和3个全连接层），如上图中的D列所示
- VGG19包含了19个隐藏层（16个卷积层和3个全连接层），如上图中的E列所示
VGG网络的结构非常一致，从头到尾全部使用的是3x3的卷积和2x2的max pooling。
### 优缺点
**优点**
- VGGNet的结构非常简洁，整个网络都使用了同样大小的卷积核尺寸（3x3）和最大池化尺寸（2x2）。
- 几个小滤波器（3x3）卷积层的组合比一个大滤波器（5x5或7x7）卷积层好：
- 验证了通过不断加深网络结构可以提升性能。  

**缺点**
- VGG耗费更多计算资源，并且使用了更多的参数（这里不是3x3卷积的锅），导致更多的内存占用（140M）。其中绝大多数的参数都是来自于第一个全连接层。VGG有3个全连接层.
## [GoogLeNet](https://blog.csdn.net/qq_31531635/article/details/72232651)
[论文链接](https://arxiv.org/pdf/1409.4842.pdf)  
inception（也称GoogLeNet）是2014年Christian Szegedy提出的一种全新的深度学习结构，在这之前的AlexNet、VGG等结构都是通过增大网络的深度（层数）来获得更好的训练效果，但层数的增加会带来很多负作用，比如过拟合、梯度消失、梯度爆炸等。inception的提出则从另一种角度来提升训练结果：能更高效的利用计算资源，在相同的计算量下能提取到更多的特征，从而提升训练结果。  
始于LeNet-5，一个有着标准的堆叠式卷积层冰带有一个或多个全连接层的结构的卷积神经网络。通常使用dropout来针对过拟合问题。   
为了提出一个更深的网络，GoogLeNet做到了22层，利用inception结构，这个结构很好地利用了网络中的计算资源，并且在不增加计算负载的情况下，增加网络的宽度和深度。同时，为了优化网络质量，采用了Hebbian原理和多尺度处理。GoogLeNet在分类和检测上都取得了不错的效果。 

### GoogleNet原理
直接提升深度神经网络的方法就是增加网络的尺寸，包括宽度和深度。深度也就是网络中的层数，宽度指每层中所用到的神经元的个数。但缺点也随之而来：  
- 网络尺寸的增加也意味着参数的增加，也就使得网络更加容易过拟合。 
- 计算资源的增加。  

通常全连接是为了更好的优化并行计算，而稀疏连接是为了打破对称来改善学习，传统常常利用卷积来利用空间域上的稀疏性，但卷积在网络的早期层中的与patches的连接也是稠密连接，因此考虑到能不能在滤波器层面上利用稀疏性，而不是神经元上。但是在非均匀稀疏数据结构上进行数值计算效率很低，并且查找和缓存未定义的开销很大，而且对计算的基础设施要求过高，因此考虑到将稀疏矩阵聚类成相对稠密子空间来倾向于对稀疏矩阵的计算优化。因此提出了inception结构。  

![](https://images2015.cnblogs.com/blog/822124/201609/822124-20160902160437324-793316644.png)  


由于滤波器数量的增加，加上池化操作使得5x5大小的滤波器的计算开销非常大，池化层输出与卷积层输出的合并增加了输出值的数量，并且可能覆盖优化稀疏结构，处理十分低效，引起计算爆炸。因此引出下面这个inception结构。 

![](https://upload-images.jianshu.io/upload_images/8904720-e54387f99054ef49.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/700)

## ResNet
ResNet在2015年被提出，在ImageNet比赛classification任务上获得第一名，因为它“简单与实用”并存，之后很多方法都建立在ResNet50或者ResNet101的基础上完成的，检测，分割，识别等领域都纷纷使用ResNet，Alpha zero也使用了ResNet，所以ResNet很好用。 
### 问题背景
因为CNN能够提取 low/mid/high-level 的特征，网络的层数越多，意味着能够提取到不同level的特征越丰富。并且，越深的网络提取的特征越抽象，越具有语义信息。  
但是，对于原来的网络，如果简单地增加深度，会导致梯度弥散或梯度爆炸。
- 对于该问题的解决方法是正则化初始化和中间的正则化层（Batch Normalization），这样的话可以训练几十层的网络。
虽然通过上述方法能够训练了，但是又会出现另一个问题，就是 **退化问题**，网络层数增加，但是在训练集上的 **准确率却饱和甚至下降了**
### Res结构原理
总结构：  
![](https://upload-images.jianshu.io/upload_images/4038437-cad347309409e3b5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/423)  

更深的瓶颈结构：  
![](https://upload-images.jianshu.io/upload_images/6095626-287fc59a3cd86488.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)