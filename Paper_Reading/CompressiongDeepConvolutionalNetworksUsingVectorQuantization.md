# Compressing Deep Convolutional Networks Using Vector Quantization
## Background && Ambition
### Background
The size of CNN models are too large (200M for example) which limit them from being applied to embeded platforms, such as mobile phones. So how to compress parameters to reduce storage requirements.   
This paper mainly consider compressing CNN for computer vision tasks. And it focus on compressing the parameters to **reduce the storage**  . 
### Ambition
As 90% of the storage is taken up by **dense connected layers** , it focus upon how to compress the *dense connected layers* to reduce the storage of neural networks.  
## COntrubution
- Systematically explore vector quantization methods for compressing the **dense connected layer** of deepCNN to reduce storage
- Performed  a comprehensive *evaluation* of different vector quantization methods
- Performed experiments on image retrieval to verify the generalization ability of the compressed model
## Conclusion

## Method
In this paper, the author put forward the concept of Vector Quantization by concluding 4  methods below   
they are:  
- Binarization
- Scalar Quantization(using K-means)
- Product Quantization
- Residual Quantization

# Knowledge Bckground
## [K-means](https://baike.baidu.com/item/K-means/4934806)
K is the number of classes you would want, and the 'means ' means to be the average value.   
The detail:  
- Randomly choose k objects as the original centers.
- Compute every object's distance from the original center, and classify them accroding to the object's closest distance  
- recompute the original centers which have changes.
- Repeat (2),(3) until original centers never change.

