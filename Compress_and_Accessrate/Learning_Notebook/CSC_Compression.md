# [Comporessed Sparse Column](https://blog.csdn.net/em_dark/article/details/54313539)
```
1 0 4 
0 3 5 
2 0 6 
一个简单的矩阵 
Array(0, 2, 3, 6) 
Array(0, 2, 1, 0, 1, 2) 
Array(1, 2, 3, 4, 5, 6)

Array(1, 2, 3, 4, 5, 6)–>表示按照列依次顺序排列非0元素 
Array(0, 2, 1, 0, 1, 2)–>表示每一列非零元素所在的行号(从0开始) 
Array(0, 2, 3, 6)–>长度为4 代表矩阵有3列, 
第一个元素始终为0, 
第2个元素’2’ 表示第一列有2个非0元素, 
第3个元素为’3’,表示前面2列里有3个非0元素 
第4个元素为’6’,表示前面3列里有6个非0元素
```
https://blog.csdn.net/wangjian1204/article/details/52149199
