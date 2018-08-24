import scipy 
import numpy as np   

#---COO（Coordinate Format坐标表示）---#
row_idx = np.array([0,0,1,2,2,3,3,3])
col_idx = np.array([0,3,1,2,3,0,1,3])
values = np.array([4,2,1,5,7,6,3,8])
coo_mat = scipy.sparse.coo_matrix((values, (row_idx , col_idx)),shape = (4,4)).toarray()
'''
array([[4, 0, 0, 2],
       [0, 1, 0, 0],
       [0, 0, 5, 7],
       [6, 3, 0, 8]])
'''
#---CSR（Compressed Sparse Row行压缩）---#
col_idx = np.array([0,3,1,2,3,0,1,3])
values = np.array([4,2,1,5,7,6,3,8])
row_ptr = np.array([0,2,3,5,8])
csr_mat = scipy.sparse.csr_matrix((values,col_idx, row_ptr),shape=(4,4)).toarray()
'''
array([[4, 0, 0, 2],
       [0, 1, 0, 0],
       [0, 0, 5, 7],
       [6, 3, 0, 8]]) 
'''
#---CSC（Compressed Sparse Column列压缩）---#
values = np.array([4,6,1,3,5,2,7,8])
row_idx = np.array([0,3,1,3,2,0,2,3])
col_ptr = np.array([0,2,4,5,8])
csc_mat = scipy.sparse.csc_matrix((values,row_idx,col_ptr),shape=(4,4)).toarray()
'''
array([[4, 0, 0, 2],
       [0, 1, 0, 0],
       [0, 0, 5, 7],
       [6, 3, 0, 8]])
'''


