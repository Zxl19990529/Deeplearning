#import packages
import numpy as np 
import testCases 
from gc_utils import *

x,y,parameters=testCases.gradient_check_n_test_case()
print(x.shape)
# print(x,y,format(parameters))