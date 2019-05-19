import numpy as np
from numpy import array

a = np.empty((0,0))
b = array([[1,2],[3,4]])
a = np.append(a,b)
a = a.reshape((2,2))
print(a.shape)
print(b.shape)
for i in a:
    print(i)
