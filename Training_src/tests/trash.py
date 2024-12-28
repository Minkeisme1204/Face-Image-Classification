import numpy as np 

x = np.array([1, 2, 3, 4], [5, 6, 7, 8])

np.expand_dims(x, axis=2)

print(x, x.shape)