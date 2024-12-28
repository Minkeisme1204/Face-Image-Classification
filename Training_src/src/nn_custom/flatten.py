import cupy as cp 

class Flatten(object):
    def __init__(self):
        pass

    def forward(self, x):
        self.original_shape = x.shape
        batch_size, channels, height, width = x.shape
        return cp.reshape(x, (batch_size, channels * height * width))
    
    def backward(self, dL_da):
        return cp.reshape(dL_da, self.original_shape)