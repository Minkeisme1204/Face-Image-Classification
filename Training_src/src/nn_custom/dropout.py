import cupy as cp 

class Dropout(object):
    def __init__(self, keep_prob):
        self.keep_prob = keep_prob
        self.mask = None
    
    def forward(self, x):
        if self.mask is None:
            self.mask = cp.random.uniform(0, 1, size=x.shape) < self.keep_prob
        return x * self.mask / self.keep_prob   
    
    def backward(self, dL_da):
        return dL_da * self.mask / self.keep_prob