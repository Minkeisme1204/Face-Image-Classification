import cupy as cp 

class Adam(object):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, momentum=0.99):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m = cp.zeros_like(params)
            self.v = cp.zeros_like(params)

        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * grads**2

        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        params -= self.lr * m_hat / (cp.sqrt(v_hat) + self.epsilon)
        self.t += 1
        return params
    
class SGD(object):
    def __init__(self, lr=0.01, momentum=0.99):
        self.lr = lr
        self.momentum = momentum
        self.v = None
    
    def update(self, params, grads):
        if self.v is None:
            self.v = cp.zeros_like(params)
        
        self.v = self.momentum * self.v + (1 - self.momentum) * grads
        params -= self.lr * self.v
        return params