import cupy as cp

# This package includes activation functions

#  Sigmoid 
def sigmoid(x: cp.ndarray) -> cp.ndarray:
    return 1 / (1 + cp.exp(-x))

def d_sigmoid(x):
    return x * (1 - x)


# Tanh
def tanh(x: cp.ndarray) -> cp.ndarray:
    return (cp.exp(x) - cp.exp(-x)) / (cp.exp(x) + cp.exp(-x))

def d_tanh(x: cp.ndarray) -> cp.ndarray:
    return 1 - (x ** 2)

# relu
def relu(x: cp.ndarray) -> cp.ndarray:
    return cp.maximum(0, x)

def d_relu(x: cp.ndarray) -> cp.ndarray:
    return cp.where(x >= 0, 1, 0)

# Softmax
def softmax(x: cp.ndarray) -> cp.ndarray:
    exps = cp.exp(x - cp.max(x))
    return exps / cp.sum(exps)

def d_softmax(x: cp.ndarray) -> cp.ndarray:
    exps = cp.exp(x - cp.max(x))
    return exps / cp.sum(exps) ** 2

f_activation = {
    "sigmoid": sigmoid, 
    "tanh": tanh,
    "relu": relu,
    "softmax": softmax
}

f_derivative = {
    "sigmoid": d_sigmoid,
    "tanh": d_tanh,
    "relu": d_relu,
    "softmax": d_softmax
}