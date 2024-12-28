import cupy as cp
from activations import *
import yaml

# Load parameters from YAML
with open('./Training_src/configs/params.yaml', 'r') as cfg_file:
    params = yaml.safe_load(cfg_file)

batch_size = params.get('TRAINING').get('BATCH_SIZE')

def xavier_initialization(shape, gain=1.0):
    """
    Xavier initialization for weights.
    :param shape: Tuple defining the shape of the weight matrix (fan_in, fan_out).
    :param gain: Scaling factor (default is 1.0 for uniform initialization).
    :return: Initialized weights.
    """
    fan_in, fan_out = shape
    limit = gain * cp.sqrt(6 / (fan_in + fan_out))  # Xavier uniform
    return cp.random.uniform(-limit, limit, size=shape).astype(cp.float32)

class FullyConnected:
    def __init__(self, in_features, units, activation="sigmoid", regularization=False, name=None):
        """
        Initializes the Fully Connected Layer.
        :param in_features: Number of input features.
        :param units: Number of neurons in the layer.
        :param activation: Activation function name.
        :param regularization: Whether to use regularization.
        :param name: Optional name for the layer.
        """
        self.units = units
        self.in_features = in_features
        self.weights = xavier_initialization((in_features, units))
        self.bias = xavier_initialization((1, units))
        self.activation = activation
        self.regularization = regularization
        self.name = name

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        """
        Forward pass.
        :param x: Input tensor of shape (batch_size, in_features).
        :return: Activated output tensor of shape (batch_size, units).
        """
        assert x.shape[1] == self.in_features, "Input features mismatch!"
        self.input = x
        self.z = cp.matmul(x, self.weights) + self.bias  # Linear transformation
        return f_activation[self.activation](self.z)  # Apply activation

    def backward(self, dL_da: cp.ndarray) -> cp.ndarray:
        """
        Backward pass.
        :param dL_da: Gradient of loss w.r.t activation (batch_size, units).
        :return: Gradient of loss w.r.t input (batch_size, in_features).
        """
        # Gradient w.r.t z
        dL_dz = dL_da * f_derivative[self.activation](self.z)

        # Gradients w.r.t weights and biases
        self.dL_dw = cp.matmul(self.input.T, dL_dz) / batch_size
        self.dL_db = cp.sum(dL_dz, axis=0, keepdims=True) / batch_size

        # Gradient w.r.t input to propagate back
        dL_dx = cp.matmul(dL_dz, self.weights.T)
        return dL_dx

    

        
