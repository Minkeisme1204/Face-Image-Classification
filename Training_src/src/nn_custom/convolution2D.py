import cupy as cp
import yaml 
from activations import *

# Load parameters from YAML
with open('./Training_src/configs/params.yaml', 'r') as cfg_file:
    params = yaml.safe_load(cfg_file)

batch_size = params.get('TRAINING').get('BATCH_SIZE')

# Convert features map into a COL matrix for calculations
def im2col(feature: cp.array, kernels: cp.array, padding=0, stride=1): 
    """
    Feature map (channels, height, width) -> Column vector (height*width*channels, feature_map_size)
    """
    batch, channels, height, width = feature.shape
    filters, kernel_height, kernel_width = kernels.shape

    # Padding input feature maps
    padwidth = ((0, 0), (0, 0), (padding, padding), (padding, padding))
    x_padded = cp.pad(feature, pad_width=padwidth, mode='constant', constant_values=0)

    output_height = (height + 2*padding - kernel_height)//stride + 1
    output_width = (width + 2*padding - kernel_width)//stride + 1

    # Extract patches from the padded image
    strides = (
        x_padded.strides[0],
        x_padded.strides[1], 
        x_padded.strides[2] * stride, 
        x_padded.strides[3] * stride, 
        x_padded.strides[2], 
        x_padded.strides[3]
    )
    shape = (batch, channels, output_height, output_width, kernel_height, kernel_width)
    sliced = cp.lib.stride_tricks.as_strided(x_padded, shape=shape, strides=strides)
    
    # Target shape is batch, channels*kernel*kernel, output_height*output_width
    sliced = sliced.transpose(0, 1, 4, 5, 2, 3)
    sliced = sliced.reshape(batch, channels*kernel_height*kernel_width, output_height*output_width)

    # Flatten kernels
    flatten_kernels = kernels.reshape(filters, -1)
    flatten_kernels = cp.tile(flatten_kernels, channels)

    return sliced, flatten_kernels
# Tested

# Column vector to input feature map reconstruction function
def col2im(features: cp.array, batch, channels, target_height, target_width) -> cp.array:
    reshaped = features.reshape(batch, channels, target_height, target_width)

    return reshaped
    pass
# Tested

# Initialize paramters function Following Xavier
def xavier_init_kernel(in_filter, filters, kernel=(3, 3)):
    """
    Xavier initialization for kernel weights.
    :param in_features: Number of input features.
    :param filter: Number of output filters.
    :param kernel: Size of the kernel (default is (3, 3)).
    :return: Initialized kernel weights.
    """
    fan_in = in_filter
    limit = cp.sqrt(6 / (fan_in + filters))  # Xavier uniform
    return cp.random.uniform(-limit, limit, size=(filters, kernel[0], kernel[1])).astype(cp.float32)

class Conv2D(object):
    def __init__ (self, in_filter: int, filters: int, kernel=(3, 3), activation="relu", stride=1, padding=0, regularization=False, name=None):
        self.stride = stride
        self.padding = padding 
        self.filters = filters
        self.kernel = xavier_init_kernel(in_filter=in_filter, filters=filters, kernel=kernel) 
        self.bias = xavier_init_kernel(in_filter=in_filter, filters=filters, kernel=(1, 1))
        self.bias = self.bias.reshape(filters, 1)
        # self.kernel = cp.ones(shape=(filters, 3, 3)).reshape(-1, 3, 3)
        # self.bias = cp.arange(0, filters).reshape(filters, 1)
        self.activation = activation
        self.regularization = regularization
        self.name = name
        self.in_filter = in_filter
        pass

    def forward(self, x : cp.ndarray) -> cp.ndarray:
        """
        Phase 1: Processing x, Size of x (channels, height, width)
        """
        batch, channels, height, width = x.shape

        self.input = x
        im2col_x, flatten_kernel = im2col(x, self.kernel, self.padding, self.stride)

        print(im2col_x)
        output = cp.matmul(flatten_kernel, im2col_x)
        # self.z = output + bias
        self.z = output
        output = f_activation[self.activation](output)
        
        output_width = (width + 2*self.padding - self.kernel.shape[-1]) // self.stride + 1
        output_height = (height + 2*self.padding - self.kernel.shape[-2]) // self.stride + 1
        output = output.reshape(batch, self.filters, output_height, output_width)
        print(output.shape)
        return f_activation[self.activation](output)
        pass

    def backward(self, dL_da : cp.array):
        # Gradient with respect to the output of the layer
        da_dz = f_derivative[self.activation](self.z)
        dL_dz = dL_da * da_dz

        # Gradient with respect to weights (kernel)
        im2col_x, flatten_kernel = im2col(self.input, self.kernel, self.padding, self.stride)
        self.dL_dF = cp.matmul(dL_dz, im2col_x).reshape(self.kernel.shape)

        # Gradient with respect to bias
        self.dL_db = cp.sum(dL_dz, axis=1, keepdims=True)

        # Gradient with respect to the input
        dL_dx = cp.matmul(flatten_kernel, dL_dz).reshape(self.input.shape)

        return dL_dx
        pass


batch = 3
filters = 5

features = cp.arange(1, batch*4*5*5 + 1).reshape(batch, 4, 5, -1)
_, infilter, _, _ = features.shape

kernels = cp.ones(shape=(filters, 3, 3)).reshape(-1, 3, 3)
bias = cp.arange(0, filters).reshape(filters, 1)

my_conv = Conv2D(in_filter=infilter, filters=filters, kernel=(3, 3), activation='relu', stride=1, padding=0)

output = my_conv.forward(features)
print(output)

