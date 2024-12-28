import cupy as cp
import yaml 
from activations import *

# Load parameters from YAML
with open('./Training_src/configs/params.yaml', 'r') as cfg_file:
    params = yaml.safe_load(cfg_file)

batch_size = params.get('TRAINING').get('BATCH_SIZE')

def im2col(feature: cp.array, pool_size=(2, 2), padding=0, stride=2): 
    """
    Feature map (channels, height, width) -> Column vector (height*width*channels, feature_map_size)
    """
    batch, channels, height, width = feature.shape

    # Padding input feature maps
    padwidth = ((0, 0), (0, 0), (padding, padding), (padding, padding))
    x_padded = cp.pad(feature, pad_width=padwidth, mode='constant', constant_values=0)

    output_height = (height + 2*padding - pool_size[0])//stride + 1
    output_width = (width + 2*padding - pool_size[1])//stride + 1

    # Extract patches from the padded image
    strides = (
        x_padded.strides[0],
        x_padded.strides[1], 
        x_padded.strides[2] * stride, 
        x_padded.strides[3] * stride, 
        x_padded.strides[2], 
        x_padded.strides[3]
    )
    shape = (batch, channels, output_height, output_width, pool_size[0], pool_size[1])
    sliced = cp.lib.stride_tricks.as_strided(x_padded, shape=shape, strides=strides)
    sliced = sliced.reshape(batch, channels, output_height*output_width, pool_size[0]*pool_size[1])

    print(sliced)
    print(sliced.shape)
    return sliced
# Tested

class MaxPooling2D(object):
    def __init__(self, pool_size=(2, 2), padding=0, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding
        pass

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        """
        x: cp.ndarray of shape (batch_size, channels, height, width)
        """
        # Get input shape 
        self.input = x
        batch, channels, height, width = x.shape

        im2col_x = im2col(x, self.pool_size, self.padding, self.stride)
        pool = cp.max(im2col_x, axis=-1)
        self.mask = cp.argmax(im2col_x, axis=-1)
        self.mask = self.mask.reshape(batch, channels, -1) # Shape of the mask: batch, channels, height_out*width_out
        print("Shape of indices", self.mask.shape)
        print("Mask of indices\n", self.mask)

        out_h = (height + 2*self.padding - self.pool_size[0])//self.stride + 1
        out_w = (width + 2*self.padding - self.pool_size[1])//self.stride + 1
        pool = pool.reshape(batch, channels, out_h, out_w)

        # Reshape the pooled feature map to match the original shape
        return pool

    def backward(self, dL_da: cp.ndarray) -> cp.ndarray:
        batch, channels, out_h, out_w = dL_da.shape
        _, _, in_h, in_w = self.input.shape
        pool_h, pool_w = self.pool_size

        dL_dx = cp.zeros_like(self.input)

        flat_mask = self.mask.ravel()  
     
        b_idx, c_idx, oh_idx, ow_idx = cp.meshgrid(
            cp.arange(batch),
            cp.arange(channels),
            cp.arange(out_w),
            cp.arange(out_h),
            indexing='ij'
        )
        b_idx = b_idx.ravel()
        c_idx = c_idx.ravel()
        oh_idx = oh_idx.ravel()
        ow_idx = ow_idx.ravel()


        local_i = flat_mask // pool_w
        local_j = flat_mask % pool_w

        i = oh_idx * self.stride - self.padding + local_i
        j = ow_idx * self.stride - self.padding + local_j

        
        dL_dx[b_idx, c_idx, i, j] = dL_da[b_idx, c_idx, oh_idx, ow_idx]

        return dL_dx

'''
batch = 3
feature = cp.arange(1, batch*4*5*5 + 1).reshape(batch, 4, 5, -1)

pool = MaxPooling2D(pool_size=(2, 2), stride=2)
pooled_feature = pool.forward(feature)
print("Pooled: \n",pooled_feature)
print(pooled_feature.shape)

dL_da = cp.arange(1, 3*4*2*2 + 1).reshape(3, 4, 2, 2)
dL_dx = pool.backward(dL_da)
print("dL_dx: \n", dL_dx)
print(dL_dx.shape)
'''